#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Discovery & Ranking v2
Senior AI Engineer (Founding Team) JD ranker.

Architecture: Multi-signal rule-based scorer with behavioral multiplier.
- No GPU, no external API calls, no model downloads
- Runs 100K candidates in ~20s on CPU
- 5 weighted profile components + behavioral multiplier
- Honeypot detection (impossible profiles zeroed out)
- Keyword-stuffer detection (skills without career evidence penalised)
"""

import argparse
import csv
import gzip
import json
import math
from datetime import date, datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# JD Intelligence — extracted from careful reading of job_description.md
# ──────────────────────────────────────────────────────────────────────────────

# Tier 1: Absolutely required — production experience with these
MUST_HAVE_SKILLS = {
    # Embeddings & retrieval
    "embeddings", "sentence transformers", "sentence-transformers",
    "openai embeddings", "bge", "e5", "embedding",
    # Vector databases / hybrid search
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "vector search", "vector database",
    "hybrid search", "dense retrieval", "semantic search",
    # Core ML/NLP
    "information retrieval", "retrieval", "ranking", "reranking", "re-ranking",
    "recommendation systems", "recommendation", "search",
    "nlp", "natural language processing",
    # Evaluation
    "ndcg", "mrr", "map", "a/b testing", "evaluation",
    # Language
    "python",
}

# Tier 2: Nice to have
NICE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms", "fine-tune",
    "xgboost", "lightgbm", "learning to rank", "learning-to-rank",
    "pytorch", "tensorflow", "transformers", "huggingface",
    "rag", "retrieval augmented generation",
    "langchain", "llm", "large language models",
    "distributed systems", "kafka", "spark",
    "bert", "gpt", "t5",
}

# Strong negative: JD explicitly says these are disqualifiers
DISQUALIFIER_TITLES = {
    "marketing manager", "hr manager", "content writer", "sales executive",
    "graphic designer", "accountant", "civil engineer", "mechanical engineer",
    "customer support", "operations manager", "scrum master",
    "product manager", "business analyst", "finance",
}

# JD explicitly names these as problematic if ENTIRE career is there
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "l&t infotech", "ltimindtree",
    "hexaware", "niit", "mindtree",
}

# Positive career keywords in descriptions (production evidence)
PRODUCTION_SIGNALS = [
    "deployed", "production", "real users", "at scale", "serving",
    "vector", "embedding", "retrieval", "ranking", "semantic",
    "recommendation", "search engine", "re-rank", "rerank",
    "latency", "throughput", "index", "faiss", "opensearch",
    "billion", "million queries", "hundred million",
]

# Location fit — JD: Pune/Noida preferred, other metros acceptable
TOP_LOCATIONS = {"pune", "noida"}
OK_LOCATIONS = {"delhi", "ncr", "gurgaon", "gurugram", "hyderabad",
                "mumbai", "bangalore", "bengaluru", "chennai"}

TODAY = date.today()


# ──────────────────────────────────────────────────────────────────────────────
# Honeypot Detection — impossible profiles, not just bad ones
# ──────────────────────────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> bool:
    """
    Returns True if profile is impossible/fabricated.
    ~80 honeypots exist in 100K dataset per spec.
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    total_months = sum(r.get("duration_months", 0) for r in career)

    # 1. Advanced/expert skills claimed with 0 months usage
    zero_dur_experts = sum(
        1 for s in skills
        if s.get("duration_months", 1) == 0
        and s.get("proficiency") in ("advanced", "expert")
    )
    if zero_dur_experts >= 3:
        return True

    # 2. Career history months << stated YOE (timeline impossible)
    if yoe > 4 and total_months > 0 and total_months < yoe * 12 * 0.35:
        return True

    # 3. Very few jobs but total months >> YOE (overlapping fabrication)
    if yoe > 0 and total_months > yoe * 12 * 2.2 and len(career) <= 3:
        return True

    # 4. Career start date too late given YOE
    start_dates = [r["start_date"] for r in career if r.get("start_date")]
    if start_dates:
        earliest_year = int(min(start_dates)[:4])
        expected_start = TODAY.year - yoe
        if earliest_year > expected_start + 6:
            return True

    # 5. Education end year far in the future
    for e in candidate.get("education", []):
        if e.get("end_year", 0) > TODAY.year + 2:
            return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Keyword Stuffer Detection
# ──────────────────────────────────────────────────────────────────────────────

def keyword_stuffer_penalty(candidate: dict) -> float:
    """
    Returns a penalty multiplier 0.0–1.0.
    1.0 = no penalty. Lower = more suspicious.
    Detects profiles that list many AI skills but have no career evidence.
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    all_desc = " ".join(r.get("description", "") for r in career).lower()

    # Count AI skills listed
    ai_skills_listed = sum(
        1 for s in skills
        if any(kw in s["name"].lower() for kw in
               {"embedding", "vector", "retrieval", "rag", "faiss", "pinecone",
                "weaviate", "qdrant", "opensearch", "semantic", "nlp", "bert",
                "transformer", "llm", "ranking", "recommendation"})
    )

    # Count AI evidence in career descriptions
    ai_desc_hits = sum(1 for sig in PRODUCTION_SIGNALS if sig in all_desc)

    # If many AI skills but NO career evidence → stuffer
    if ai_skills_listed >= 5 and ai_desc_hits == 0:
        return 0.25
    if ai_skills_listed >= 3 and ai_desc_hits == 0:
        return 0.45
    return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Scoring Components
# ──────────────────────────────────────────────────────────────────────────────

def score_skills(candidate: dict) -> float:
    """
    0–1 score. Weights skill by proficiency × duration × endorsements.
    Catches keyword stuffers via duration/endorsement trust.
    """
    sdetail = {
        s["name"].lower(): s
        for s in candidate.get("skills", [])
    }
    sset = set(sdetail.keys())

    def skill_trust(skill_name: str) -> float:
        """Trust score for a single skill hit."""
        info = sdetail.get(skill_name, {})
        prof = info.get("proficiency", "intermediate")
        dur = info.get("duration_months", 6)
        endorse = info.get("endorsements", 0)

        prof_w = {"beginner": 0.3, "intermediate": 0.65,
                  "advanced": 0.9, "expert": 1.0}.get(prof, 0.65)
        dur_w = min(dur, 48) / 48.0 if dur > 0 else 0.15
        end_w = min(math.log1p(endorse) / math.log1p(30), 1.0)

        return prof_w * (0.45 + 0.35 * dur_w + 0.20 * end_w)

    # Must-have skills score
    must_score = 0.0
    for skill in MUST_HAVE_SKILLS:
        # Match exact or partial
        matched = skill in sset or any(skill in s for s in sset)
        if matched:
            # Find best matching skill
            best = max(
                (skill_trust(s) for s in sset if skill in s or s in skill),
                default=skill_trust(skill)
            )
            must_score += best

    must_score = min(must_score / max(len(MUST_HAVE_SKILLS) * 0.28, 1), 1.0)

    # Nice-to-have skills
    nice_score = 0.0
    for skill in NICE_SKILLS:
        if skill in sset or any(skill in s for s in sset):
            nice_score += 1
    nice_score = min(nice_score / max(len(NICE_SKILLS) * 0.3, 1), 1.0)

    # Platform assessment scores (verified skill tests)
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    assess_score = 0.0
    if assessments:
        relevant = [
            v for k, v in assessments.items()
            if any(kw in k.lower() for kw in
                   {"ml", "python", "nlp", "search", "retrieval", "embedding", "data"})
        ]
        if relevant:
            assess_score = sum(relevant) / (len(relevant) * 100)

    return 0.55 * must_score + 0.25 * nice_score + 0.20 * assess_score


def score_career(candidate: dict) -> float:
    """
    0–1 score. Checks title fit, company type, production evidence in descriptions.
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "").lower()

    # Hard disqualifier titles
    if any(dis in current_title for dis in DISQUALIFIER_TITLES):
        return 0.03

    # Positive title signals
    good_titles = {
        "ml engineer", "machine learning engineer", "ai engineer",
        "nlp engineer", "search engineer", "applied scientist",
        "research engineer", "data scientist", "senior engineer",
        "software engineer", "backend engineer", "recommendation",
        "applied ml", "staff engineer", "principal engineer",
    }
    title_score = 0.25 if any(kw in current_title for kw in good_titles) else 0.05

    # Services-only penalty
    all_companies = [r.get("company", "").lower() for r in career]
    services_only = len(all_companies) > 0 and all(
        any(svc in co for svc in SERVICES_COMPANIES)
        for co in all_companies if co
    )
    if services_only:
        title_score *= 0.25

    # Product company ratio
    total_months = sum(r.get("duration_months", 0) for r in career)
    product_months = 0
    for r in career:
        months = r.get("duration_months", 0)
        co = r.get("company", "").lower()
        ind = r.get("industry", "").lower()
        if any(svc in co for svc in SERVICES_COMPANIES):
            continue
        if any(kw in ind for kw in {"software", "saas", "ai", "technology",
                                     "internet", "fintech", "edtech"}):
            product_months += months
        elif any(kw in co for kw in {"tech", "ai", "labs", "systems",
                                      "platform", "data", "analytics"}):
            product_months += months * 0.7

    product_ratio = (product_months / total_months) if total_months > 0 else 0
    product_score = 0.35 * product_ratio

    # Production evidence in role descriptions
    all_desc = " ".join(r.get("description", "") for r in career).lower()
    prod_hits = sum(1 for kw in PRODUCTION_SIGNALS if kw in all_desc)
    # Weight more heavily the most relevant signals
    key_hits = sum(1 for kw in ["vector", "embedding", "retrieval", "ranking",
                                  "semantic", "recommendation", "re-rank", "rerank"]
                   if kw in all_desc)
    desc_score = 0.25 * min(prod_hits / 5, 1.0) + 0.15 * min(key_hits / 3, 1.0)

    return min(title_score + product_score + desc_score, 1.0)


def score_experience(candidate: dict) -> float:
    """0–1 score. Optimal: 5–9 yrs. Hard ramps below/above."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        return 1.0
    elif yoe < 3:
        return yoe / 3 * 0.4
    elif yoe < 5:
        return 0.4 + 0.6 * (yoe - 3) / 2
    elif yoe <= 12:
        return 1.0 - 0.25 * (yoe - 9) / 3
    else:
        return max(0.3, 0.75 - 0.1 * (yoe - 12))


def score_location(candidate: dict) -> float:
    """0–1 score. Pune/Noida best, other Indian metros ok, relocatable partial."""
    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    relocate = sig.get("willing_to_relocate", False)

    if any(p in loc for p in TOP_LOCATIONS):
        return 1.0
    if any(p in loc for p in OK_LOCATIONS):
        return 0.85
    if country == "india":
        return 0.70 if relocate else 0.50
    # Outside India
    return 0.25 if relocate else 0.10


def score_education(candidate: dict) -> float:
    """0–1 score. Tier 1 best, unknown is neutral."""
    edu = candidate.get("education", [])
    if not edu:
        return 0.50
    tier_map = {"tier_1": 1.0, "tier_2": 0.80, "tier_3": 0.60,
                "tier_4": 0.40, "unknown": 0.50}
    return max(tier_map.get(e.get("tier", "unknown"), 0.50) for e in edu)


def score_behavioral(candidate: dict) -> float:
    """
    0–1 engagement/availability multiplier.
    A great-on-paper candidate who is inactive/unresponsive ranks lower.
    """
    sig = candidate.get("redrob_signals", {})

    # Availability
    open_flag = 1.0 if sig.get("open_to_work_flag") else 0.55

    # Recency — decays over 12 months
    days_inactive = _days_since(sig.get("last_active_date", "2020-01-01"))
    recency = max(0.0, 1.0 - days_inactive / 365)

    # Responsiveness
    resp_rate = sig.get("recruiter_response_rate", 0.5)
    resp_time = sig.get("avg_response_time_hours", 48)
    responsiveness = resp_rate * max(0, 1 - resp_time / 168)

    # Profile quality
    completeness = sig.get("profile_completeness_score", 50) / 100

    # Technical activity
    github = sig.get("github_activity_score", -1)
    github_score = github / 100 if github >= 0 else 0.30

    # Reliability
    interview_rate = sig.get("interview_completion_rate", 0.7)

    # Notice period — JD loves ≤30 days
    notice = sig.get("notice_period_days", 60)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.75
    elif notice <= 90:
        notice_score = 0.55
    else:
        notice_score = max(0.25, 1.0 - notice / 180)

    # Work mode — hybrid/flexible is ideal per JD
    wm = sig.get("preferred_work_mode", "flexible")
    wm_score = {"hybrid": 1.0, "flexible": 1.0,
                "onsite": 0.80, "remote": 0.65}.get(wm, 0.80)

    # Recruiter demand signals
    saved = min(sig.get("saved_by_recruiters_30d", 0) / 8, 1.0)

    # Trust signals
    verified = (
        0.4 * int(bool(sig.get("verified_email"))) +
        0.4 * int(bool(sig.get("verified_phone"))) +
        0.2 * int(bool(sig.get("linkedin_connected")))
    )

    score = (
        0.20 * open_flag +
        0.18 * recency +
        0.15 * responsiveness +
        0.12 * notice_score +
        0.10 * github_score +
        0.08 * interview_rate +
        0.07 * completeness +
        0.05 * saved +
        0.03 * wm_score +
        0.02 * verified
    )
    return min(score, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Composite Score
# ──────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "skills":     0.38,
    "career":     0.30,
    "experience": 0.14,
    "location":   0.10,
    "education":  0.08,
}

def score_candidate(candidate: dict) -> float:
    """Returns final 0–1 score."""
    if is_honeypot(candidate):
        return 0.0

    s_skills   = score_skills(candidate)
    s_career   = score_career(candidate)
    s_exp      = score_experience(candidate)
    s_loc      = score_location(candidate)
    s_edu      = score_education(candidate)

    profile_score = (
        WEIGHTS["skills"]     * s_skills  +
        WEIGHTS["career"]     * s_career  +
        WEIGHTS["experience"] * s_exp     +
        WEIGHTS["location"]   * s_loc     +
        WEIGHTS["education"]  * s_edu
    )

    # Keyword stuffer penalty (multiplier)
    stuffer_mult = keyword_stuffer_penalty(candidate)

    # Behavioral multiplier: scale to [0.45, 1.0]
    beh = score_behavioral(candidate)
    beh_mult = 0.45 + 0.55 * beh

    return min(profile_score * stuffer_mult * beh_mult, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Reasoning Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, score: float, rank: int) -> str:
    """Specific, honest 1-2 sentence reasoning. No hallucination."""
    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    loc = profile.get("location", "Unknown")
    country = profile.get("country", "")

    sset = {s["name"].lower() for s in candidate.get("skills", [])}
    core_hits = [s for s in MUST_HAVE_SKILLS if s in sset or any(s in sk for sk in sset)]
    nice_hits = [s for s in NICE_SKILLS if s in sset or any(s in sk for sk in sset)]

    all_desc = " ".join(r.get("description", "") for r in career).lower()
    has_prod = any(kw in all_desc for kw in ["deployed","production","at scale","serving"])
    has_retrieval = any(kw in all_desc for kw in ["retrieval","ranking","recommendation","vector","semantic","search"])

    notice = sig.get("notice_period_days", 60)
    resp_rate = sig.get("recruiter_response_rate", 0.5)
    days_inactive = _days_since(sig.get("last_active_date", "2020-01-01"))
    github = sig.get("github_activity_score", -1)
    open_flag = sig.get("open_to_work_flag", False)

    # Sentence 1 — profile
    parts = [f"{title} with {yoe:.1f} yrs experience"]
    if core_hits:
        parts.append(f"core skills matched: {', '.join(core_hits[:4])}")
    if nice_hits:
        parts.append(f"bonus: {', '.join(nice_hits[:2])}")
    if has_retrieval and has_prod:
        parts.append("production retrieval/ranking evidence in career")
    elif has_retrieval:
        parts.append("retrieval/search work in career descriptions")

    s1 = "; ".join(parts) + "."

    # Sentence 2 — signals
    positives, concerns = [], []
    if open_flag: positives.append("open-to-work")
    if days_inactive < 14: positives.append("active recently")
    elif days_inactive > 180: concerns.append(f"inactive {days_inactive}d")
    if notice <= 30: positives.append("notice ≤30d")
    elif notice > 90: concerns.append(f"{notice}d notice")
    if resp_rate >= 0.75: positives.append(f"{int(resp_rate*100)}% recruiter response")
    elif resp_rate < 0.3: concerns.append(f"low response rate ({int(resp_rate*100)}%)")
    if github >= 60: positives.append(f"GitHub score {int(github)}")
    loc_l = loc.lower()
    if any(p in loc_l for p in TOP_LOCATIONS | OK_LOCATIONS):
        positives.append(f"based in {loc}")
    elif country.lower() != "india":
        concerns.append(f"outside India ({loc})")

    if rank > 75:
        tail = " Borderline fit; included at bottom of shortlist."
    else:
        tail = ""

    if positives and concerns:
        s2 = f"Signals — {', '.join(positives[:3])}. Concerns: {', '.join(concerns[:2])}.{tail}"
    elif concerns:
        s2 = f"Concerns: {', '.join(concerns[:3])}.{tail}"
    else:
        s2 = f"Signals — {', '.join(positives[:4]) if positives else 'moderate engagement overall'}.{tail}"

    return f"{s1} {s2}"


# ──────────────────────────────────────────────────────────────────────────────
# I/O Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _days_since(date_str: str) -> int:
    try:
        return (TODAY - datetime.strptime(date_str[:10], "%Y-%m-%d").date()).days
    except Exception:
        return 9999


def load_candidates(path: str):
    p = Path(path)
    opener = (lambda: gzip.open(p, "rt", encoding="utf-8")) if p.suffix == ".gz" \
             else (lambda: open(p, "r", encoding="utf-8"))
    with opener() as f:
        raw = f.read().strip()
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except Exception:
            pass
    candidates = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                candidates.append(json.loads(line))
            except Exception:
                pass
    return candidates


def rank_candidates(candidates):
    scored = []
    for c in candidates:
        s = score_candidate(c)
        scored.append((c["candidate_id"], s, c))
    def sort_key(x):
        try:
            num = int(x[0].split("_")[1])
        except Exception:
            num = 0
        return (-x[1], num)
    scored.sort(key=sort_key)
    return scored


def write_submission(scored, out_path: str):
    top100 = scored[:100]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, candidate) in enumerate(top100, start=1):
            reasoning = generate_reasoning(candidate, score, rank)
            writer.writerow([cid, rank, round(score, 6), reasoning])
    print(f"✅ Wrote {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker v2")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"📂 Loading from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"✅ Loaded {len(candidates):,} candidates")

    print("⚡ Scoring...")
    import time
    t0 = time.time()
    scored = rank_candidates(candidates)
    elapsed = time.time() - t0

    honeypots = sum(1 for _, s, _ in scored if s == 0.0)
    print(f"🕵️  Honeypots detected & zeroed: {honeypots}")
    print(f"🏆 Top 5 scores: {[round(s, 4) for _, s, _ in scored[:5]]}")
    print(f"⏱️  Scoring time: {elapsed:.1f}s")

    write_submission(scored, args.out)


if __name__ == "__main__":
    main()
