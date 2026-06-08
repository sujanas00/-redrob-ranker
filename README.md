# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

## Approach

**Rule-based multi-signal ranker with behavioral multiplier.**

No GPU. No LLM API calls. Runs in < 2 minutes on 100K candidates on a standard CPU laptop.

### Scoring Architecture

Five profile components weighted and combined:

| Component | Weight | Description |
|-----------|--------|-------------|
| Skills Match | 40% | Core required skills (embeddings, vector DB, retrieval, NLP, Python) + nice-to-have (LLM fine-tuning, LTR). Each skill scored by proficiency, usage duration, and endorsements to detect keyword stuffers. |
| Career Trajectory | 30% | Product-company vs services-company ratio, current title relevance, production deployment keywords in role descriptions. Services-only careers (TCS, Infosys, etc.) penalised per JD. |
| Experience Fit | 15% | Optimal band 5–9 yrs per JD. Soft ramp-in below and ramp-out above. |
| Location Fit | 10% | Pune/Noida/Delhi/Hyderabad/Mumbai/Bangalore preferred. India + willing to relocate partial credit. |
| Education | 5% | Institution tier (tier_1 → tier_4) as a weak signal. |

**Behavioral multiplier** (0.5–1.0 applied to profile score):
- Recency of last login
- Open-to-work flag
- Recruiter response rate and response time
- GitHub activity score
- Interview completion rate
- Notice period fit
- Work mode preference

### Honeypot Detection

Automatically penalises profiles with:
- 5+ advanced/expert skills with 0 months of usage
- Career history months wildly inconsistent with stated YOE
- Implausible endorsement counts across too many skills

### Why No Embeddings?

The submission spec requires: ≤5 min wall-clock, ≤16GB RAM, CPU only, no network.

Running sentence-transformer embeddings for 100K candidates takes 15–30 min on CPU. This rule-based approach scores 100K candidates in ~90 seconds with no model download required and no external dependencies beyond pandas.

The scoring logic embeds the JD semantics directly: skill aliases are hand-matched, career description keywords are production-signal keywords from the JD, and negative signals (services companies, wrong titles) come directly from the JD's disqualifier list.

---

## Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Run ranker
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate output
python validate_submission.py submission.csv
```

For gzipped file:
```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

## File structure

```
rank.py                  # Main ranker (single file, no external models)
requirements.txt         # Dependencies
validate_submission.py   # Format validator (from hackathon bundle)
submission_metadata.yaml # Submission metadata
README.md                # This file
```

## Compute environment

Tested on standard CPU laptop, Python 3.11, 16GB RAM.
Runtime for 100K candidates: ~90 seconds.
No GPU required. No external API calls.
