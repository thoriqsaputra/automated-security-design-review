# Evaluation Suite

Evaluation scripts for the automated security design review system, covering the full pipeline:
standard extraction → retrieval → multi-agent debate.

## Directory structure

```
evaluations/
├── extraction/          Standard ingestion quality
├── debate/              Multi-agent debate quality
├── retrieval/           RAG retrieval quality
├── vision/              Vision agent quality
├── shared/              Shared judges, metrics, dataset utils
├── data/                Ground truth files and gold ID sets
└── results/             ← All output files land here (host-mounted volume)
```

## Prerequisites

All scripts run inside the backend Docker container. The `results/` directory is mounted
from the host so output files appear locally after every run.

**Check your current ingestion job IDs before running:**

```bash
docker exec automated-security-design-review-backend-1 python -c "
import sys; sys.path.insert(0, '/app')
from sdr.core.database import SessionLocal
from sdr.apps.standards.models import StandardIngestionJob, StandardSourceDocument, CategoryParameterParent, CategoryParameterChild
with SessionLocal() as db:
    for j in db.query(StandardIngestionJob).order_by(StandardIngestionJob.id).all():
        docs = db.query(StandardSourceDocument).filter_by(ingestion_job_id=j.id).all()
        pids = [p.id for p in db.query(CategoryParameterParent).filter_by(ingestion_job_id=j.id).all()]
        rc = db.query(CategoryParameterChild).filter(CategoryParameterChild.parent_id.in_(pids)).count() if pids else 0
        print(f'job={j.id} active={j.is_active} reqs={rc} doc={docs[0].name[:60] if docs else \"?\"}')
"
```

Current state (update when you re-ingest):

| Job | Standard        | Active | Requirements |
|-----|-----------------|--------|--------------|
| 27  | ASVS 4.0.3      | No     | 278          |
| 28  | ASVS 5.0        | Yes    | 341          |

---

## Extraction evaluations

These three scripts validate the standard ingestion pipeline. Run them after every re-ingestion.
Always pass `--job-id` for a specific version, or `--active-only` for the current active job.

### 1. Purity — schema quality

Checks that every extracted row has a valid control ID, no duplicates, no blank rows.

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/extraction/purity_eval.py --job-id 27
```

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `control_id_purity` | ≥ 0.99 | Fraction of rows with a valid X.Y.Z control ID |
| `duplicate_rate` | ≤ 0.01 | Fraction of rows sharing an ID with another row |
| `empty_rate` | ≤ 0.01 | Fraction of rows with blank requirement text |

Output: `results/eval_extraction_purity.json`

---

### 2. Coverage — recall vs gold set

Measures how many of the 286 known ASVS 4.0.3 controls were captured.
Only meaningful against `--job-id 27` (ASVS 4.0.3); ASVS 5.0 has a different control set.

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/extraction/coverage_eval.py --job-id 27

# Filter to specific chapters only:
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/extraction/coverage_eval.py --job-id 27 \
  --chapter V1 --chapter V9
```

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `extraction_recall` | ≥ 0.90 | |extracted ∩ gold| / |gold| |
| `extraction_precision` | ≥ 0.95 | |extracted ∩ gold| / |extracted| |
| `f1_score` | ≥ 0.90 | Harmonic mean |

Output: `results/eval_extraction_coverage.json`

---

### 3. Category accuracy — LLM tagging quality

Compares `requirement_category` tags (design / code / infrastructure / process) against
53 hand-labeled controls in `data/extraction_ground_truth.json`.

The `design` recall is the most critical metric because only `design`-tagged requirements
enter the downstream debate pipeline.

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/extraction/category_eval.py --job-id 27

# Add LLM judge explanations for each mismatch (costs API tokens):
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/extraction/category_eval.py --job-id 27 \
  --explain-mismatches
```

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `accuracy` | ≥ 0.80 | Overall label match rate |
| `design_recall` | ≥ 0.85 | Fraction of true-design items correctly tagged (gating metric) |
| `match_rate` | ≥ 0.90 | Fraction of GT items found in DB (extraction completeness) |

Output: `results/eval_extraction_category.json`

---

## Debate evaluations

These three scripts evaluate the multi-agent debate pipeline (Hunter → Critic → Mediator).
They require at least one completed security design review in the database.

**Find your review ID:**

```bash
docker exec automated-security-design-review-backend-1 python -c "
import sys; sys.path.insert(0, '/app')
import sdr.apps.standards.models, sdr.apps.designs.models
import sdr.apps.reviews.models.review
from sdr.core.database import SessionLocal
from sdr.apps.reviews.models.review import Review
with SessionLocal() as db:
    for r in db.query(Review).all():
        print(f'review_id={r.id} status={r.status} created={str(r.created_at)[:16]}')
"
```

---

### Exp 6 — Ablation: Hunter-only vs 3-agent debate (FPR)

Proves the Critic + Mediator loop reduces the False Positive Rate compared to Hunter alone.
Requires manual ground-truth labels (see below).

**Step 1 — fill in ground truth labels:**

Edit `data/debate_ground_truth.json` with your finding IDs and labels:

```json
{
  "review_id": 1,
  "items": [
    { "finding_id": 1, "label": "met",     "notes": "TSD documents the control" },
    { "finding_id": 2, "label": "not_met", "notes": "no TSD evidence found" },
    { "finding_id": 3, "label": "na",      "notes": "out of TSD scope" }
  ]
}
```

Labeling rubric: `met` = requirement satisfied per TSD evidence | `not_met` = absent/violated |
`na` = cannot evaluate from TSD (exclude from FPR calculation). Aim for ≥10 met + 10 not_met.

**Step 2 — run the ablation:**

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/debate/ablation_eval.py \
  --review-id 1 \
  --ground-truth /app/sdr/apps/ai/evaluations/data/debate_ground_truth.json
```

| Metric | Target | Meaning |
|--------|--------|---------|
| `hunter_only.fpr` | higher | FPR without Critic/Mediator |
| `debate_final.fpr` | lower | FPR with full 3-agent debate |
| `delta_fpr` | > 0 | FPR suppression achieved by debate |

Output: `results/debate_ablation_results.json`

---

### Exp 7 — Citation trace accuracy

Proves citations are not hallucinated — every cited block exists in the TSD source
and the quoted text verbatim-matches the block content.

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/debate/citation_trace_eval.py \
  --review-id 1

# Adjust manual sample size (default 30):
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/debate/citation_trace_eval.py \
  --review-id 1 --sample-size 20
```

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `block_existence_rate` | ≥ 0.95 | % of cited block_ids that exist in TSD index |
| `quote_grounding_rate` | ≥ 0.80 | % where quoted_text substring-matches block content |
| `citation_coverage_rate` | ≥ 0.80 | % of findings that have ≥1 valid citation |

Output: `results/citation_audit.json` — includes a `manual_validation_sample` list
(N=30) for the human-reviewer relevance check (Exp 7 Layer 3).

---

### Exp 8 — Debate dynamics (Critic effectiveness)

Proves the Critic is not a rubber stamp. Fully automated — no manual labels needed.

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/debate/dynamics_eval.py \
  --review-id 1
```

| Metric | Target | Meaning |
|--------|--------|---------|
| `critic_intervention_rate` | > 30% | % of debates where Critic outcome ≠ UPHOLD |
| `verdict_revision_rate` | > 10% | % where Critic changed its own revised verdict |
| `avg_citation_rejection_rate` | > 0 | Avg invalid citations per debate / total Hunter citations |
| `escalation_rate` | any | % of debates that ran > 1 round |
| `avg_confidence_delta` | any | Mediator confidence − Hunter initial confidence |

Output: `results/debate_dynamics.json`

---

## Retrieval evaluations

These require a prepared TSD design and a full evaluation dataset. They are more expensive
(LLM judge calls) and are run on-demand, not after every ingestion.

### Vector-only vs Hybrid ablation

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/retrieval/ablation_eval.py \
  --design-id 1 --category-code web_application --job-id 27
```

### Cross-boundary (V1 Architecture + V9 Communications)

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/retrieval/cross_boundary_eval.py \
  --design-id 1 --job-id 27
```

---

## Vision evaluations

### Blindness mitigation

Tests whether the vision debate detects a control's absence from synthetic diagrams
(no text context at all — pure visual detection).

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/vision/blindness_eval.py
```

---

## Results

All output files are written to `results/` which is bind-mounted from:
```
backend/sdr/apps/ai/evaluations/results/   ← host
/app/sdr/apps/ai/evaluations/results/      ← container
```

The mount is defined in `docker-compose.yml`. If you just added it, apply with:
```bash
docker-compose up -d backend
```

After that, results appear on the host automatically after every eval run — no `docker cp` needed.

**Until you restart** (container running without the mount), copy manually:
```bash
docker cp automated-security-design-review-backend-1:/app/sdr/apps/ai/evaluations/results/. \
  backend/sdr/apps/ai/evaluations/results/
```

---

## Quick reference — full extraction re-check

Run after every ASVS re-ingestion to confirm all thresholds pass:

```bash
JOB=27  # ASVS 4.0.3
for script in purity_eval coverage_eval category_eval; do
  docker exec automated-security-design-review-backend-1 \
    python /app/sdr/apps/ai/evaluations/extraction/${script}.py --job-id $JOB
done
```

Expected: all six threshold checks return `True`.
