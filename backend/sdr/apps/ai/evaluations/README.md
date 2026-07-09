# Evaluation Suite

Evaluation scripts for the automated security design review system, covering the full pipeline:
standard extraction → retrieval → multi-agent debate.

## Directory structure

```
evaluations/
├── extraction/          Standard ingestion quality
├── debate/              Multi-agent debate quality
├── retrieval/           RAG retrieval quality
├── vision/              Shared real-diagram sourcing (used by retrieval/debate diagram evals)
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
| 53  | ASVS 4.0.3      | Yes    | 278          |
| 55  | ASVS 5.0        | No     | 345          |

---

## Extraction evaluations

These three scripts validate the standard ingestion pipeline. Run them after every re-ingestion.
Always pass `--job-id` for a specific version, or `--active-only` for the current active job.

### 1. Purity — schema quality

Checks that every extracted row has a valid control ID, no duplicates, no blank rows.

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/extraction/purity_eval.py --job-id 53
```

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `control_id_purity` | ≥ 0.99 | Fraction of rows with a valid X.Y.Z control ID |
| `duplicate_rate` | ≤ 0.01 | Fraction of rows sharing an ID with another row |
| `empty_rate` | ≤ 0.01 | Fraction of rows with blank requirement text |

Output: `results/eval_extraction_purity.json`

---

### 2. Coverage — recall vs gold set

Measures how many known controls from a curated gold set were captured.
Use the matching bundled gold set for the ingestion version under test.

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/extraction/coverage_eval.py --job-id 53 \
  --gold-set /app/sdr/apps/ai/evaluations/data/asvs_403_gold_ids.json

docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/extraction/coverage_eval.py --job-id 55 \
  --gold-set /app/sdr/apps/ai/evaluations/data/asvs_500_gold_ids.json

# Filter to specific chapters only:
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/extraction/coverage_eval.py --job-id 53 \
  --gold-set /app/sdr/apps/ai/evaluations/data/asvs_403_gold_ids.json \
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
  python /app/sdr/apps/ai/evaluations/extraction/category_eval.py --job-id 53

# Add LLM judge explanations for each mismatch (costs API tokens):
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/extraction/category_eval.py --job-id 53 \
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

The repository currently contains these retrieval eval scripts:

- `retrieval/ablation_eval.py` — 4-way ablation: `vector_only`, `raptor_low`,
  `raptor_only`, `hybrid`
- `retrieval/lost_in_middle_eval.py` — balanced front/middle/back evaluation
  for middle-zone recovery
- `retrieval/diagram_retrieval_eval.py` — diagram requirement retrieval:
  production vector selector vs naive fallback

The previously documented `retrieval/cross_boundary_eval.py` is not present in
this repository and should not be referenced as a runnable evaluation.

### Ablation: vector-only vs RAPTOR vs hybrid

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/retrieval/ablation_eval.py \
  --design-id 8 --dataset eval_dataset_30_design8.json \
  --output eval_ablation_retrieval_design8.json
```

Run once per prepared TSD design with its matching dataset:

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/retrieval/ablation_eval.py \
  --design-id 9 --dataset eval_dataset_30_design9.json \
  --output eval_ablation_retrieval_design9.json

docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/retrieval/ablation_eval.py \
  --design-id 10 --dataset eval_dataset_30_design10.json \
  --output eval_ablation_retrieval_design10.json
```

The ablation datasets now live under `evaluations/data/`. Passing the bare
filename is sufficient because the script resolves it from that shared data
directory automatically.

Canonical retained outputs:

- `results/retrieval/eval_ablation_retrieval_design8.json`
- `results/retrieval/eval_ablation_retrieval_design9.json`
- `results/retrieval/eval_ablation_retrieval_design10.json`

### Lost-in-the-middle

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/retrieval/lost_in_middle_eval.py \
  --design-id 8 --samples-per-zone 10 \
  --output eval_lost_in_middle_design8.json

docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/retrieval/lost_in_middle_eval.py \
  --design-id 9 --samples-per-zone 10 \
  --output eval_lost_in_middle_design9.json

docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/retrieval/lost_in_middle_eval.py \
  --design-id 10 --samples-per-zone 10 \
  --output eval_lost_in_middle_design10.json
```

Canonical retained outputs:

- `results/retrieval/eval_lost_in_middle_design8.json`
- `results/retrieval/eval_lost_in_middle_design9.json`
- `results/retrieval/eval_lost_in_middle_design10.json`

Note on thesis metrics:

- `raptor_middle_recovery` is always interpretable
- `middle_deficit_reduction_pct` is only reported when the vector-only baseline
  actually has a positive middle-zone deficit. If the baseline deficit is `<= 0`,
  the script reports `n/a` instead of a misleading negative or undefined percentage.

---

## Diagram evaluations (real diagrams, ground-truth-backed)

Two scripts, both consuming the same real-diagram ground truth (Step 1 below):
retrieval quality (Tujuan 2 — hybrid retrieval mitigating lost-in-the-middle for
diagram requirements) and multi-agent debate ablation (Tujuan 2 + 3 — multimodal
debate mitigating single-model visual blindness and suppressing false positives
vs. single-model inference).

### Step 1 — Build the ground-truth template

Requires at least one completed, vision-enabled review with diagram findings
(`analysis_mode` other than `text_only`).

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/data/build_diagram_ground_truth_template.py \
  --review-id 1
```

This downloads every diagram's marked (Set-of-Mark) image to
`results/vision/ground_truth_images/review_1/` and writes
`data/diagram_ground_truth_review_1.json`, listing — for every diagram — the **full**
candidate pool of diagram requirements for the review's category (not just the ones the
system happened to select). Open each image, then for every `candidate_requirements`
entry set:

- `"relevant": true/false` — is this requirement genuinely checkable from this diagram?
- `"label": "met"|"not_met"|"na"` — only for `relevant: true` rows.

**LLM-as-judge mode** — add `--llm-judge` to auto-populate `"relevant"` (plus a
`"judge_reasoning"` field) via a vision LLM call (`component=eval_judge`, i.e.
`AI_MODEL_EVAL_JUDGE` — deliberately a different model family from Hunter/Critic/
Mediator, so this isn't the system grading its own homework) instead of manual review.
Costs one vision LLM call per `(diagram, candidate requirement)` row — e.g. 3 diagrams
× 41 candidates ≈ 123 calls for a typical review. Writes to
`diagram_ground_truth_review_<id>_llm_judged.json` by default (not the plain
`diagram_ground_truth_review_<id>.json` name) so it never collides with or overwrites
a hand-labeled file:

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/data/build_diagram_ground_truth_template.py \
  --review-id 1 --llm-judge
```

`"label"` still requires manual review either way — the judge only covers relevance
scoping, not met/not_met verdicts.

This one labeled file feeds both evals below.

### Step 2a — Diagram requirement retrieval eval

Compares the production vector-search selector (`DiagramRequirementSelector`, cosine
search over `CategoryDiagramRequirementEmbedding`) against the naive ordinal fallback
production uses when embedding/search fails.

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/retrieval/diagram_retrieval_eval.py \
  --design-id 1 --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_review_1.json
```

| Metric | Meaning |
|--------|---------|
| `precision` | \|retrieved ∩ relevant\| / \|retrieved\| |
| `recall` | \|retrieved ∩ relevant\| / \|relevant\| |
| `hit_rate` | fraction of diagrams with ≥1 relevant requirement retrieved |
| `mrr` | 1/rank of the first relevant requirement in the ranked list |

Output: `results/retrieval/eval_diagram_retrieval.json`

### Step 2b — Diagram debate ablation (Hunter-only vs full debate)

Same FPR-suppression thesis as Exp 6 above, but matched at `(diagram_id, requirement_id)`
granularity since one diagram finding can cover multiple requirements at once. Runs
entirely off data already stored in the database (`Finding.requirement_metadata`
from a completed, vision-enabled review) — no LLM calls, free to re-run.

```bash
docker exec automated-security-design-review-backend-1 \
  python /app/sdr/apps/ai/evaluations/debate/diagram_ablation_eval.py \
  --review-id 1 --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_review_1.json
```

| Metric | Target | Meaning |
|--------|--------|---------|
| `hunter_only.fpr` | higher | FPR without Critic/Mediator |
| `debate_final.fpr` | lower | FPR with full 3-agent debate |
| `delta_fpr` | > 0 | FPR suppression achieved by debate (Tujuan 3) |
| `blindness_mitigation_rate` | > 0 | fraction of matched pairs where a wrong Hunter-only (single vision model) verdict was corrected by the debate (Tujuan 2 — visual blindness mitigation) |
| `blindness_mitigation_cases` | — | the actual `(diagram_id, requirement_id)` rows behind that rate, for qualitative citation in the thesis |

Output: `results/debate/diagram_debate_ablation_results.json`

---

## Threats to validity

Both diagram evals currently draw on a single design/review (design 15 /
review 53), which has 53 usable met/not_met-labeled `(diagram, requirement)`
samples (12 `met`, 41 `not_met`). This is a real generalizability limit —
results have not been confirmed to hold across other designs or diagram
styles. Extending coverage requires running
`data/build_diagram_ground_truth_template.py` against another completed,
vision-enabled review and manually labeling the resulting candidates; it is
not something that can be scripted away.

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
