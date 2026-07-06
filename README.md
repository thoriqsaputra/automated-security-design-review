# Automated Security Design Review (SDR)

> An AI-powered system for automated security design review of web application architectures, implementing Multimodal LLMs, Hybrid Retrieval, and Multi-Agent Debate to enforce OWASP ASVS compliance.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Configuration](#environment-configuration)
  - [Running with Docker Compose](#running-with-docker-compose)
  - [Running Locally (Development)](#running-locally-development)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Research Context](#research-context)

---

## Overview

The **Automated Security Design Review** system addresses the critical *tooling gap* in Secure Software Development Life Cycle (SSDLC) — where code-level security scanning is mature, but **architectural design review remains largely manual**.

This system automates the review of technical design documents (PDFs) and architecture diagrams (DFDs, cloud infrastructure diagrams) against the **OWASP Application Security Verification Standard (ASVS)**. It is built as a research artifact for a thesis project following the *Design Science Research Methodology* (DSRM).

### Core Problems Solved

| Problem | Solution |
|---|---|
| LLM hallucination &amp; lack of grounding | Dynamic knowledge base injection from OWASP ASVS |
| Lost-in-the-middle on long documents | RAPTOR hierarchical summarization |
| Visual blindness on architecture diagrams | Multimodal vision agents with Set-of-Mark prompting |
| High false-positive rate in single-model inference | Multi-Agent Debate (Hunter → Critic → Mediator) |
| Non-transparent audit findings | Automated Citation Trace linked to ASVS requirements |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Frontend (React + Vite)                     │
│          Design upload · Review dashboard · Report viewer        │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP / REST
┌───────────────────────────────▼─────────────────────────────────┐
│                    Backend API (FastAPI)                          │
│  /designs  ·  /reviews  ·  /standards  ·  /workspace            │
└──────┬──────────────────────────────────────────┬───────────────┘
       │ Celery Tasks                              │ DB / Storage
┌──────▼──────────────────┐           ┌───────────▼───────────────┐
│   AI Engine             │           │  PostgreSQL + pgvector     │
│                         │           │  Redis (broker/cache)      │
│  ① Standard Ingestion   │           │  MinIO (file storage)      │
│     ASVS PDF → chunks   │           └───────────────────────────┘
│     → structured KB     │
│                         │           ┌───────────────────────────┐
│  ② TSD Processing       │           │  Embeddings Service        │
│     PDF → RAPTOR tree   │◄──────────│  (mxbai-embed-large-v1)   │
│     → semantic blocks   │           └───────────────────────────┘
│                         │
│  ③ Hybrid Retrieval     │           ┌───────────────────────────┐
│     Vector + BM25 +     │◄──────────│  LLM APIs                 │
│     RAPTOR → evidence   │           │  (NVIDIA / OpenRouter /    │
│                         │           │   Ollama / RouteLLM)       │
│  ④ Multi-Agent Debate   │           └───────────────────────────┘
│     Hunter → Critic     │
│     → Mediator verdict  │
│                         │
│  ⑤ Vision Analysis      │
│     DFD marker overlay  │
│     → spatial findings  │
└─────────────────────────┘
```

### AI Pipeline Stages

1. **Standard Ingestion** — OWASP ASVS PDF is parsed, semantically chunked, and requirements are extracted per section/subsection/item with LLM-assisted validation. Visual diagram requirements get a `D-V` prefix for routing to the vision agent.

2. **TSD Processing** — Uploaded design documents are parsed via PyMuPDF, OCR (Tesseract), and PDF text extraction. Content is filtered for relevance, split into semantic blocks, and indexed into the RAPTOR tree (up to 4 hierarchy levels).

3. **Hybrid Retrieval** — Three parallel retrieval strategies are merged via *tiered evidence grading*:
   - **Dense vector search** (pgvector + `mxbai-embed-large-v1`) for semantic matching
   - **BM25 keyword search** for exact technical term matching
   - **RAPTOR summaries** for cross-section context recovery

4. **Multi-Agent Debate** — For each ASVS requirement:
   - `Hunter` agent finds evidence of non-compliance (biased toward flagging issues)
   - `Critic` agent validates every citation verbatim against source blocks (can overturn Hunter)
   - `Mediator` agent produces binding final verdict with Citation Trace

5. **Vision Analysis** — Architecture diagrams are preprocessed with *Set-of-Mark* numbered markers overlaid on pixels. A multimodal LLM agent reasons over spatial trust boundaries and references marker IDs as verifiable evidence.

---

## Key Features

- 🔍 **OWASP ASVS-grounded** — Every finding is anchored to a specific ASVS requirement, not free-form LLM opinion
- 🧠 **Multimodal** — Processes both text (TSD/PDF) and visual (DFD/architecture diagrams)
- 🌳 **RAPTOR Retrieval** — Hierarchical summarization prevents context loss on long documents
- ⚖️ **Multi-Agent Debate** — Hunter/Critic/Mediator trio reduces false positives vs. single-model inference
- 📎 **Citation Trace** — Each finding includes verifiable quotes from the source document
- 🔄 **Provider-agnostic LLM** — Supports NVIDIA NIM, OpenRouter, Ollama, and RouteLLM via LiteLLM
- 📊 **Built-in Evaluation Suite** — Ablation studies, retrieval benchmarks, and citation grounding metrics
- 🐳 **Fully Dockerized** — One-command deployment with Docker Compose

---

## Tech Stack

### Backend

| Layer | Technology |
|---|---|
| Framework | FastAPI (Python 3.14) |
| Task Queue | Celery + Redis |
| Database | PostgreSQL 16 + pgvector |
| File Storage | MinIO (S3-compatible) |
| ORM / Migrations | SQLAlchemy + Alembic |
| LLM Orchestration | LiteLLM |
| Embeddings Server | HuggingFace TEI (`mxbai-embed-large-v1`) |
| PDF Processing | PyMuPDF, pdfminer, Tesseract OCR |
| Retrieval | pgvector (dense), rank-bm25 (sparse), RAPTOR (hierarchical) |
| Package Manager | uv |

### Frontend

| Layer | Technology |
|---|---|
| Framework | React 19 + TypeScript |
| Build Tool | Vite |
| Styling | Tailwind CSS v4 |
| Routing | React Router v7 |
| Graph/Diagram | React Flow (`@xyflow/react`), D3.js |
| PDF Viewer | react-pdf |
| HTTP Client | Axios |

### Infrastructure

| Service | Image |
|---|---|
| Reverse Proxy | Nginx Alpine |
| Embeddings | `ghcr.io/huggingface/text-embeddings-inference:cpu-latest` |
| Vector DB | `pgvector/pgvector:pg16` |
| Cache / Broker | `redis:7-alpine` |
| Object Storage | `minio/minio:latest` |

---

## Project Structure

```
automated-security-design-review/
├── backend/
│   ├── sdr/
│   │   ├── apps/
│   │   │   ├── ai/                   # Core AI engine
│   │   │   │   ├── agents/           # Hunter, Critic, Mediator agents
│   │   │   │   ├── engine/           # Pipeline orchestration
│   │   │   │   │   ├── debate/       # Multi-agent debate logic
│   │   │   │   │   ├── extraction/   # Standard & TSD extraction
│   │   │   │   │   ├── preparation/  # Document preprocessing
│   │   │   │   │   ├── reporting/    # Report generation
│   │   │   │   │   └── pipeline.py   # Main AI pipeline
│   │   │   │   ├── evaluations/      # Benchmark & ablation study harnesses
│   │   │   │   ├── prompts/          # All LLM prompt templates
│   │   │   │   ├── retrieval/        # Hybrid retrieval (vector, BM25, RAPTOR)
│   │   │   │   └── tsd_processing/   # PDF/diagram ingestion
│   │   │   ├── designs/              # Design document management API
│   │   │   ├── reviews/              # Security review API
│   │   │   ├── standards/            # OWASP ASVS standards management API
│   │   │   └── workspace/            # Project workspace API
│   │   ├── core/
│   │   │   └── config.py             # Pydantic settings
│   │   ├── main.py                   # FastAPI app factory
│   │   └── celery_app.py             # Celery configuration
│   ├── alembic/                      # DB migrations
│   ├── tests/                        # Pytest test suite
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   └── package.json
├── nginx/
│   └── docker-http.conf
├── docs/                             # Thesis chapters (Indonesian)
├── dataset/                          # Evaluation datasets
├── docker-compose.yml
└── README.md
```

---

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) &amp; [Docker Compose](https://docs.docker.com/compose/) v2+
- An API key from at least one supported LLM provider:
  - [NVIDIA NIM](https://build.nvidia.com/) (`NVIDIA_API_KEY`)
  - [OpenRouter](https://openrouter.ai/) (`OPENROUTER_API_KEY`)

### Environment Configuration

Copy and configure the backend environment file:

```bash
cp backend/.env.example backend/.env
```

Key variables to set in `backend/.env`:

```dotenv
# -- Database --
POSTGRES_DB=sdr
POSTGRES_USER=sdr_user
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql+psycopg://sdr_user:your_secure_password@postgres:5432/sdr

# -- Redis --
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# -- MinIO --
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=password123
MINIO_ENDPOINT=minio:9000
MINIO_BUCKET_NAME=sdr-media

# -- LLM Providers (configure at least one) --
NVIDIA_API_KEY=nvapi-...
OPENROUTER_API_KEY=sk-or-...

# -- Model Selection (defaults use NVIDIA NIM) --
AI_MODEL_HUNTER=meta/llama-3.1-70b-instruct
AI_MODEL_CRITIC=meta/llama-3.1-70b-instruct
AI_MODEL_MEDIATOR=meta/llama-3.1-70b-instruct
AI_MODEL_VISION=meta/llama-3.2-90b-vision-instruct
AI_MODEL_EMBEDDING=nvidia/nv-embedqa-e5-v5

# -- Security --
SECRET_KEY=your_secret_key_here
ENVIRONMENT=prod
```

### Running with Docker Compose

```bash
# Pull and start all services
docker compose up -d

# Check service health
docker compose ps

# View backend logs
docker compose logs -f backend

# Run database migrations (first-time setup)
docker compose exec backend alembic upgrade head
```

The application will be available at:
- **Frontend / API**: `http://localhost` (via Nginx)
- **API Docs**: `http://localhost/docs` (dev mode only)
- **MinIO Console**: `http://localhost:9001`

### Running Locally (Development)

**Backend:**

```bash
cd backend

# Install uv (if not installed)
pip install uv

# Create virtualenv and install dependencies
uv sync

# Set up environment
cp .env.example .env  # then edit .env

# Run database migrations
uv run alembic upgrade head

# Start the API server
uv run uvicorn sdr.main:app --reload --host 0.0.0.0 --port 8000

# Start the Celery worker (in a separate terminal)
uv run celery -A sdr.celery_app worker --loglevel=info --queues=sdr_analysis,celery
```

**Frontend:**

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend dev server runs at `http://localhost:5173`.

---

## API Reference

The REST API is versioned under `/api/v1/`. Interactive documentation is available at `/docs` (Swagger UI) and `/redoc` in non-production environments.

| Router | Prefix | Description |
|---|---|---|
| Designs | `/api/v1/designs` | Upload and manage TSD/architecture documents |
| Reviews | `/api/v1/reviews` | Trigger and retrieve security review results |
| Standards | `/api/v1/standards` | Manage OWASP ASVS knowledge base |
| Workspace | `/api/v1/workspace` | Project workspace management |
| Health | `/health` | Service liveness probe |

---

## Evaluation

The system includes a built-in evaluation suite (`backend/sdr/apps/ai/evaluations/`) covering four domains:

| Domain | Description | Key Metrics |
|---|---|---|
| **Standard Extraction** | How completely the ASVS pipeline captures requirements | Extraction Recall, Precision, F1 per chapter |
| **Retrieval** | Lost-in-the-middle test (front/middle/back zones) and strategy ablation | Context Recall, Faithfulness (vector-only vs. hybrid) |
| **Multi-Agent Debate** | Single-Hunter vs. full 3-agent verdict; citation audit | False Positive Rate, Critic Intervention Rate, Citation Grounding Rate |
| **Vision** | Spatial detection of missing controls from diagrams only (no text) | Marker ID accuracy, Visual Grounding Rate |

---

## Research Context

This system is developed as a **thesis research artifact** (Tugas Akhir) at an Indonesian university, following the *Design Science Research Methodology* (DSRM) by Peffers et al.

**Research Questions:**
1. How effective is dynamic knowledge base injection (OWASP ASVS) in eliminating LLM hallucination without model retraining?
2. How does multimodal processing + hierarchical hybrid retrieval (RAPTOR) mitigate *lost-in-the-middle* and visual blindness in detecting cross-component threats?
3. How effective is Multi-Agent Debate (with Chain-of-Thought reasoning) at reducing false positives and producing transparent Citation Traces compared to single-model inference?

**Key Research Innovations:**
- Tiered evidence grading for hybrid retrieval fusion
- Set-of-Mark visual prompting for architecture diagram analysis
- Autonomous citation verification by the Critic agent (not relying on LLM self-assessment)
- Deterministic debate analytics computed from stored trace data without extra LLM calls

---

*This project is a research prototype. It is not intended for production security auditing without expert human review of AI-generated findings.*
