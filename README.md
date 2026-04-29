# StartupInvestorMatcher

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-enabled-orange)
![Pytest](https://img.shields.io/badge/tests-43%20passed-success)
![Status](https://img.shields.io/badge/status-MVP-success)

Production-style startup-investor matching service built with Python, FastAPI, semantic embeddings, cosine retrieval, and explainable reranking.

## Features

- Semantic startup-investor retrieval with sentence embeddings
- Cosine similarity candidate generation
- Weighted rule-based reranking with explainable score breakdowns
- Demo-ready FastAPI endpoints with default sample data
- Precomputed investor embedding index with runtime fallback
- Offline benchmark flow for retrieval vs reranking
- Synthetic dataset generation for larger demo and evaluation runs
- Baseline entity resolution and deduplication utilities

## Tech Stack

- Python
- FastAPI
- pandas
- numpy
- scikit-learn
- sentence-transformers
- pydantic
- pytest

## API Endpoints

### `GET /health`

Health check endpoint.

Response:

```json
{
  "status": "ok"
}
```

### `GET /startups`

Returns startup records from `data/examples/startups.sample.csv`.

### `GET /investors`

Returns investor records from `data/examples/investors.sample.csv`.

### `POST /match`

Matches a custom startup payload against a provided investor list.

Request:

```json
{
  "startup": {
    "startup_id": "s1",
    "name": "Acme AI",
    "description": "AI tooling for diligence",
    "industries": ["AI"],
    "stage": "Seed",
    "country": "US",
    "region": "North America",
    "fundraising_amount": 1000000,
    "currency": "USD"
  },
  "investors": [
    {
      "investor_id": "i1",
      "name": "North Star Ventures",
      "description": "AI-focused seed fund",
      "industries": ["AI"],
      "preferred_stages": ["Seed"],
      "countries": ["US"],
      "regions": ["North America"],
      "ticket_min": 250000,
      "ticket_max": 2000000,
      "currency": "USD"
    }
  ],
  "top_k": 5,
  "candidate_pool_size": 25
}
```

Response:

```json
{
  "matches": [
    {
      "investor_id": "i1",
      "investor_name": "North Star Ventures",
      "rank": 1,
      "final_score": 0.97,
      "score_breakdown": {
        "semantic_similarity": 0.91,
        "industry_match": 1.0,
        "stage_match": 1.0,
        "geo_match": 1.0,
        "ticket_size_fit": 1.0,
        "weighted_score": 0.97,
        "matched_industries": ["AI"],
        "reasons": [
          "Strong semantic similarity",
          "Industry overlap: AI",
          "Stage preference match",
          "Geographic focus match",
          "Ticket size fit"
        ]
      },
      "reasons": [
        "Strong semantic similarity",
        "Industry overlap: AI",
        "Stage preference match",
        "Geographic focus match",
        "Ticket size fit"
      ]
    }
  ]
}
```

### `GET /match/{startup_id}`

Matches a startup from the sample catalog against the sample investor catalog.

Example:

```text
GET /match/s1?top_k=2&candidate_pool_size=4
```

## Run Locally

### Clone Repository

```bash
git clone https://github.com/r0st-ma1n/StartupInvestorMatcher.git
cd StartupInvestorMatcher
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Server

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Example Requests

Health check:

```bash
curl "http://127.0.0.1:8000/health"
```

Match startup from sample catalog:

```bash
curl "http://127.0.0.1:8000/match/s1?top_k=2&candidate_pool_size=4"
```

Match custom startup:

```bash
curl -X POST "http://127.0.0.1:8000/match" ^
  -H "Content-Type: application/json" ^
  -d "{\"startup\":{\"startup_id\":\"s1\",\"name\":\"Acme AI\",\"description\":\"AI tooling for diligence\",\"industries\":[\"AI\"],\"stage\":\"Seed\",\"country\":\"US\",\"region\":\"North America\",\"fundraising_amount\":1000000,\"currency\":\"USD\"},\"investors\":[{\"investor_id\":\"i1\",\"name\":\"North Star Ventures\",\"description\":\"AI-focused seed fund\",\"industries\":[\"AI\"],\"preferred_stages\":[\"Seed\"],\"countries\":[\"US\"],\"regions\":[\"North America\"],\"ticket_min\":250000,\"ticket_max\":2000000,\"currency\":\"USD\"}],\"top_k\":1,\"candidate_pool_size\":5}"
```

## Benchmark and Data Generation

### Build Precomputed Investor Index

```bash
python scripts/build_index.py
```

Artifact:

```text
data/artifacts/investor_index.npz
```

API behavior:

- uses precomputed investor embeddings when the artifact matches the current investor catalog and model
- falls back to in-memory embedding generation when the artifact is missing or stale

Production tradeoff:

- this removes repeated investor embedding work from request time
- it does not yet provide background refresh, remote artifact storage, or multi-worker coordination

### Run Benchmark

Sample data:

```bash
python scripts/run_benchmark.py --k 2
```

Generated data:

```bash
python scripts/run_benchmark.py --k 5 --startups-path data/generated/startups.csv --investors-path data/generated/investors.csv --ground-truth-path data/generated/ground_truth.csv
```

Compared pipelines:

- semantic-only retrieval
- retrieval + rule-based reranking

Metrics:

- `precision@k`
- `recall@k`
- `hit-rate@k`
- `MRR@k`

### Generate Synthetic Dataset

Generate a small dataset:

```bash
python scripts/generate_synthetic_data.py --size small --seed 42
```

Overwrite an existing generated dataset:

```bash
python scripts/generate_synthetic_data.py --size medium --seed 42 --overwrite
```

Generated files:

- `data/generated/startups.csv`
- `data/generated/investors.csv`
- `data/generated/ground_truth.csv`

## Testing

Run tests:

```bash
python -m pytest
```

Validate syntax:

```bash
python -m py_compile app/main.py scripts/run_benchmark.py scripts/build_index.py scripts/generate_synthetic_data.py
```

## CSV Schemas

### `startups.csv`

| column | required | description |
| --- | --- | --- |
| `startup_id` | yes | Unique startup identifier |
| `name` | yes | Startup name |
| `description` | yes | Free-text startup description |
| `industries` | yes | Comma-separated industries |
| `stage` | yes | Funding stage such as `Pre-Seed`, `Seed`, `Series A` |
| `country` | yes | Country focus |
| `region` | yes | Region focus |
| `fundraising_amount` | yes | Target raise amount |
| `currency` | yes | Currency code |
| `website` | no | Optional startup website |

### `investors.csv`

| column | required | description |
| --- | --- | --- |
| `investor_id` | yes | Unique investor identifier |
| `name` | yes | Investor or fund name |
| `description` | yes | Free-text investor description |
| `industries` | yes | Comma-separated target industries |
| `preferred_stages` | yes | Comma-separated preferred startup stages |
| `countries` | yes | Comma-separated country focus |
| `regions` | yes | Comma-separated region focus |
| `ticket_min` | yes | Minimum check size |
| `ticket_max` | yes | Maximum check size |
| `currency` | yes | Currency code |
| `investor_type` | no | Optional type such as `VC`, `Angel`, `CVC` |
| `website` | no | Optional investor website |

### `ground_truth.csv`

| column | required | description |
| --- | --- | --- |
| `startup_id` | yes | Startup identifier |
| `investor_id` | yes | Relevant investor identifier |

## Project Structure

```text
app/
├── api/
├── core/
├── models/
├── services/
├── utils/
├── config.py
└── main.py
data/
├── artifacts/
├── examples/
└── generated/
scripts/
tests/
```

## Why This Project Matters

This repository is meant to show applied ML and backend engineering together:

- semantic embeddings for retrieval
- candidate generation and top-k ranking
- explainable rule-based reranking
- benchmarkable ranking metrics
- production-style service boundaries

## Future Improvements

- cached or remote-managed embedding refresh workflows
- richer stage taxonomy and geo normalization
- larger labeled benchmark datasets
- stronger probabilistic entity resolution
- ANN retrieval or vector index integration

## Suggested GitHub Topics

`python` `fastapi` `machine-learning` `nlp` `embeddings` `retrieval` `ranking` `recommendation-system` `applied-ml` `data-engineering`

## Author

Built as a Python backend and applied ML portfolio project focused on retrieval, ranking, evaluation, and service design.
