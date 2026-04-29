# venture-match-engine

Production-style portfolio project for startup-investor matching with a hybrid applied-ML and backend architecture.

Suggested GitHub topics:

- `python`
- `fastapi`
- `machine-learning`
- `nlp`
- `embeddings`
- `information-retrieval`
- `ranking`
- `data-engineering`
- `entity-resolution`

## What It Demonstrates

- semantic retrieval with sentence embeddings
- cosine-similarity candidate generation
- explainable weighted reranking
- FastAPI inference endpoint
- evaluation metrics for ranking quality
- baseline entity resolution / deduplication

## Current Architecture

```text
app/
  api/
  models/
  services/
data/
tests/
```

Core pipeline:

1. load typed startup and investor profiles from CSV
2. generate embeddings for profile text
3. retrieve semantic top-k candidates with cosine similarity
4. rerank candidates with weighted business rules
5. return explainable match results

Why this repository is relevant for applied ML and data engineering interviews:

- it demonstrates semantic retrieval rather than keyword-only matching
- it separates candidate generation from reranking and business scoring
- it includes evaluation metrics for retrieval quality
- it includes a baseline entity resolution workflow for messy real-world data
- it packages the logic behind a typed FastAPI service instead of notebook-only code

## Main Services

- `app/services/data_loader.py`
  Loads and validates CSV data into typed domain models.
- `app/services/embedding_service.py`
  Formats profiles into retrieval text and generates embeddings.
- `app/services/retrieval_service.py`
  Computes cosine similarity and retrieves semantic candidates.
- `app/services/scoring_service.py`
  Applies rule-based scoring for industry, stage, geography, and ticket fit.
- `app/services/matching_service.py`
  Orchestrates retrieval and reranking into final `MatchResult` outputs.
- `app/services/evaluation_service.py`
  Computes `precision@k`, `recall@k`, and `hit-rate@k`.
- `app/services/entity_resolution_service.py`
  Performs lightweight deduplication with normalization, blocking, and fuzzy matching.

## API

`POST /match`

Request body:

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

## Quickstart

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the API locally:

```bash
uvicorn app.main:app --reload
```

Run tests:

```bash
pytest --basetemp .pytest_tmp
```

Validate importable app syntax:

```bash
python -m py_compile app/main.py
```

## Expected CSV Inputs

`startups.csv`

| column | required | description |
| --- | --- | --- |
| `startup_id` | yes | Unique startup identifier |
| `name` | yes | Startup name |
| `description` | yes | Free-text startup description |
| `industries` | yes | Comma-separated industries or a normalized list upstream |
| `stage` | yes | Funding stage such as `Pre-Seed`, `Seed`, `Series A` |
| `country` | yes | Country focus |
| `region` | yes | Region focus |
| `fundraising_amount` | yes | Target raise amount |
| `currency` | yes | Currency code, default domain assumption is `USD` |
| `website` | no | Optional startup website |

`investors.csv`

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

Notes:

- list-like fields are normalized by the loader from comma-separated strings
- missing optional fields are handled gracefully in scoring and dedup flows
- `ticket_min` must not exceed `ticket_max`

Sample files are provided in:

- `data/examples/startups.sample.csv`
- `data/examples/investors.sample.csv`

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/match" ^
  -H "Content-Type: application/json" ^
  -d "{\"startup\":{\"startup_id\":\"s1\",\"name\":\"Acme AI\",\"description\":\"AI tooling for diligence\",\"industries\":[\"AI\"],\"stage\":\"Seed\",\"country\":\"US\",\"region\":\"North America\",\"fundraising_amount\":1000000,\"currency\":\"USD\"},\"investors\":[{\"investor_id\":\"i1\",\"name\":\"North Star Ventures\",\"description\":\"AI-focused seed fund\",\"industries\":[\"AI\"],\"preferred_stages\":[\"Seed\"],\"countries\":[\"US\"],\"regions\":[\"North America\"],\"ticket_min\":250000,\"ticket_max\":2000000,\"currency\":\"USD\"}],\"top_k\":1,\"candidate_pool_size\":5}"
```

## Testing

The repository includes unit tests for:

- CSV loading
- embedding service behavior
- retrieval ranking
- scoring logic
- end-to-end matching orchestration
- API request/response flow
- evaluation metrics
- entity resolution

CI coverage:

- dependency installation from `requirements.txt`
- `python -m py_compile app/main.py`
- `pytest`

## Next Extensions

- richer stage taxonomy and adjacency rules
- cached investor embeddings for batch workloads
- offline benchmark datasets for evaluation
- stronger probabilistic entity resolution
- optional vector index or ANN retrieval for scale
