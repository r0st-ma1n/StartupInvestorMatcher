from fastapi import FastAPI
from app.api.routes import router

OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Service health and readiness checks.",
    },
    {
        "name": "catalog",
        "description": "Sample startup and investor catalog endpoints.",
    },
    {
        "name": "matching",
        "description": "Startup-investor retrieval and reranking endpoints.",
    },
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="venture-match-engine",
        description=(
            "Production-style startup-investor matching API with semantic retrieval, "
            "rule-based scoring, and explainable reranking."
        ),
        openapi_tags=OPENAPI_TAGS,
    )
    app.include_router(router)
    return app


app = create_app()
