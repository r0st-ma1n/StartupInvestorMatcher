from fastapi import FastAPI
from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="venture-match-engine")
    app.include_router(router)
    return app


app = create_app()
