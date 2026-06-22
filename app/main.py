import structlog
import weave
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import auth, documents, entities, health, query
from app.core.config import settings
from app.core.exceptions import http_exception_handler
from app.core.logging import setup_logging

log = structlog.get_logger()


def create_app() -> FastAPI:
    setup_logging()

    try:
        weave.init(settings.wandb_project)
        log.info("weave.initialized", project=settings.wandb_project)
    except Exception as exc:
        log.warning("weave.init_failed", error=str(exc))

    app = FastAPI(
        title="IDP RAG System",
        description="Intelligent Document Processing with RAG pipeline",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(HTTPException, http_exception_handler)

    # slowapi
    app.state.limiter = documents.limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    if settings.enable_metrics:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(entities.router)

    @app.on_event("startup")
    async def startup():
        log.info("app.started", environment=settings.environment)

    @app.on_event("shutdown")
    async def shutdown():
        log.info("app.shutdown")

    return app


app = create_app()

