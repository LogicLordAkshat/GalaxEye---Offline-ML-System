import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.logging import logger
from app.db.database import db
from app.ml.model import model_manager
from app.api.routes import router

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing offline database...", extra={"event": "startup_begin"})
    db.init_db()

    logger.info(f"Loading local model from {settings.MODEL_PATH}...", extra={"event": "model_startup_load"})
    model_manager.load_model()

    logger.info(f"Service ready: {settings.APP_NAME} v{settings.APP_VERSION}", extra={"event": "startup_complete"})
    yield
    logger.info("Service shutdown completed.", extra={"event": "shutdown"})


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Offline Satellite Tile Ingestion, Classification, Persistence, and Analyst Triage Service.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    if request.url.path not in ("/health", "/docs", "/openapi.json", "/favicon.ico"):
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
            extra={
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms
            }
        )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": {"stage": "request_validation", "message": "Invalid request parameters", "detail": str(exc.errors())}}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"stage": "server", "message": "An unexpected internal server error occurred."}}
    )


@app.get("/", response_class=HTMLResponse, summary="Satellite Intelligence & Triage Console")
async def root_console():
    if TEMPLATE_PATH.exists():
        return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h2>GalaxEye Satellite Classification Service</h2><p>Online. Visit <a href='/docs'>/docs</a>.</p>")


# Include API routes
app.include_router(router)
app.include_router(router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
