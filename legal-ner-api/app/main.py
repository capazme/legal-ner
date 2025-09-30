from fastapi import FastAPI, Request, Response
from app.api.v1.endpoints import predict, feedback, active_learning, documents, annotations, process, export, models, labels
from app.core.logging import configure_logging
from app.core.model_manager import model_manager
from app.database.database import SessionLocal
import numpy as np
import torch
import structlog
import time
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging on startup
configure_logging()
log = structlog.get_logger()

app = FastAPI(
    title="Legal-NER-API",
    version="1.0.0",
    description="API for Named Entity Recognition in Italian legal texts with Active Learning.",
    json_encoders={
        np.ndarray: lambda x: x.tolist(),
        torch.Tensor: lambda x: x.tolist()
    }
)

# Logging middleware for detailed request/response tracking
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Start timer
        start_time = time.time()

        # Log incoming request
        log.info(
            "Incoming request",
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_host=request.client.host if request.client else "unknown"
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            log.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2)
            )

            return response

        except Exception as e:
            duration = time.time() - start_time
            log.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration * 1000, 2)
            )
            raise

# Add middleware
app.add_middleware(LoggingMiddleware)

# Startup event with detailed logging
@app.on_event("startup")
async def startup_event():
    log.info("Legal-NER-API starting up", version="1.0.0")
    
    # Load the active model from the database
    log.info("Loading active ML model...")
    db = SessionLocal()
    try:
        model_manager.load_active_model(db)
    finally:
        db.close()
    
    log.info("Registered routers", routers=["predict", "feedback", "active_learning", "documents", "annotations", "process", "export"])

# Shutdown event with logging
@app.on_event("shutdown")
async def shutdown_event():
    log.info("Legal-NER-API shutting down")

# Register API routers
log.info("Registering API routers")
app.include_router(predict.router, prefix="/api/v1", tags=["NER"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
app.include_router(active_learning.router, prefix="/api/v1/active-learning", tags=["Active Learning"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(annotations.router, prefix="/api/v1", tags=["Annotations"])
app.include_router(process.router, prefix="/api/v1", tags=["Processing"])
app.include_router(export.router, prefix="/api/v1", tags=["Export"])
app.include_router(models.router, prefix="/api/v1", tags=["Models"])
app.include_router(labels.router, prefix="/api/v1", tags=["Labels"])

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
