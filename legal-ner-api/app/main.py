from fastapi import FastAPI
from app.api.v1.endpoints import predict, feedback
from app.core.logging import configure_logging

# Configure logging on startup
configure_logging()

app = FastAPI(
    title="Legal-NER-API",
    version="1.0.0",
    description="API for Named Entity Recognition in Italian legal texts."
)

app.include_router(predict.router, prefix="/api/v1", tags=["NER"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
