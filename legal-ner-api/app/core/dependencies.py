from functools import lru_cache
from fastapi import Depends, Header, HTTPException, status
from app.core.config import settings
from app.services.specialized_pipeline import LegalSourceExtractionPipeline
from app.services.feedback_loop import FeedbackLoop
from app.feedback.dataset_builder import DatasetBuilder
from app.core.model_manager import model_manager

def get_legal_pipeline() -> LegalSourceExtractionPipeline:
    """
    Returns the globally managed instance of the pipeline from the ModelManager.
    """
    return model_manager.get_pipeline()

def get_feedback_loop() -> FeedbackLoop:
    """
    Returns a new instance of the FeedbackLoop (lightweight, database-backed).
    Non serve più caching perché non ha stato interno.
    """
    return FeedbackLoop()

def get_dataset_builder() -> DatasetBuilder:
    """
    Returns a new instance of the DatasetBuilder.
    Non dipende più da FeedbackLoop, quindi non serve caching.
    """
    return DatasetBuilder()

async def get_api_key(x_api_key: str = Header(...)):
    if x_api_key == settings.API_KEY:
        return x_api_key
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
