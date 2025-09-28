from functools import lru_cache
from fastapi import Header, HTTPException, status
from app.core.config import settings
from app.services.ensemble_predictor import EnsemblePredictor
from app.services.legal_source_extractor import LegalSourceExtractor
from app.services.semantic_validator import SemanticValidator
from app.services.entity_merger import EntityMerger
from app.services.confidence_calibrator import ConfidenceCalibrator
from app.feedback.dataset_builder import DatasetBuilder

@lru_cache(maxsize=1)
def get_predictor() -> EnsemblePredictor:
    """
    Returns a cached instance of the EnsemblePredictor.
    The model is loaded only once.
    """
    return EnsemblePredictor()

@lru_cache(maxsize=1)
def get_legal_source_extractor() -> LegalSourceExtractor:
    """
    Returns a cached instance of the LegalSourceExtractor.
    """
    return LegalSourceExtractor()

@lru_cache(maxsize=1)
def get_semantic_validator() -> SemanticValidator:
    """
    Returns a cached instance of the SemanticValidator.
    """
    return SemanticValidator()

@lru_cache(maxsize=1)
def get_entity_merger() -> EntityMerger:
    """
    Returns a cached instance of the EntityMerger.
    """
    return EntityMerger()

@lru_cache(maxsize=1)
def get_dataset_builder() -> DatasetBuilder:
    """
    Returns a cached instance of the DatasetBuilder.
    """
    return DatasetBuilder()

@lru_cache(maxsize=1)
def get_confidence_calibrator() -> ConfidenceCalibrator:
    """
    Returns a cached instance of the ConfidenceCalibrator.
    """
    return ConfidenceCalibrator()

async def get_api_key(x_api_key: str = Header(...)):
    if x_api_key == settings.API_KEY:
        return x_api_key
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
