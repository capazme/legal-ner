import uuid
from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from app.api.v1 import schemas
from app.core.dependencies import get_legal_pipeline
from app.services.specialized_pipeline import LegalSourceExtractionPipeline
from app.database import crud
from app.database.database import get_db
import numpy as np
from enum import Enum
import torch
from pathlib import Path

router = APIRouter()

@router.post("/predict", response_model=schemas.NERResponse)
async def predict(
    request: schemas.NERRequest,
    background_tasks: BackgroundTasks,
    pipeline: LegalSourceExtractionPipeline = Depends(get_legal_pipeline),
    db: Session = Depends(get_db)
):
    """
    Receives text and returns recognized legal entities using the specialized pipeline.
    The text and the entities are saved to the database.
    """
    request_id = str(uuid.uuid4())
    
    # Create a dedicated log file for this request
    log_dir = Path("logs/requests")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{request_id}.log"

    # The pipeline now returns structured legal sources.
    # We will use this to populate both pydantic_legal_sources and pydantic_entities.
    extracted_legal_sources_data = await pipeline.extract_legal_sources(request.text, log_file_path=str(log_file_path))

    pydantic_entities = []
    pydantic_legal_sources = []

    # Helper function to convert numpy types to Python native types
    def convert_to_json_compatible(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()  # Convert numpy scalars to Python native types
        if isinstance(obj, dict):
            return {k: convert_to_json_compatible(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_json_compatible(elem) for elem in obj]
        if isinstance(obj, Enum):
            return obj.value
        return obj

    for legal_source_data in extracted_legal_sources_data:
        # Convert to JSON-compatible types first
        json_compatible_legal_source_data = convert_to_json_compatible(legal_source_data)

        # Create LegalSource schema
        legal_source = schemas.LegalSource(
            source_type=str(json_compatible_legal_source_data.get('source_type')) if json_compatible_legal_source_data.get('source_type') else None,
            text=json_compatible_legal_source_data['text'],
            confidence=json_compatible_legal_source_data['confidence'],  # Already converted to native Python float
            start_char=json_compatible_legal_source_data['start_char'],
            end_char=json_compatible_legal_source_data['end_char'],
            act_type=str(json_compatible_legal_source_data.get('act_type')) if json_compatible_legal_source_data.get('act_type') else None,
            date=json_compatible_legal_source_data.get('date'),
            act_number=json_compatible_legal_source_data.get('act_number'),
            article=json_compatible_legal_source_data.get('article'),
            version=json_compatible_legal_source_data.get('version'),
            version_date=json_compatible_legal_source_data.get('version_date'),
            annex=json_compatible_legal_source_data.get('annex')
        )
        pydantic_legal_sources.append(legal_source)

        # Create generic Entity schema from the legal source data
        pydantic_entity = schemas.Entity(
            text=json_compatible_legal_source_data['text'],
            label=json_compatible_legal_source_data.get('source_type', 'UNKNOWN_LEGAL_ENTITY'), # Use source_type as label
            start_char=json_compatible_legal_source_data['start_char'],
            end_char=json_compatible_legal_source_data['end_char'],
            confidence=json_compatible_legal_source_data['confidence'],  # Already converted
            model='specialized_pipeline', # Hardcode model as it's from this pipeline
            stage='structure_building', # Final stage of the pipeline
            structured_data=json_compatible_legal_source_data, # Store the full structured data here
            validation_score=None, # Not yet calculated by the pipeline
            semantic_correlations=None, # Not yet calculated by the pipeline
            final_quality_score=None # Not yet calculated by the pipeline
        )
        pydantic_entities.append(pydantic_entity)

    # Calculate aggregate metrics (using pydantic_entities for confidence)
    if pydantic_entities:
        # Calculate uncertainty from confidence scores
        confidences = [e.confidence for e in pydantic_entities]
        avg_confidence = sum(confidences) / len(confidences)
        overall_uncertainty = 1.0 - avg_confidence
        requires_review = overall_uncertainty > 0.3 or any(c < 0.7 for c in confidences)
    else:
        overall_uncertainty = 1.0
        requires_review = True

    # Save the document and entities to the database
    document = crud.create_document(db, text=request.text)
    crud.create_entities_for_document(db, document_id=document.id, entities=pydantic_entities)

    if requires_review:
        crud.create_annotation_task(db, document_id=document.id, priority=overall_uncertainty)

    response_data = schemas.NERResponse(
        entities=pydantic_entities,
        legal_sources=pydantic_legal_sources,
        requires_review=requires_review,
        request_id=request_id,
    )
    return jsonable_encoder(response_data) # Explicitly encode the response