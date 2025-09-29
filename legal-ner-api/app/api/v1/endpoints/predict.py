import uuid
from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.encoders import jsonable_encoder # Added import
from sqlalchemy.orm import Session
from app.api.v1 import schemas
from app.core.dependencies import get_legal_pipeline
from app.services.specialized_pipeline import LegalSourceExtractionPipeline
from app.database import crud
from app.database.database import get_db
import numpy as np
from enum import Enum
import torch

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

    # The pipeline now returns structured legal sources.
    # We will use this to populate both pydantic_legal_sources and pydantic_entities.
    extracted_legal_sources_data = await pipeline.extract_legal_sources(request.text)

    pydantic_entities = []
    pydantic_legal_sources = []

    for legal_source_data in extracted_legal_sources_data:
        # Create LegalSource schema
        legal_source = schemas.LegalSource(
            source_type=str(legal_source_data.get('source_type')) if legal_source_data.get('source_type') else None,
            text=legal_source_data['text'],
            confidence=float(legal_source_data['confidence']),
            start_char=legal_source_data['start_char'],
            end_char=legal_source_data['end_char'],
            act_type=str(legal_source_data.get('act_type')) if legal_source_data.get('act_type') else None,
            date=legal_source_data.get('date'),
            act_number=legal_source_data.get('act_number'),
            article=legal_source_data.get('article'),
            version=legal_source_data.get('version'),
            version_date=legal_source_data.get('version_date'),
            annex=legal_source_data.get('annex')
        )
        pydantic_legal_sources.append(legal_source)

        # Create generic Entity schema from the legal source data
        def convert_to_json_compatible(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_to_json_compatible(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_json_compatible(elem) for elem in obj]
            if isinstance(obj, Enum): # Handle Enums
                return obj.value
            return obj

        json_compatible_legal_source_data = convert_to_json_compatible(legal_source_data)

        pydantic_entity = schemas.Entity(
            text=legal_source_data['text'],
            label=legal_source_data.get('source_type', 'UNKNOWN_LEGAL_ENTITY'), # Use source_type as label
            start_char=legal_source_data['start_char'],
            end_char=legal_source_data['end_char'],
            confidence=float(legal_source_data['confidence']),
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