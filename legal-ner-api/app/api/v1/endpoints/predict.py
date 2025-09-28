import uuid
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from app.api.v1 import schemas
from app.core.dependencies import get_predictor, get_legal_source_extractor
from app.services.ensemble_predictor import EnsemblePredictor
from app.services.legal_source_extractor import LegalSourceExtractor
from app.database import crud
from app.database.database import get_db

router = APIRouter()

@router.post("/predict", response_model=schemas.NERResponse)
async def predict(
    request: schemas.NERRequest,
    background_tasks: BackgroundTasks,
    predictor: EnsemblePredictor = Depends(get_predictor),
    extractor: LegalSourceExtractor = Depends(get_legal_source_extractor),
    db: Session = Depends(get_db)
):
    """
    Receives text and returns recognized legal entities.
    The text and the entities are saved to the database.
    """
    request_id = str(uuid.uuid4())

    entities, requires_review, overall_uncertainty = await predictor.predict(request.text)

    # Convert dicts to Pydantic models
    pydantic_entities = [schemas.Entity(**e) for e in entities]
    pydantic_legal_sources = [schemas.LegalSource(**s) for s in extracted_sources]

    # Save the document and entities to the database
    document = crud.create_document(db, text=request.text)
    crud.create_entities_for_document(db, document_id=document.id, entities=pydantic_entities)
    
    if requires_review:
        crud.create_annotation_task(db, document_id=document.id, priority=overall_uncertainty)

    # background_tasks.add_task(log_prediction, request_id, entities)

    return schemas.NERResponse(
        entities=pydantic_entities,
        legal_sources=pydantic_legal_sources,
        requires_review=requires_review,
        request_id=request_id,
    )
