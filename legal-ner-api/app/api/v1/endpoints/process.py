"""
Process API Endpoint

Pre-elabora documenti con il pipeline NER.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database import models
from app.core.dependencies import get_api_key, get_legal_pipeline
from app.services.specialized_pipeline import LegalSourceExtractionPipeline
import structlog
from pydantic import BaseModel
from typing import Optional, List

log = structlog.get_logger()

router = APIRouter()


# Request/Response models
class ProcessRequest(BaseModel):
    document_id: int


class ProcessResponse(BaseModel):
    document_id: int
    entities_found: int
    message: str


@router.post("/process", response_model=ProcessResponse)
async def process_document(
    request: ProcessRequest,
    db: Session = Depends(get_db),
    pipeline: LegalSourceExtractionPipeline = Depends(get_legal_pipeline),
    api_key: str = Depends(get_api_key)
):
    """
    Pre-elabora un documento con il pipeline NER ed estrae le entità.
    """
    try:
        log.info("Processing document with NER pipeline", document_id=request.document_id)

        # Get document
        document = db.query(models.Document).filter(models.Document.id == request.document_id).first()

        if not document:
            log.warning("Document not found for processing", document_id=request.document_id)
            raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found")

        # Extract entities using pipeline (async method)
        results = await pipeline.extract_legal_sources(document.text)

        log.info("Entities extracted", document_id=request.document_id, count=len(results))

        # Save entities to database
        entities_created = 0
        for result in results:
            # Convert tensor/numpy values to Python native types
            start_char = result.get("start_char", 0)
            end_char = result.get("end_char", 0)
            confidence = result.get("confidence", 0.0)

            # Convert torch.Tensor or numpy types to int/float
            if hasattr(start_char, 'item'):
                start_char = int(start_char.item())
            else:
                start_char = int(start_char)

            if hasattr(end_char, 'item'):
                end_char = int(end_char.item())
            else:
                end_char = int(end_char)

            if hasattr(confidence, 'item'):
                confidence = float(confidence.item())
            else:
                confidence = float(confidence)

            entity = models.Entity(
                document_id=request.document_id,
                text=result.get("text", ""),
                label=result.get("act_type", "unknown"),
                start_char=start_char,
                end_char=end_char,
                confidence=confidence,
                model="specialized_pipeline"
            )
            db.add(entity)
            entities_created += 1

        db.commit()

        log.info("Document processed successfully",
                 document_id=request.document_id,
                 entities_created=entities_created)

        return ProcessResponse(
            document_id=request.document_id,
            entities_found=entities_created,
            message=f"Document processed successfully. {entities_created} entities extracted."
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("Error processing document", document_id=request.document_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")