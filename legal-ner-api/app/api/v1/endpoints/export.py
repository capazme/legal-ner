"""
Export API Endpoint

Esporta entità nel formato richiesto da VisuaLex API.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database import models
from app.core.dependencies import get_api_key
from app.services.visualex_mapper import VisuaLexMapper
import structlog
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

log = structlog.get_logger()

router = APIRouter()


# Request/Response models
class ExportVisuaLexRequest(BaseModel):
    document_id: Optional[int] = None
    task_id: Optional[int] = None


class ExportVisuaLexResponse(BaseModel):
    visualex_requests: List[Dict[str, Any]]
    total_entities: int
    converted_entities: int
    document_id: int
    message: str


@router.post("/export-visualex", response_model=ExportVisuaLexResponse)
async def export_visualex(
    request: ExportVisuaLexRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Esporta le entità estratte nel formato VisuaLex NormaRequest.

    Accetta document_id o task_id e converte tutte le entità associate
    nel formato richiesto dall'API VisuaLex per lo scraping.
    """
    try:
        log.info("Starting VisuaLex export",
                 document_id=request.document_id,
                 task_id=request.task_id)

        # Validate input
        if not request.document_id and not request.task_id:
            raise HTTPException(
                status_code=400,
                detail="Must provide either document_id or task_id"
            )

        # Get document_id from task_id if needed
        if request.task_id:
            task = db.query(models.AnnotationTask).filter(
                models.AnnotationTask.id == request.task_id
            ).first()

            if not task:
                log.warning("Task not found for export", task_id=request.task_id)
                raise HTTPException(
                    status_code=404,
                    detail=f"Task {request.task_id} not found"
                )

            document_id = task.document_id
            log.info("Task found, using document",
                     task_id=request.task_id,
                     document_id=document_id)
        else:
            document_id = request.document_id

        # Get document
        document = db.query(models.Document).filter(
            models.Document.id == document_id
        ).first()

        if not document:
            log.warning("Document not found for export", document_id=document_id)
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )

        # Get all entities for this document
        entities = db.query(models.Entity).filter(
            models.Entity.document_id == document_id
        ).all()

        log.info("Entities retrieved for export",
                 document_id=document_id,
                 entity_count=len(entities))

        # Convert entities to dictionary format
        entities_dict = [
            {
                "text": entity.text,
                "label": entity.label,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
                "confidence": entity.confidence if entity.confidence else 0.0,
                "model": entity.model if entity.model else "unknown"
            }
            for entity in entities
        ]

        # Use VisuaLexMapper to convert
        mapper = VisuaLexMapper()
        visualex_requests = mapper.entities_to_visualex_batch(
            entities_dict,
            context=document.text
        )

        log.info("VisuaLex conversion completed",
                 document_id=document_id,
                 total_entities=len(entities),
                 converted_entities=len(visualex_requests))

        return ExportVisuaLexResponse(
            visualex_requests=visualex_requests,
            total_entities=len(entities),
            converted_entities=len(visualex_requests),
            document_id=document_id,
            message=f"Exported {len(visualex_requests)} entities in VisuaLex format"
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Error exporting to VisuaLex format",
                  document_id=request.document_id,
                  task_id=request.task_id,
                  error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export: {str(e)}"
        )


@router.get("/export-visualex/{document_id}", response_model=ExportVisuaLexResponse)
async def export_visualex_by_document(
    document_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Esporta le entità di un documento specifico nel formato VisuaLex.

    Versione GET dell'endpoint per accesso diretto via document_id.
    """
    try:
        log.info("Starting VisuaLex export by document ID", document_id=document_id)

        # Get document
        document = db.query(models.Document).filter(
            models.Document.id == document_id
        ).first()

        if not document:
            log.warning("Document not found for export", document_id=document_id)
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )

        # Get all entities
        entities = db.query(models.Entity).filter(
            models.Entity.document_id == document_id
        ).all()

        log.info("Entities retrieved for export",
                 document_id=document_id,
                 entity_count=len(entities))

        # Convert to dict format
        entities_dict = [
            {
                "text": entity.text,
                "label": entity.label,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
                "confidence": entity.confidence if entity.confidence else 0.0,
                "model": entity.model if entity.model else "unknown"
            }
            for entity in entities
        ]

        # Convert with mapper
        mapper = VisuaLexMapper()
        visualex_requests = mapper.entities_to_visualex_batch(
            entities_dict,
            context=document.text
        )

        log.info("VisuaLex conversion completed",
                 document_id=document_id,
                 total_entities=len(entities),
                 converted_entities=len(visualex_requests))

        return ExportVisuaLexResponse(
            visualex_requests=visualex_requests,
            total_entities=len(entities),
            converted_entities=len(visualex_requests),
            document_id=document_id,
            message=f"Exported {len(visualex_requests)} entities in VisuaLex format"
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Error exporting to VisuaLex format",
                  document_id=document_id,
                  error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export: {str(e)}"
        )