"""
Documents API Endpoints

Gestisce la creazione, lettura e gestione dei documenti.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database import models
from app.core.dependencies import get_api_key
import structlog
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

log = structlog.get_logger()

router = APIRouter()


# Request/Response models
class CreateDocumentRequest(BaseModel):
    text: str


class DocumentResponse(BaseModel):
    id: int
    text: str
    created_at: str

    class Config:
        from_attributes = True


@router.post("/documents", response_model=DocumentResponse, status_code=201)
async def create_document(
    request: CreateDocumentRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Crea un nuovo documento nel database.
    """
    try:
        log.info("Creating new document", text_length=len(request.text))

        # Create document
        document = models.Document(
            text=request.text
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        log.info("Document created successfully", document_id=document.id)

        return DocumentResponse(
            id=document.id,
            text=document.text,
            created_at=document.created_at.isoformat() if document.created_at else datetime.now().isoformat()
        )

    except Exception as e:
        db.rollback()
        log.error("Error creating document", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create document: {str(e)}")


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Recupera un documento dal database.
    """
    try:
        log.info("Fetching document", document_id=document_id)

        document = db.query(models.Document).filter(models.Document.id == document_id).first()

        if not document:
            log.warning("Document not found", document_id=document_id)
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        log.info("Document fetched successfully", document_id=document_id)

        return DocumentResponse(
            id=document.id,
            text=document.text,
            created_at=document.created_at.isoformat() if document.created_at else datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Error fetching document", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch document: {str(e)}")


@router.get("/documents")
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Lista tutti i documenti con paginazione.
    """
    try:
        log.info("Listing documents", skip=skip, limit=limit)

        documents = db.query(models.Document)\
            .order_by(models.Document.created_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()

        total = db.query(models.Document).count()

        log.info("Documents listed successfully", count=len(documents), total=total)

        return {
            "documents": [
                DocumentResponse(
                    id=doc.id,
                    text=doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,  # Truncate for list
                    created_at=doc.created_at.isoformat() if doc.created_at else datetime.now().isoformat()
                )
                for doc in documents
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }

    except Exception as e:
        log.error("Error listing documents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Elimina un documento dal database.
    """
    try:
        log.info("Deleting document", document_id=document_id)

        document = db.query(models.Document).filter(models.Document.id == document_id).first()

        if not document:
            log.warning("Document not found for deletion", document_id=document_id)
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        db.delete(document)
        db.commit()

        log.info("Document deleted successfully", document_id=document_id)

        return {"status": "success", "message": f"Document {document_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("Error deleting document", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")