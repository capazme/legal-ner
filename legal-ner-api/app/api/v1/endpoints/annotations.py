"""
Annotations API Endpoints

Gestisce le annotation tasks e le entità estratte.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database import models
from app.core.dependencies import get_api_key
import structlog
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import json

log = structlog.get_logger()

router = APIRouter()


# Request/Response models
class CreateTaskRequest(BaseModel):
    document_id: int
    priority: float = 0.5


class TaskResponse(BaseModel):
    id: int
    document_id: int
    status: str
    created_at: str
    priority: Optional[float] = 0.5

    class Config:
        from_attributes = True


class EntityResponse(BaseModel):
    id: int
    document_id: int
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    model: str

    class Config:
        from_attributes = True


class CreateEntityRequest(BaseModel):
    document_id: int
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    model: str = "manual"


class UpdateTaskRequest(BaseModel):
    status: str


@router.post("/annotations/tasks", response_model=TaskResponse, status_code=201)
async def create_annotation_task(
    request: CreateTaskRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Crea un nuovo task di annotazione per un documento.
    """
    try:
        log.info("Creating annotation task", document_id=request.document_id, priority=request.priority)

        # Check if document exists
        document = db.query(models.Document).filter(models.Document.id == request.document_id).first()
        if not document:
            log.warning("Document not found for task creation", document_id=request.document_id)
            raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found")

        # Create task
        task = models.AnnotationTask(
            document_id=request.document_id,
            status="pending",
            priority=request.priority,
            created_at=datetime.now()
        )

        db.add(task)
        db.commit()
        db.refresh(task)

        log.info("Annotation task created successfully", task_id=task.id, document_id=request.document_id)

        return TaskResponse(
            id=task.id,
            document_id=task.document_id,
            status=task.status,
            created_at=task.created_at.isoformat() if task.created_at else datetime.now().isoformat(),
            priority=task.priority
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("Error creating annotation task", error=str(e), document_id=request.document_id)
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


@router.get("/annotations/tasks/{task_id}", response_model=TaskResponse)
async def get_annotation_task(
    task_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Recupera un task di annotazione.
    """
    try:
        log.info("Fetching annotation task", task_id=task_id)

        task = db.query(models.AnnotationTask).filter(models.AnnotationTask.id == task_id).first()

        if not task:
            log.warning("Annotation task not found", task_id=task_id)
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        log.info("Annotation task fetched successfully", task_id=task_id)

        return TaskResponse(
            id=task.id,
            document_id=task.document_id,
            status=task.status,
            created_at=task.created_at.isoformat() if task.created_at else datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Error fetching annotation task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch task: {str(e)}")


@router.get("/annotations/tasks/next", response_model=TaskResponse)
async def get_next_annotation_task(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Recupera il prossimo task di annotazione con la priorità più alta.
    """
    try:
        log.info("Fetching next high-priority annotation task")

        task = db.query(models.AnnotationTask)\
            .filter(models.AnnotationTask.status == "pending")\
            .order_by(models.AnnotationTask.priority.desc())\
            .first()

        if not task:
            log.warning("No pending annotation tasks available")
            raise HTTPException(status_code=404, detail="No pending tasks available")

        log.info("Next annotation task fetched successfully", task_id=task.id, priority=task.priority)

        return TaskResponse(
            id=task.id,
            document_id=task.document_id,
            status=task.status,
            created_at=task.created_at.isoformat() if task.created_at else datetime.now().isoformat(),
            priority=task.priority
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Error fetching next annotation task", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch next task: {str(e)}")


@router.get("/annotations/tasks")
async def list_annotation_tasks(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Lista i task di annotazione, ordinati per priorità.
    """
    try:
        log.info("Listing annotation tasks", status=status, skip=skip, limit=limit)

        query = db.query(models.AnnotationTask)

        if status:
            query = query.filter(models.AnnotationTask.status == status)

        # Changed order_by to priority
        tasks = query.order_by(models.AnnotationTask.priority.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()

        total = query.count()

        log.info("Annotation tasks listed successfully", count=len(tasks), total=total, status_filter=status)

        return {
            "tasks": [
                TaskResponse(
                    id=task.id,
                    document_id=task.document_id,
                    status=task.status,
                    created_at=task.created_at.isoformat() if task.created_at else datetime.now().isoformat(),
                    priority=task.priority
                )
                for task in tasks
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }

    except Exception as e:
        log.error("Error listing annotation tasks", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")


@router.put("/annotations/tasks/{task_id}", response_model=TaskResponse)
async def update_annotation_task(
    task_id: int,
    request: UpdateTaskRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Aggiorna lo stato di un task di annotazione.
    """
    try:
        log.info("Updating annotation task", task_id=task_id, new_status=request.status)

        task = db.query(models.AnnotationTask).filter(models.AnnotationTask.id == task_id).first()

        if not task:
            log.warning("Annotation task not found for update", task_id=task_id)
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        old_status = task.status
        task.status = request.status
        db.commit()
        db.refresh(task)

        log.info("Annotation task updated successfully", task_id=task_id, old_status=old_status, new_status=request.status)

        return TaskResponse(
            id=task.id,
            document_id=task.document_id,
            status=task.status,
            created_at=task.created_at.isoformat() if task.created_at else datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("Error updating annotation task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update task: {str(e)}")


@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Recupera una singola entità tramite il suo ID.
    """
    try:
        log.info("Fetching entity", entity_id=entity_id)

        entity = db.query(models.Entity).filter(models.Entity.id == entity_id).first()

        if not entity:
            log.warning("Entity not found", entity_id=entity_id)
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

        log.info("Entity fetched successfully", entity_id=entity_id)

        return EntityResponse(
            id=entity.id,
            document_id=entity.document_id,
            text=entity.text,
            label=entity.label,
            start_char=entity.start_char,
            end_char=entity.end_char,
            confidence=entity.confidence,
            model=entity.model
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Error fetching entity", entity_id=entity_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch entity: {str(e)}")


@router.get("/entities")
async def list_entities(
    document_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Lista le entità estratte, opzionalmente filtrate per documento.
    """
    try:
        log.info("Listing entities", document_id=document_id, skip=skip, limit=limit)

        query = db.query(models.Entity)

        if document_id:
            query = query.filter(models.Entity.document_id == document_id)

        entities = query.offset(skip).limit(limit).all()
        total = query.count()

        log.info("Entities listed successfully", count=len(entities), total=total, document_id=document_id)

        return {
            "entities": [
                EntityResponse(
                    id=entity.id,
                    document_id=entity.document_id,
                    text=entity.text,
                    label=entity.label,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    confidence=entity.confidence if entity.confidence else 0.0,
                    model=entity.model if entity.model else "unknown"
                )
                for entity in entities
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }

    except Exception as e:
        log.error("Error listing entities", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list entities: {str(e)}")


@router.post("/entities", response_model=EntityResponse, status_code=201)
async def create_entity(
    request: CreateEntityRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Crea una nuova entità estratta.
    """
    try:
        log.info("Creating entity", document_id=request.document_id, label=request.label, text=request.text)

        # Check if document exists
        document = db.query(models.Document).filter(models.Document.id == request.document_id).first()
        if not document:
            log.warning("Document not found for entity creation", document_id=request.document_id)
            raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found")

        # Create entity
        entity = models.Entity(
            document_id=request.document_id,
            text=request.text,
            label=request.label,
            start_char=request.start_char,
            end_char=request.end_char,
            confidence=request.confidence,
            model=request.model
        )

        db.add(entity)
        db.commit()
        db.refresh(entity)

        log.info("Entity created successfully", entity_id=entity.id, document_id=request.document_id, label=request.label)

        return EntityResponse(
            id=entity.id,
            document_id=entity.document_id,
            text=entity.text,
            label=entity.label,
            start_char=entity.start_char,
            end_char=entity.end_char,
            confidence=entity.confidence,
            model=entity.model
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("Error creating entity", error=str(e), document_id=request.document_id)
        raise HTTPException(status_code=500, detail=f"Failed to create entity: {str(e)}")


@router.delete("/entities/{entity_id}")
async def delete_entity(
    entity_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Elimina un'entità.
    """
    try:
        log.info("Deleting entity", entity_id=entity_id)

        entity = db.query(models.Entity).filter(models.Entity.id == entity_id).first()

        if not entity:
            log.warning("Entity not found for deletion", entity_id=entity_id)
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

        db.delete(entity)
        db.commit()

        log.info("Entity deleted successfully", entity_id=entity_id)

        return {"status": "success", "message": f"Entity {entity_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("Error deleting entity", entity_id=entity_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete entity: {str(e)}")


@router.delete("/annotations/tasks/{task_id}")
async def delete_annotation_task(
    task_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Elimina un task di annotazione.
    """
    try:
        log.info("Deleting annotation task", task_id=task_id)

        task = db.query(models.AnnotationTask).filter(models.AnnotationTask.id == task_id).first()

        if not task:
            log.warning("Annotation task not found for deletion", task_id=task_id)
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        db.delete(task)
        db.commit()

        log.info("Annotation task deleted successfully", task_id=task_id)

        return {"status": "success", "message": f"Task {task_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("Error deleting annotation task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")