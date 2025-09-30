"""
Active Learning API Endpoints

Gestisce il ciclo di active learning:
1. Identificazione samples incerti
2. Creazione task di annotazione
3. Training modello con feedback
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.feedback.active_learning import ActiveLearningManager
from app.models.model_trainer import ModelTrainer
from app.core.config import settings
import structlog
from typing import Dict, Any, Optional
from pydantic import BaseModel
from app.services.feedback_loop_service import FeedbackLoopService

log = structlog.get_logger()

router = APIRouter()

# Request/Response models
class TriggerIterationRequest(BaseModel):
    batch_size: int = 10

class TriggerIterationResponse(BaseModel):
    status: str
    document_count: int = 0
    task_ids: list = []
    message: str

class TrainModelRequest(BaseModel):
    model_name: str = "DeepMount00/Italian_NER_XXL_v2"
    dataset_version: str = None

class TrainModelResponse(BaseModel):
    status: str
    model_path: str = None
    message: str

@router.post("/trigger-iteration", response_model=TriggerIterationResponse)
async def trigger_active_learning_iteration(
    request: TriggerIterationRequest,
    db: Session = Depends(get_db)
):
    """
    Avvia un'iterazione di active learning:
    1. Identifica documenti con previsioni incerte
    2. Crea task di annotazione per revisione umana
    """
    try:
        log.info("Triggering active learning iteration", batch_size=request.batch_size)

        manager = ActiveLearningManager()
        result = manager.run_active_learning_iteration(db, batch_size=request.batch_size)

        return TriggerIterationResponse(**result)

    except Exception as e:
        log.error("Error triggering active learning iteration", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to trigger iteration: {str(e)}")

@router.post("/train-model", response_model=TrainModelResponse)
async def train_model_with_feedback(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Avvia il training di un modello con i feedback raccolti.

    Il training viene eseguito in background per non bloccare l'API.
    """
    try:
        log.info("Starting model training", model_name=request.model_name)

        manager = ActiveLearningManager()

        # Esegui training in background
        background_tasks.add_task(
            _train_model_background,
            db,
            manager,
            request.model_name
        )

        return TrainModelResponse(
            status="training_started",
            message="Model training started in background. Check logs for progress."
        )

    except Exception as e:
        log.error("Error starting model training", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

async def _train_model_background(db: Session, manager: ActiveLearningManager, model_name: str):
    """Background task per training del modello."""
    try:
        model_path = manager.train_model_with_feedback(db, model_name)
        log.info("Model training completed successfully", model_path=model_path)
    except Exception as e:
        log.error("Model training failed", error=str(e))

@router.get("/training-stats")
async def get_training_stats(db: Session = Depends(get_db)):
    """
    Restituisce statistiche sul training:
    - Numero di annotazioni validate
    - Dimensione dataset corrente
    - Modelli disponibili
    """
    try:
        from app.database import models

        # Count validated annotations
        validated_count = db.query(models.Annotation)\
            .filter(models.Annotation.is_correct.isnot(None))\
            .count()

        # Count pending tasks
        pending_tasks = db.query(models.AnnotationTask)\
            .filter(models.AnnotationTask.status == "pending")\
            .count()

        # Get dataset versions
        datasets = db.query(models.DatasetVersion)\
            .order_by(models.DatasetVersion.created_at.desc())\
            .limit(10)\
            .all()

        dataset_info = [
            {
                "version_name": d.version_name,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "description": d.description
            }
            for d in datasets
        ]

        return {
            "validated_annotations": validated_count,
            "pending_tasks": pending_tasks,
            "total_documents": db.query(models.Document).count(),
            "datasets": dataset_info
        }

    except Exception as e:
        log.error("Error fetching training stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

feedback_loop_service_instance: Optional[FeedbackLoopService] = None

@router.post("/feedback-loop/start", status_code=202)
async def start_feedback_loop(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Starts the automated feedback loop in the background.
    The loop will periodically check for new annotations and trigger training if a threshold is met.
    """
    global feedback_loop_service_instance
    if feedback_loop_service_instance and feedback_loop_service_instance.is_running:
        raise HTTPException(status_code=400, detail="Feedback loop is already running.")

    log.info("Starting feedback loop service via API request.")
    # We need to create a new session for the background task
    from app.database.database import SessionLocal
    db_session = SessionLocal()
    feedback_loop_service_instance = FeedbackLoopService(db_session)
    background_tasks.add_task(feedback_loop_service_instance.start_loop)
    
    return {"status": "success", "message": "Automated feedback loop started in the background."}

@router.post("/feedback-loop/stop", status_code=200)
async def stop_feedback_loop():
    """
    Stops the automated feedback loop if it is running.
    """
    global feedback_loop_service_instance
    if not feedback_loop_service_instance or not feedback_loop_service_instance.is_running:
        raise HTTPException(status_code=404, detail="Feedback loop is not running.")

    log.info("Stopping feedback loop service via API request.")
    feedback_loop_service_instance.stop_loop()
    feedback_loop_service_instance = None
    
    return {"status": "success", "message": "Automated feedback loop stopped."}