"""
Model Management API Endpoints

Handles listing, activating, and managing trained models.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database import models
from app.core.dependencies import get_api_key
from app.core.model_manager import model_manager
import structlog
from pydantic import BaseModel
from typing import List, Optional

log = structlog.get_logger()

router = APIRouter()

# Pydantic Models
class TrainedModelResponse(BaseModel):
    id: int
    model_name: str
    version: str
    path: str
    description: Optional[str] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    created_at: str
    is_active: bool

    class Config:
        from_attributes = True

@router.get("/models", response_model=List[TrainedModelResponse])
async def list_trained_models(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Lists all trained models available in the database.
    """
    try:
        log.info("Listing all trained models")
        db_models = db.query(models.TrainedModel).order_by(models.TrainedModel.created_at.desc()).all()
        return [
            TrainedModelResponse(
                **model.__dict__,
                created_at=model.created_at.isoformat()
            ) for model in db_models
        ]
    except Exception as e:
        log.error("Error listing trained models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list trained models")

@router.post("/models/{version}/activate", response_model=dict)
async def activate_model(
    version: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Activates a specific model version, triggering a hot-swap of the pipeline.
    """
    try:
        log.info("Received request to activate model version", version=version)
        result = model_manager.reload_pipeline(db, model_version=version)
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        
        log.info("Model activated successfully", new_version=version)
        return {"status": "success", "message": f"Model {version} activated successfully.", "active_version": result["active_version"]}
    except HTTPException:
        raise
    except Exception as e:
        log.error("Error activating model", version=version, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {str(e)}")

@router.get("/models/active", response_model=dict)
async def get_active_model(
    api_key: str = Depends(get_api_key)
):
    """
    Gets the version of the currently active model in the pipeline.
    """
    active_version = model_manager.get_active_model_version()
    return {"active_model_version": active_version}
