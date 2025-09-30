from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.v1 import schemas
from app.database import crud
from app.database.database import get_db
from app.core.dependencies import get_api_key, get_dataset_builder, get_feedback_loop
from app.feedback.dataset_builder import DatasetBuilder
from app.services.feedback_loop import FeedbackLoop
from datetime import datetime

router = APIRouter()

@router.post("/feedback", status_code=201)
def submit_feedback(
    feedback: schemas.FeedbackRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Receives and stores human-provided annotations for a given document (legacy endpoint).
    """
    # A real implementation would get the user_id from an authentication dependency
    user_id = "default_user"

    crud.create_annotations(db, annotations=feedback.annotations, user_id=user_id)

    return {"message": "Feedback received successfully."}

@router.post("/enhanced-feedback", response_model=schemas.FeedbackResponse)
def provide_enhanced_feedback(
    request: schemas.EnhancedFeedbackRequest,
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    """
    Enhanced feedback endpoint for the specialized pipeline system.
    Integrates directly with database annotations.

    Note: Il nuovo FeedbackLoop non ha più process_feedback() async.
    Le annotazioni vengono salvate direttamente nel database tramite l'UI.
    """
    # TODO: Questa funzione non è più necessaria nel nuovo design
    # Le annotazioni sono gestite direttamente tramite l'UI e salvate nel DB
    raise HTTPException(
        status_code=501,
        detail="This endpoint is deprecated. Use the annotation UI workflow instead."
    )

@router.get("/system-stats", response_model=schemas.SystemStatsResponse)
def get_system_stats(
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Get comprehensive system statistics including feedback and golden dataset metrics.
    """
    try:
        # Get feedback statistics from database
        feedback_stats = feedback_loop.get_feedback_statistics(db, days=30)

        # Get quality metrics
        quality_metrics = feedback_loop.calculate_quality_metrics(db)

        # Combine stats
        return schemas.SystemStatsResponse(
            predictor_type="specialized_pipeline",
            feedback_stats=feedback_stats,
            golden_dataset_size=feedback_stats.get("golden_dataset_size", 0),
            system_accuracy=quality_metrics.get("precision", 0.0),  # Use precision as system accuracy
            status="active"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")

@router.get("/feedback-statistics")
def get_feedback_statistics(
    days: int = 30,
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Get detailed feedback statistics for the last N days.
    """
    try:
        stats = feedback_loop.get_feedback_statistics(db, days=days)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feedback statistics: {str(e)}")

@router.get("/quality-metrics")
def get_quality_metrics(
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Get real quality metrics (Precision, Recall, F1) based on annotations.
    """
    try:
        metrics = feedback_loop.calculate_quality_metrics(db)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating quality metrics: {str(e)}")

@router.get("/golden-dataset")
def get_golden_dataset(
    min_feedback_count: int = 1,
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Get the golden dataset built from validated annotations.
    """
    try:
        golden_dataset = feedback_loop.get_golden_dataset(db, min_feedback_count=min_feedback_count)
        return {
            "dataset_size": len(golden_dataset),
            "total_entities": sum(len(doc["entities"]) for doc in golden_dataset),
            "avg_quality": sum(doc["quality_score"] for doc in golden_dataset) / len(golden_dataset) if golden_dataset else 0.0,
            "documents": golden_dataset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting golden dataset: {str(e)}")

@router.get("/golden-dataset/for-training")
def get_golden_dataset_for_training(
    min_quality_score: float = 0.8,
    max_entries: int | None = None,
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Get high-quality training data from golden dataset.
    """
    try:
        training_data = feedback_loop.get_golden_dataset_for_training(
            db,
            min_quality_score=min_quality_score,
            max_entries=max_entries
        )

        return {
            "training_entries": len(training_data),
            "min_quality_threshold": min_quality_score,
            "avg_quality": sum(e["quality_score"] for e in training_data) / len(training_data) if training_data else 0.0,
            "data": training_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training data: {str(e)}")

@router.get("/retraining-status")
def check_retraining_status(
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Check if retraining should be triggered based on recent annotations.
    """
    try:
        status = feedback_loop.check_retraining_trigger(db)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking retraining status: {str(e)}")

@router.post("/build-dataset", status_code=201)
def build_dataset_endpoint(
    version_name: str,
    db: Session = Depends(get_db),
    builder: DatasetBuilder = Depends(get_dataset_builder),
    api_key: str = Depends(get_api_key)
):
    """
    Triggers the process of building a new dataset from collected annotations
    and uploading it to MinIO.
    """
    try:
        object_name = builder.build_dataset(db, version_name)
        return {
            "message": "Dataset built and uploaded successfully.",
            "object_name": object_name,
            "version_name": version_name
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build dataset: {str(e)}")
