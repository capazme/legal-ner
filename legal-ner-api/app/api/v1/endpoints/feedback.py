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
    api_key: str = Depends(get_api_key) # Add API key dependency
):
    """
    Receives and stores human-provided annotations for a given document (legacy endpoint).
    """
    # A real implementation would get the user_id from an authentication dependency
    user_id = "default_user"

    crud.create_annotations(db, annotations=feedback.annotations, user_id=user_id)

    return {"message": "Feedback received successfully."}

@router.post("/enhanced-feedback", response_model=schemas.FeedbackResponse)
async def provide_enhanced_feedback(
    request: schemas.EnhancedFeedbackRequest,
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    """
    Enhanced feedback endpoint for the specialized pipeline system.
    Integrates directly with golden dataset and continuous learning.
    """
    try:
        # Convert request to feedback data format
        feedback_data = {
            "type": request.feedback_type,
            "original_entity": request.original_entity.dict() if request.original_entity else None,
            "corrected_entity": request.corrected_entity.dict() if request.corrected_entity else None,
            "confidence_score": request.confidence_score,
            "notes": request.notes
        }

        # A real implementation would get the user_id from an authentication dependency
        user_id = "default_user" # Placeholder for now

        # Process feedback through feedback loop
        result = await feedback_loop.process_feedback(
            document_id=str(request.document_id), # Ensure document_id is passed
            user_id=user_id,
            feedback_data=feedback_data
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return schemas.FeedbackResponse(
            feedback_id=result.get("feedback_id", "unknown"),
            status=result.get("status", "processed"),
            quality_impact=result.get("quality_impact", 0.0),
            should_retrain=result.get("should_retrain", False),
            golden_dataset_size=result.get("golden_dataset_size", 0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@router.get("/system-stats", response_model=schemas.SystemStatsResponse)
async def get_system_stats(
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    api_key: str = Depends(get_api_key)
):
    """
    Get comprehensive system statistics including feedback and golden dataset metrics.
    """
    try:
        stats = feedback_loop.get_statistics()

        return schemas.SystemStatsResponse(
            predictor_type="specialized_pipeline",
            feedback_stats=stats,
            golden_dataset_size=stats.get("golden_dataset_size", 0),
            system_accuracy=stats.get("system_accuracy", 0.0),
            status="active"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")

@router.get("/golden-dataset/export", response_model=schemas.GoldenDatasetExportResponse)
async def export_golden_dataset(
    format: str = "json",
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    api_key: str = Depends(get_api_key)
):
    """
    Export the golden dataset for analysis or external training.
    Supported formats: json, conll
    """
    try:
        if format not in ["json", "conll"]:
            raise HTTPException(status_code=400, detail="Supported formats: json, conll")

        exported_data = await feedback_loop.export_golden_dataset(format=format)

        # Count entries based on format
        if format == "json":
            import json
            data = json.loads(exported_data) if isinstance(exported_data, str) else exported_data
            entry_count = len(data) if isinstance(data, list) else 0
        else:  # conll
            entry_count = str(exported_data).count("# ") // 2 if exported_data else 0

        return schemas.GoldenDatasetExportResponse(
            format=format,
            data=exported_data,
            entry_count=entry_count,
            export_timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting dataset: {str(e)}")

@router.post("/build-dataset", status_code=201)
def build_dataset_endpoint(
    version_name: str,
    db: Session = Depends(get_db),
    builder: DatasetBuilder = Depends(get_dataset_builder),
    api_key: str = Depends(get_api_key) # Protect with API key
):
    """
    Triggers the process of building a new dataset from collected annotations
    and uploading it to MinIO.
    """
    try:
        object_name = builder.build_dataset(db, version_name)
        return {"message": "Dataset built and uploaded successfully.", "object_name": object_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build dataset: {str(e)}")

@router.get("/training-data")
async def get_training_data(
    min_quality: float = 0.8,
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    api_key: str = Depends(get_api_key)
):
    """
    Get high-quality training data from golden dataset.
    """
    try:
        # Get training data from feedback loop
        training_data = await feedback_loop.get_training_data(min_quality=min_quality)

        return {
            "training_entries": len(training_data) if training_data else 0,
            "min_quality_threshold": min_quality,
            "data": training_data or []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training data: {str(e)}")