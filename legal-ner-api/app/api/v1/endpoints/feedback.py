from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.v1 import schemas
from app.database import crud
from app.database.database import get_db
from app.core.dependencies import get_api_key, get_dataset_builder, get_predictor
from app.feedback.dataset_builder import DatasetBuilder
from app.services.ensemble_predictor import EnsemblePredictor
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
    predictor: EnsemblePredictor = Depends(get_predictor),
    api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    """
    Enhanced feedback endpoint for the new three-stage pipeline system.
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

        # Process feedback through enhanced predictor
        result = await predictor.process_feedback(
            document_id=request.document_id,
            user_id="anonymous",  # TODO: Implement user management
            feedback_data=feedback_data
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])

        return schemas.FeedbackResponse(
            feedback_id=result["feedback_id"],
            status=result["status"],
            quality_impact=result["quality_impact"],
            should_retrain=result["should_retrain"],
            golden_dataset_size=result["golden_dataset_size"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@router.get("/system-stats", response_model=schemas.SystemStatsResponse)
async def get_system_stats(
    predictor: EnsemblePredictor = Depends(get_predictor),
    api_key: str = Depends(get_api_key)
):
    """
    Get comprehensive system statistics including feedback and golden dataset metrics.
    """
    try:
        stats = await predictor.get_system_stats()

        if stats["status"] == "error":
            raise HTTPException(status_code=500, detail=stats["error"])

        return schemas.SystemStatsResponse(
            predictor_type=stats["predictor_type"],
            feedback_stats=stats["feedback_stats"],
            golden_dataset_size=stats["golden_dataset_size"],
            system_accuracy=stats["system_accuracy"],
            status=stats["status"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")

@router.get("/golden-dataset/export", response_model=schemas.GoldenDatasetExportResponse)
async def export_golden_dataset(
    format: str = "json",
    predictor: EnsemblePredictor = Depends(get_predictor),
    api_key: str = Depends(get_api_key)
):
    """
    Export the golden dataset for analysis or external training.
    Supported formats: json, conll
    """
    try:
        if format not in ["json", "conll"]:
            raise HTTPException(status_code=400, detail="Supported formats: json, conll")

        exported_data = await predictor.export_golden_dataset(format=format)

        # Count entries based on format
        if format == "json":
            import json
            data = json.loads(exported_data)
            entry_count = len(data)
        else:  # conll
            entry_count = exported_data.count("# ") // 2  # Rough estimate

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
    predictor: EnsemblePredictor = Depends(get_predictor),
    api_key: str = Depends(get_api_key)
):
    """
    Get high-quality training data from golden dataset.
    """
    try:
        training_data = await predictor.get_training_data(min_quality=min_quality)

        return {
            "training_entries": len(training_data),
            "min_quality_threshold": min_quality,
            "data": training_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training data: {str(e)}")
