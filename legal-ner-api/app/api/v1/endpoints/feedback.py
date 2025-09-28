from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.v1 import schemas
from app.database import crud
from app.database.database import get_db
from app.core.dependencies import get_api_key, get_dataset_builder
from app.feedback.dataset_builder import DatasetBuilder

router = APIRouter()

@router.post("/feedback", status_code=201)
def submit_feedback(
    feedback: schemas.FeedbackRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key) # Add API key dependency
):
    """
    Receives and stores human-provided annotations for a given document.
    """
    # A real implementation would get the user_id from an authentication dependency
    user_id = "default_user"

    crud.create_annotations(db, annotations=feedback.annotations, user_id=user_id)

    return {"message": "Feedback received successfully."}

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
