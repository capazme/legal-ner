from sqlalchemy.orm import Session
from app.database import models
from app.core.config import settings
import structlog
from minio import Minio
from minio.error import S3Error
import json
import io

log = structlog.get_logger()

class DatasetBuilder:
    def __init__(self):
        log.info("Initializing DatasetBuilder")
        self.minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False # Use True for HTTPS
        )
        # Ensure the bucket exists
        try:
            if not self.minio_client.bucket_exists(settings.MINIO_BUCKET):
                self.minio_client.make_bucket(settings.MINIO_BUCKET)
                log.info("MinIO bucket created", bucket=settings.MINIO_BUCKET)
        except S3Error as e:
            log.error("Error checking/creating MinIO bucket", error=str(e))
            raise

    def build_dataset(self, db: Session, version_name: str) -> str:
        """Builds a new dataset from reviewed annotations and uploads it to MinIO."""
        log.info("Building dataset", version_name=version_name)

        # TODO: Implement a robust dataset building process.
        # The current implementation is a placeholder that creates a dummy dataset.
        # A complete implementation should:
        # 1. Query the database for completed and validated annotation tasks.
        # 2. Retrieve the associated documents and their corrected/validated entities.
        # 3. Convert the data into a standard training format (e.g., IOB2, CoNLL, spaCy JSON).
        #    - This involves tokenizing the text and assigning IOB tags to each token.
        # 4. The generated dataset should be versioned and stored, and the version
        #    should be recorded in the `dataset_versions` table.

        # Query for all correct annotations and their associated entities and documents
        correct_annotations = db.query(models.Annotation, models.Entity, models.Document)\
            .join(models.Entity, models.Annotation.entity_id == models.Entity.id)\
            .join(models.Document, models.Entity.document_id == models.Document.id)\
            .filter(models.Annotation.is_correct == True)\
            .all()

        # Group entities by document
        documents_data = {}
        for annotation, entity, document in correct_annotations:
            if document.id not in documents_data:
                documents_data[document.id] = {
                    "text": document.text,
                    "labels": []
                }
            
            # Use corrected_label if available, otherwise use the original entity label
            label_to_use = annotation.corrected_label if annotation.corrected_label else entity.label
            
            documents_data[document.id]["labels"].append((entity.text, label_to_use))
        
        # Convert dictionary values to a list
        dataset_to_upload = list(documents_data.values())

        dataset_content = json.dumps(dataset_to_upload, indent=2)
        dataset_bytes = dataset_content.encode('utf-8')
        data_stream = io.BytesIO(dataset_bytes)
        data_length = len(dataset_bytes)

        object_name = f"datasets/{version_name}.json"
        try:
            self.minio_client.put_object(
                settings.MINIO_BUCKET,
                object_name,
                data_stream,
                data_length,
                content_type="application/json"
            )
            log.info("Dataset uploaded to MinIO", bucket=settings.MINIO_BUCKET, object_name=object_name)

            # Record dataset version in DB
            db_version = models.DatasetVersion(version_name=version_name, description="Generated from reviewed annotations", model_version="N/A")
            db.add(db_version)
            db.commit()
            db.refresh(db_version)
            log.info("Dataset version recorded in DB", version_id=db_version.id)

            return object_name
        except S3Error as e:
            log.error("Error uploading dataset to MinIO", error=str(e))
            raise
