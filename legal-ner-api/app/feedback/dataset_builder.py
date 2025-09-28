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

        # Placeholder: In a real scenario, this would query for completed annotation tasks,
        # retrieve associated documents and entities, and transform them into a training format (e.g., IOB2).
        # For now, we'll create a dummy dataset.

        dummy_data = [
            {"text": "La Corte di Cassazione", "labels": [("Corte di Cassazione", "ORG")]},
            {"text": "Il sig. Rossi è un avvocato.", "labels": [("Rossi", "PER"), ("avvocato", "PROFESSIONE")]}
        ]

        dataset_content = json.dumps(dummy_data, indent=2)
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
