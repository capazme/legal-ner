from sqlalchemy.orm import Session
from app.database import models
from app.core.config import settings
from app.core.active_learning_config import get_active_learning_config
import structlog
from minio import Minio
from minio.error import S3Error
import json
import io
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer

log = structlog.get_logger()

class DatasetBuilder:
    def __init__(self):
        log.info("Initializing DatasetBuilder")

        # Load active learning configuration
        self.config = get_active_learning_config()

        # Initialize MinIO client from config
        self.minio_client = Minio(
            self.config.minio.endpoint,
            access_key=self.config.minio.access_key,
            secret_key=self.config.minio.secret_key,
            secure=self.config.minio.secure
        )

        # Ensure the bucket exists
        try:
            if not self.minio_client.bucket_exists(self.config.minio.bucket):
                self.minio_client.make_bucket(self.config.minio.bucket)
                log.info("MinIO bucket created", bucket=self.config.minio.bucket)
        except S3Error as e:
            log.error("Error checking/creating MinIO bucket", error=str(e))
            raise

    def build_dataset(self, db: Session, version_name: str) -> str:
        """Builds a new dataset from reviewed annotations and uploads it to MinIO."""
        log.info("Building dataset", version_name=version_name)

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
            
            # Use corrected values if available, otherwise use the original entity values
            label_to_use = annotation.corrected_label if annotation.corrected_label else entity.label
            text_to_use = annotation.corrected_text if annotation.corrected_text else entity.text
            start_char = entity.start_char
            end_char = entity.end_char
            
            # If boundaries were corrected, use those instead
            if annotation.corrected_boundaries:
                boundaries = json.loads(annotation.corrected_boundaries)
                start_char = boundaries.get("start_char", start_char)
                end_char = boundaries.get("end_char", end_char)
            
            documents_data[document.id]["labels"].append({
                "text": text_to_use,
                "label": label_to_use,
                "start_char": start_char,
                "end_char": end_char
            })
        
        # Convert dictionary values to a list
        dataset_to_upload = list(documents_data.values())

        # Also create IOB format for token classification training
        iob_dataset = self._convert_to_iob_format(dataset_to_upload)
        
        # Upload both formats
        self._upload_dataset(dataset_to_upload, f"datasets/{version_name}.json")
        self._upload_dataset(iob_dataset, f"datasets/{version_name}_iob.json")

        # Record dataset version in DB
        db_version = models.DatasetVersion(
            version_name=version_name, 
            description="Generated from reviewed annotations", 
            model_version="N/A"
        )
        db.add(db_version)
        db.commit()
        db.refresh(db_version)
        log.info("Dataset version recorded in DB", version_id=db_version.id)

        return f"datasets/{version_name}.json"
    
    def _convert_to_iob_format(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts span-based annotations to IOB format for token classification training.
        """
        try:
            # Use tokenizer from active learning config
            tokenizer = AutoTokenizer.from_pretrained(self.config.training.base_model)
            
            iob_dataset = []
            
            for doc in dataset:
                text = doc["text"]
                labels = doc["labels"]
                
                # Sort labels by start position
                sorted_labels = sorted(labels, key=lambda x: x["start_char"])
                
                # Tokenize the text
                tokens = tokenizer.tokenize(text)
                token_spans = self._get_token_spans(text, tokens, tokenizer)
                
                # Create IOB tags
                iob_tags = ["O"] * len(tokens)
                
                for label_info in sorted_labels:
                    label = label_info["label"]
                    start_char = label_info["start_char"]
                    end_char = label_info["end_char"]
                    
                    # Find which tokens overlap with this entity
                    entity_token_indices = []
                    for i, (token_start, token_end) in enumerate(token_spans):
                        if token_end > start_char and token_start < end_char:
                            entity_token_indices.append(i)
                    
                    # Assign IOB tags
                    if entity_token_indices:
                        iob_tags[entity_token_indices[0]] = f"B-{label}"
                        for idx in entity_token_indices[1:]:
                            iob_tags[idx] = f"I-{label}"
                
                iob_dataset.append({
                    "tokens": tokens,
                    "tags": iob_tags,
                    "text": text
                })
                
            return iob_dataset
            
        except Exception as e:
            log.error("Error converting to IOB format", error=str(e))
            # Return empty dataset on error
            return []
    
    def _get_token_spans(self, text: str, tokens: List[str], tokenizer) -> List[tuple]:
        """
        Maps tokens to their character spans in the original text.
        This is a simplified approach and might need refinement for your specific tokenizer.
        """
        spans = []
        current_pos = 0
        
        for token in tokens:
            # Handle special tokens
            if token.startswith("##"):
                token = token[2:]
            
            # Find token in text
            token_stripped = token.replace("##", "")
            if token_stripped:
                token_pos = text.find(token_stripped, current_pos)
                if token_pos != -1:
                    spans.append((token_pos, token_pos + len(token_stripped)))
                    current_pos = token_pos + len(token_stripped)
                else:
                    # If token not found, use approximate position
                    spans.append((current_pos, current_pos + len(token_stripped)))
                    current_pos += len(token_stripped)
            else:
                # For special tokens with no text representation
                spans.append((current_pos, current_pos))
        
        return spans
    
    def _upload_dataset(self, dataset: Any, object_name: str) -> None:
        """Uploads a dataset to MinIO."""
        dataset_content = json.dumps(dataset, indent=2)
        dataset_bytes = dataset_content.encode('utf-8')
        data_stream = io.BytesIO(dataset_bytes)
        data_length = len(dataset_bytes)

        try:
            self.minio_client.put_object(
                self.config.minio.bucket,
                object_name,
                data_stream,
                data_length,
                content_type="application/json"
            )
            log.info("Dataset uploaded to MinIO", bucket=self.config.minio.bucket, object_name=object_name)
        except S3Error as e:
            log.error("Error uploading dataset to MinIO", error=str(e))
            raise
