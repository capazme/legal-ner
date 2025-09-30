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
    """
    Costruisce dataset per training da annotazioni database-backed.

    Non dipende più da FeedbackLoop, ma query direttamente il database.
    """

    def __init__(self):
        log.info("Initializing DatasetBuilder (Database-backed)")

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
        """
        Costruisce un nuovo dataset dalle annotazioni validate nel database.

        Args:
            db: Database session
            version_name: Nome versione del dataset

        Returns:
            Path del dataset in MinIO
        """
        log.info("Building dataset from database annotations", version_name=version_name)

        # Ottieni documenti con annotazioni validate
        # Query per ottenere tutti i documenti che hanno almeno un'annotazione
        documents_with_annotations = (
            db.query(models.Document)
            .join(models.Entity, models.Document.id == models.Entity.document_id)
            .join(models.Annotation, models.Entity.id == models.Annotation.entity_id)
            .distinct()
            .all()
        )

        if not documents_with_annotations:
            log.warning("No documents with annotations found for dataset building")
            raise ValueError("No annotated documents found to build training dataset.")

        log.info(f"Found {len(documents_with_annotations)} documents with annotations")

        # Costruisci dataset in formato span-based
        dataset_to_upload = []

        for document in documents_with_annotations:
            # Ottieni tutte le entità per questo documento che hanno annotazioni
            entities = (
                db.query(models.Entity)
                .filter(models.Entity.document_id == document.id)
                .join(models.Annotation, models.Entity.id == models.Annotation.entity_id)
                .distinct()
                .all()
            )

            # Per ogni entità, verifica se è stata validata come corretta (majority voting)
            validated_entities = []
            for entity in entities:
                annotations = db.query(models.Annotation).filter(
                    models.Annotation.entity_id == entity.id
                ).all()

                if not annotations:
                    continue

                # Majority voting
                correct_count = sum(1 for a in annotations if a.is_correct)
                incorrect_count = len(annotations) - correct_count

                # Include solo se la maggioranza dice che è corretta
                if correct_count > incorrect_count:
                    validated_entities.append({
                        "text": entity.text,
                        "label": entity.label,
                        "start_char": entity.start_char,
                        "end_char": entity.end_char
                    })
                # Se marcata come incorretta e c'è una label corretta, usa quella
                elif incorrect_count > correct_count:
                    # Cerca se c'è una corrected_label
                    corrected_labels = [a.corrected_label for a in annotations if a.corrected_label]
                    if corrected_labels:
                        # Usa la label corretta più frequente
                        most_common_label = max(set(corrected_labels), key=corrected_labels.count)
                        validated_entities.append({
                            "text": entity.text,
                            "label": most_common_label,
                            "start_char": entity.start_char,
                            "end_char": entity.end_char
                        })

            # Aggiungi documento al dataset solo se ha entità validate
            if validated_entities:
                dataset_to_upload.append({
                    "text": document.text,
                    "labels": validated_entities
                })

        if not dataset_to_upload:
            log.warning("No validated entities found for dataset building")
            raise ValueError("No validated entities found to build training dataset.")

        log.info(f"Dataset built with {len(dataset_to_upload)} documents and {sum(len(d['labels']) for d in dataset_to_upload)} entities")

        # Converti a formato IOB per token classification
        iob_dataset = self._convert_to_iob_format(dataset_to_upload)

        if not iob_dataset:
            log.error("IOB conversion failed or produced empty dataset")
            raise ValueError("Failed to convert dataset to IOB format")

        # Upload entrambi i formati
        self._upload_dataset(dataset_to_upload, f"datasets/{version_name}.json")
        self._upload_dataset(iob_dataset, f"datasets/{version_name}_iob.json")

        # Record dataset version in DB
        db_version = models.DatasetVersion(
            version_name=version_name,
            description=f"Generated from {len(dataset_to_upload)} annotated documents",
            model_version="N/A"
        )
        db.add(db_version)
        db.commit()
        db.refresh(db_version)
        log.info("Dataset version recorded in DB", version_id=db_version.id, version_name=version_name)

        return f"datasets/{version_name}.json"

    def _convert_to_iob_format(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts span-based annotations to IOB format for token classification training.
        Uses tokenizer's offset_mapping for robust token-to-char alignment.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.training.base_model)

            iob_dataset = []

            for doc_idx, doc in enumerate(dataset):
                text = doc["text"]
                labels = doc["labels"]

                log.debug("IOB conversion: Processing document", doc_idx=doc_idx, text_len=len(text), num_labels=len(labels))

                # Tokenize the text with offset mapping
                tokenized_input = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=self.config.dataset.max_sequence_length
                )

                tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
                offset_mapping = tokenized_input["offset_mapping"]

                log.debug("IOB conversion: Tokenization complete", num_tokens=len(tokens))

                # Initialize IOB tags
                iob_tags = ["O"] * len(tokens)

                # Sort labels by start position per evitare sovrapposizioni
                sorted_labels = sorted(labels, key=lambda x: x["start_char"])

                log.debug("IOB conversion: Sorted labels", num_labels=len(sorted_labels))

                for label_info in sorted_labels:
                    label = label_info["label"]
                    start_char = label_info["start_char"]
                    end_char = label_info["end_char"]

                    log.debug("IOB conversion: Processing label", label=label, start_char=start_char, end_char=end_char)

                    # Find which tokens overlap with this entity using offset_mapping
                    entity_token_indices = []
                    for i, (token_start, token_end) in enumerate(offset_mapping):
                        # Skip special tokens (where start == end)
                        if token_start == token_end:
                            continue

                        # Check for overlap
                        # Un token appartiene all'entità se c'è sovrapposizione tra gli span
                        if max(start_char, token_start) < min(end_char, token_end):
                            entity_token_indices.append(i)

                    log.debug("IOB conversion: Entity token indices found", label=label, entity_token_indices=entity_token_indices)

                    # Assign IOB tags
                    if entity_token_indices:
                        # Solo se non è già stato marcato (per evitare sovrapposizioni)
                        if iob_tags[entity_token_indices[0]] == "O":
                            iob_tags[entity_token_indices[0]] = f"B-{label}"
                            for idx in entity_token_indices[1:]:
                                if iob_tags[idx] == "O":
                                    iob_tags[idx] = f"I-{label}"

                log.debug("IOB conversion: Final IOB tags for document", num_B_tags=sum(1 for t in iob_tags if t.startswith("B-")))

                iob_dataset.append({
                    "tokens": tokens,
                    "tags": iob_tags,
                    "text": text # Keep original text for reference
                })

            log.info("IOB conversion: Successfully converted documents",
                    num_docs=len(iob_dataset),
                    total_entities=sum(sum(1 for t in doc["tags"] if t.startswith("B-")) for doc in iob_dataset))
            return iob_dataset

        except Exception as e:
            log.error("Error converting to IOB format", error=str(e), exc_info=True)
            # Return empty dataset on error
            return []

    def _upload_dataset(self, dataset: Any, object_name: str) -> None:
        """Uploads a dataset to MinIO."""
        dataset_content = json.dumps(dataset, indent=2, ensure_ascii=False)
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
            log.info("Dataset uploaded to MinIO",
                    bucket=self.config.minio.bucket,
                    object_name=object_name,
                    size_bytes=data_length)
        except S3Error as e:
            log.error("Error uploading dataset to MinIO", error=str(e))
            raise
