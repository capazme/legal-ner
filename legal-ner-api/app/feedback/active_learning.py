from sqlalchemy.orm import Session, subqueryload
from sqlalchemy.sql import func
from app.database import models
from app.services.specialized_pipeline import LegalSourceExtractionPipeline
from app.core.config import settings
from app.core.active_learning_config import get_active_learning_config
from app.feedback.dataset_builder import DatasetBuilder
import structlog
from typing import List, Dict, Any, Optional
import random
from datetime import datetime
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
import os
import json
import numpy as np

log = structlog.get_logger()

class ActiveLearningManager:
    """
    Manages the active learning process for iteratively improving NER models.
    This version is refactored for efficiency and proper uncertainty handling.
    """

    def __init__(self):
        log.info("Initializing ActiveLearningManager")
        self.config = get_active_learning_config()
        self.dataset_builder = DatasetBuilder()
        # Initialize the pipeline once to be reused
        self.pipeline = LegalSourceExtractionPipeline()

    async def create_tasks_for_uncertain_documents(self, db: Session, batch_size: int = 10) -> Dict[str, Any]:
        """
        Efficiently identifies uncertain documents, creates annotation tasks with a priority score,
        and saves the pre-computed entities in a single pass.

        Returns:
            A dictionary with the status and details of the operation.
        """
        log.info("Starting uncertain document processing", batch_size=batch_size)

        # 1. Get documents that do not have an associated annotation task yet
        subquery = db.query(models.AnnotationTask.document_id).distinct()
        candidate_docs = db.query(models.Document).filter(models.Document.id.notin_(subquery)).limit(batch_size * 5).all()

        if not candidate_docs:
            log.info("No new documents found to process for active learning.")
            return {"status": "no_new_documents", "message": "No new documents found to process."}

        log.info(f"Found {len(candidate_docs)} new documents to analyze for uncertainty.")

        # 2. Process each document ONCE to get entities and calculate uncertainty
        doc_scores = []
        for doc in candidate_docs:
            try:
                # Run the pipeline only once
                entities = await self.pipeline.extract_legal_sources(doc.text)

                if not entities:
                    continue

                # Calculate uncertainty score for the document.
                # Metric: 1.0 - average_confidence. High score = high uncertainty.
                confidences = [e.get("confidence", 0.0) for e in entities]
                avg_confidence = np.mean(confidences)
                uncertainty_score = 1.0 - avg_confidence

                doc_scores.append({
                    "doc_id": doc.id,
                    "uncertainty": uncertainty_score,
                    "entities": entities
                })

            except Exception as e:
                log.error("Error processing document for uncertainty", doc_id=doc.id, error=str(e))

        if not doc_scores:
            log.info("Could not generate predictions for any of the new documents.")
            return {"status": "no_predictions", "message": "Prediction failed for all new documents."}

        # 3. Select the top 'batch_size' most uncertain documents
        doc_scores.sort(key=lambda x: x["uncertainty"], reverse=True)
        top_uncertain_docs = doc_scores[:batch_size]

        # 4. Create AnnotationTasks and Entities in the database
        created_tasks_count = 0
        task_ids = []
        for doc_data in top_uncertain_docs:
            session = db
            try:
                # Create Annotation Task with priority
                task = models.AnnotationTask(
                    document_id=doc_data["doc_id"],
                    status="pending",
                    priority=doc_data["uncertainty"], # Use the calculated uncertainty as priority
                    created_at=datetime.now()
                )
                session.add(task)
                session.flush() # Flush to get task.id for association

                # Create entities from the pre-computed results
                # Usa la mappatura centralizzata per le label
                from app.core.label_mapping import act_type_to_label as convert_act_type_to_label

                for entity_data in doc_data["entities"]:
                    # Converti valori a tipi Python nativi per evitare errori con PostgreSQL
                    import torch
                    import numpy as np

                    start_char_val = entity_data.get("start_char", 0)
                    end_char_val = entity_data.get("end_char", 0)
                    confidence_val = entity_data.get("confidence", 0.0)

                    # Converti tensori/numpy a int/float Python nativi
                    if isinstance(start_char_val, (torch.Tensor, np.integer)):
                        start_char_val = int(start_char_val.item() if hasattr(start_char_val, 'item') else start_char_val)
                    else:
                        start_char_val = int(start_char_val)

                    if isinstance(end_char_val, (torch.Tensor, np.integer)):
                        end_char_val = int(end_char_val.item() if hasattr(end_char_val, 'item') else end_char_val)
                    else:
                        end_char_val = int(end_char_val)

                    if isinstance(confidence_val, (torch.Tensor, np.floating, np.integer)):
                        confidence_val = float(confidence_val.item() if hasattr(confidence_val, 'item') else confidence_val)
                    else:
                        confidence_val = float(confidence_val)

                    # Converti act_type in label standardizzata
                    act_type = entity_data.get("act_type", "unknown")
                    label = convert_act_type_to_label(act_type)

                    entity = models.Entity(
                        document_id=doc_data["doc_id"],
                        text=entity_data.get("text", ""),
                        label=label,
                        start_char=start_char_val,
                        end_char=end_char_val,
                        confidence=confidence_val,
                        model=self.pipeline.config.models.entity_detector_primary # Or a version string
                    )
                    session.add(entity)

                session.commit()
                task_ids.append(task.id)
                created_tasks_count += 1
            except Exception as e:
                log.error("Error creating annotation task and entities", doc_id=doc_data["doc_id"], error=str(e))
                session.rollback()

        log.info(f"Successfully created {created_tasks_count} annotation tasks.")
        return {
            "status": "tasks_created",
            "document_count": created_tasks_count,
            "task_ids": task_ids,
            "message": f"Successfully created {created_tasks_count} annotation tasks for the most uncertain documents."
        }

    async def run_active_learning_iteration(self, db: Session, batch_size: int = 10) -> Dict[str, Any]:
        """
        Runs one complete iteration of the active learning process by calling the
        refactored and efficient task creation method.
        """
        log.info("Starting active learning iteration", batch_size=batch_size)
        return await self.create_tasks_for_uncertain_documents(db, batch_size)

    def train_model_with_feedback(self, db: Session, model_name: str) -> str:
        """
        Trains a model using the collected feedback and validated annotations.
        (This part remains the same)
        """
        log.info("Training model with feedback", model_name=model_name)

        dataset_version = f"{self.config.versioning.version_prefix}_{datetime.now().strftime(self.config.versioning.timestamp_format)}"
        dataset_path = self.dataset_builder.build_dataset(db, dataset_version)
        dataset_path_iob = dataset_path.replace(".json", "_iob.json")

        from app.models.model_trainer import ModelTrainer
        model_trainer = ModelTrainer()
        output_dir = self.config.get_model_output_dir(dataset_version)

        try:
            trained_model_path = model_trainer.train_model(
                db=db, # Add db
                version=dataset_version, # Add version
                dataset_path=dataset_path_iob,
                model_name=model_name,
                output_dir=output_dir
            )
            log.info("Model training completed successfully", model_path=trained_model_path)
            return trained_model_path
        except Exception as e:
            log.error("Error during model training", error=str(e))
            raise
