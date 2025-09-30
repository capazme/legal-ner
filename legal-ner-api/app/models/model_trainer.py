import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, DatasetDict
import json
import os
import structlog
from typing import Dict, List, Any, Optional
from minio import Minio
import io
import numpy as np
from sklearn.model_selection import train_test_split
from app.core.active_learning_config import get_active_learning_config
from app.core.config import settings # Import settings
from sqlalchemy.orm import Session
from app.database import models

log = structlog.get_logger()

class ModelTrainer:
    """
    Handles fine-tuning of models using collected feedback data.

    Uses configuration from active_learning_config.yaml.
    """

    def __init__(self):
        log.info("Initializing ModelTrainer")

        # Load configuration
        self.config = get_active_learning_config()

        # Initialize MinIO client
        self.minio_client = Minio(
            self.config.minio.endpoint,
            access_key=self.config.minio.access_key,
            secret_key=self.config.minio.secret_key,
            secure=self.config.minio.secure
        )

        # Load label configuration from YAML
        self.label_list = self.config.labels.label_list
        self.label2id = self.config.labels.label2id
        self.id2label = self.config.labels.id2label

        log.info("ModelTrainer initialized",
                 num_labels=len(self.label_list),
                 base_model=self.config.training.base_model)

    def _compute_metrics(self, p):
        from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (-100) and handle out-of-range predictions
        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            pred_labels = []
            true_label_list = []

            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:  # Skip padding
                    # Handle out-of-range predictions by mapping to "O"
                    if pred_id in self.id2label:
                        pred_labels.append(self.id2label[pred_id])
                    else:
                        pred_labels.append("O")

                    if label_id in self.id2label:
                        true_label_list.append(self.id2label[label_id])
                    else:
                        true_label_list.append("O")

            if pred_labels and true_label_list:
                true_predictions.append(pred_labels)
                true_labels.append(true_label_list)

        return {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    
    def train_model(self, db: Session, version: str, dataset_path: str, model_name: str = None, output_dir: str = None, eval_split: float = 0.2) -> str:
        """
        Fine-tunes a model, evaluates it, and saves the metadata to the database.

        Args:
            db: The database session.
            version: The unique version string for this model.
            dataset_path: Path to the dataset in MinIO (IOB format).
            model_name: Name of the base model to fine-tune. Defaults to config base_model.
            output_dir: Directory to save the fine-tuned model.
            eval_split: Fraction of data to use for evaluation.

        Returns:
            Path to the fine-tuned model.
        """
        # Use config base_model if model_name not provided or invalid
        if not model_name or not model_name.startswith(('dbmdz/', 'DeepMount00/', 'bert-', 'xlm-')):
            model_name = self.config.training.base_model
            log.info("Using base model from config", base_model=model_name)

        log.info("Starting model training", dataset=dataset_path, model=model_name, version=version)

        # Load dataset from MinIO (IOB format)
        training_data = self._load_dataset_from_minio(dataset_path)

        if not training_data or len(training_data) < 10:
            raise ValueError(f"Insufficient training data: {len(training_data)} samples (minimum 10 required)")

        # Split into train/eval
        train_data, eval_data = train_test_split(training_data, test_size=eval_split, random_state=42)
        log.info(f"Dataset split: {len(train_data)} train, {len(eval_data)} eval")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Prepare datasets with tokenization and alignment
        train_dataset = self._prepare_dataset(train_data, tokenizer)
        eval_dataset = self._prepare_dataset(eval_data, tokenizer)

        # Load model with correct number of labels
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True  # In case base model has different number of labels
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Define training arguments with better defaults
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="epoch",  # Updated from evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            learning_rate=2e-5,
            save_total_limit=2,
        )

        # Create data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics
        )

        # Train model
        log.info("Starting training...")
        trainer.train()

        # Evaluate model
        log.info("Evaluating model...")
        eval_results = trainer.evaluate()
        log.info("Evaluation results", **eval_results)

        # Save final model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save label mappings
        label_config = {
            "label_list": self.label_list,
            "label2id": self.label2id,
            "id2label": self.id2label
        }
        with open(os.path.join(output_dir, "label_config.json"), "w") as f:
            json.dump(label_config, f, indent=2)

        # Save model metadata to the database
        try:
            new_model_entry = models.TrainedModel(
                model_name=model_name,
                version=version,
                path=output_dir,
                description=f"Fine-tuned from {model_name} on dataset {dataset_path}",
                accuracy=eval_results.get("eval_accuracy"),
                f1_score=eval_results.get("eval_f1"),
                precision=eval_results.get("eval_precision"),
                recall=eval_results.get("eval_recall"),
                is_active=False # Do not activate automatically
            )
            db.add(new_model_entry)
            db.commit()
            log.info("Saved trained model metadata to database", version=version)
        except Exception as e:
            log.error("Failed to save model metadata to database", error=str(e))
            db.rollback()
            # The model is trained, but not registered. This is a problem to be handled.
            # For now, we just log it.
            
        log.info("Model training completed", output_dir=output_dir)

        return output_dir
    
    def _load_dataset_from_minio(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Loads a dataset from MinIO."""
        try:
            response = self.minio_client.get_object(
                self.config.minio.bucket, # Use the bucket from active_learning_config
                dataset_path
            )
            
            dataset_content = response.read().decode('utf-8')
            return json.loads(dataset_content)
        except Exception as e:
            log.error("Error loading dataset from MinIO", error=str(e))
            raise
    
    def _prepare_dataset(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> Dataset:
        """
        Converts IOB dataset to tokenized format for training.

        Handles:
        - Tokenization with proper alignment
        - Label conversion to IDs
        - Subword token handling
        """
        tokenized_inputs = []
        aligned_labels = []

        for item in data:
            tokens = item["tokens"]
            tags = item["tags"]

            # Convert tags to IDs
            label_ids = [self.label2id.get(tag, self.label2id["O"]) for tag in tags]

            # Tokenize and align labels
            tokenized_input = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                padding=False  # Will be done by data collator
            )

            # Align labels with subword tokens
            word_ids = tokenized_input.word_ids()
            aligned_label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    # Special token (CLS, SEP, PAD)
                    aligned_label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word
                    aligned_label_ids.append(label_ids[word_idx])
                else:
                    # Continuation subword - use -100 to ignore in loss
                    aligned_label_ids.append(-100)

                previous_word_idx = word_idx

            tokenized_inputs.append(tokenized_input)
            aligned_labels.append(aligned_label_ids)

        # Create dataset
        dataset_dict = {
            "input_ids": [t["input_ids"] for t in tokenized_inputs],
            "attention_mask": [t["attention_mask"] for t in tokenized_inputs],
            "labels": aligned_labels
        }

        return Dataset.from_dict(dataset_dict)