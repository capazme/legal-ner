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
from minio.error import S3Error
import io
import numpy as np
from sklearn.model_selection import train_test_split
from app.core.active_learning_config import get_active_learning_config
from app.core.config import settings # Import settings
from sqlalchemy.orm import Session
from app.database import models
from app.core.label_mapping import act_type_to_label

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

    def _convert_spans_to_iob(self, span_data: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
        """Converts a dataset from span-based format to IOB format."""
        iob_dataset = []
        for doc in span_data:
            text = doc.get("text")
            entities = doc.get("entities", [])
            if not text:
                continue

            tokenized_input = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.config.dataset.max_sequence_length
            )
            tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
            offset_mapping = tokenized_input["offset_mapping"]
            
            iob_tags = ["O"] * len(tokens)

            for entity in sorted(entities, key=lambda x: x["start_char"]):
                standard_label = act_type_to_label(entity["label"])
                start_char = entity["start_char"]
                end_char = entity["end_char"]

                entity_token_indices = []
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start == token_end: continue
                    if max(start_char, token_start) < min(end_char, token_end):
                        entity_token_indices.append(i)
                
                if entity_token_indices:
                    if iob_tags[entity_token_indices[0]] == "O":
                        iob_tags[entity_token_indices[0]] = f"B-{standard_label}"
                        for idx in entity_token_indices[1:]:
                            if iob_tags[idx] == "O":
                                iob_tags[idx] = f"I-{standard_label}"

            iob_dataset.append({"tokens": tokens, "tags": iob_tags})
        return iob_dataset

    def train_model(self, db: Session, version: str, dataset_path: str, model_name: str = None, output_dir: str = None) -> str:
        """
        Fine-tunes a model, evaluates it against a golden test set, saves it to MinIO, and records metadata in the database.
        """
        if not model_name or '/' not in model_name:
            log.warning("Invalid or missing model name provided. Falling back to base model from config.", 
                        provided_name=model_name, 
                        fallback_model=self.config.training.base_model)
            model_name = self.config.training.base_model

        log.info("Starting model training", dataset=dataset_path, model=model_name, version=version)

        training_data = self._load_dataset_from_minio(dataset_path)
        if not training_data or len(training_data) < self.config.training.min_training_samples:
            raise ValueError(f"Insufficient training data: {len(training_data)} samples (minimum {self.config.training.min_training_samples} required)")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            golden_dataset_path = "data/golden_dataset.jsonl"
            with open(golden_dataset_path, 'r', encoding='utf-8') as f:
                span_based_golden_data = [json.loads(line) for line in f if line.strip()]
            
            log.info("Converting golden dataset to IOB format...")
            golden_data_iob = self._convert_spans_to_iob(span_based_golden_data, tokenizer)
            log.info("Loaded and converted golden test set for final evaluation", path=golden_dataset_path, samples=len(golden_data_iob))
        except Exception as e:
            log.warning("Could not load or convert golden test set. Final evaluation will be on the validation set.", error=str(e))
            golden_data_iob = None

        train_data, eval_data = train_test_split(training_data, test_size=self.config.training.eval_split, random_state=42)
        log.info(f"Dataset split: {len(train_data)} train, {len(eval_data)} validation")

        train_dataset = self._prepare_dataset(train_data, tokenizer)
        eval_dataset = self._prepare_dataset(eval_data, tokenizer)
        test_dataset = self._prepare_dataset(golden_data_iob, tokenizer) if golden_data_iob else None

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            warmup_steps=self.config.training.warmup_steps,
            weight_decay=self.config.training.weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy=self.config.training.eval_strategy,
            save_strategy=self.config.training.save_strategy,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            learning_rate=self.config.training.learning_rate,
            save_total_limit=self.config.training.save_total_limit,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics
        )

        log.info("Starting training...")
        trainer.train()

        log.info("Evaluating model on the golden test set...")
        if test_dataset:
            final_metrics = trainer.evaluate(eval_dataset=test_dataset)
            log.info("Final evaluation results on golden test set", **final_metrics)
        else:
            log.warning("No golden test set. Using validation set metrics as final metrics.")
            final_metrics = trainer.evaluate()
            log.info("Final evaluation results on validation set", **final_metrics)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        label_config = {"label_list": self.label_list, "label2id": self.label2id, "id2label": self.id2label}
        with open(os.path.join(output_dir, "label_config.json"), "w") as f:
            json.dump(label_config, f, indent=2)
        
        minio_model_path = f"models/{version}"
        log.info("Uploading model to MinIO", path=minio_model_path)
        for root, _, files in os.walk(output_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                minio_path = os.path.join(minio_model_path, os.path.relpath(local_path, output_dir)).replace("\\", "/")
                try:
                    self.minio_client.fput_object(self.config.minio.bucket, minio_path, local_path)
                except S3Error as e:
                    log.error("Failed to upload to MinIO", file=filename, error=str(e))
                    raise
        log.info("Model uploaded to MinIO successfully.")

        try:
            new_model_entry = models.TrainedModel(
                model_name=model_name,
                version=version,
                path=minio_model_path,
                description=f"Fine-tuned from {model_name} on dataset {dataset_path}",
                accuracy=final_metrics.get("eval_accuracy"),
                f1_score=final_metrics.get("eval_f1"),
                precision=final_metrics.get("eval_precision"),
                recall=final_metrics.get("eval_recall"),
                is_active=False
            )
            db.add(new_model_entry)
            db.commit()
            log.info("Saved trained model metadata to database", version=version, minio_path=minio_model_path)
        except Exception as e:
            log.error("Failed to save model metadata to database", error=str(e))
            db.rollback()
            raise
            
        log.info("Model training completed", output_dir=output_dir, minio_path=minio_model_path)

        return minio_model_path
    
    def _load_dataset_from_minio(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Loads a dataset from MinIO."""
        try:
            response = self.minio_client.get_object(
                self.config.minio.bucket,
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
        if not data:
            return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})

        tokenized_inputs = []
        aligned_labels = []

        for item in data:
            tokens = item["tokens"]
            tags = item["tags"]

            label_ids = [self.label2id.get(tag, self.label2id["O"]) for tag in tags]

            tokenized_input = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                padding=False
            )

            word_ids = tokenized_input.word_ids()
            aligned_label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    aligned_label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    aligned_label_ids.append(label_ids[word_idx])
                else:
                    aligned_label_ids.append(-100)

                previous_word_idx = word_idx

            tokenized_inputs.append(tokenized_input)
            aligned_labels.append(aligned_label_ids)

        dataset_dict = {
            "input_ids": [t["input_ids"] for t in tokenized_inputs],
            "attention_mask": [t["attention_mask"] for t in tokenized_inputs],
            "labels": aligned_labels
        }

        return Dataset.from_dict(dataset_dict)