from typing import List, Tuple, Dict, Any
from app.core.config import settings
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import structlog
from app.services.semantic_validator import SemanticValidator
from app.services.entity_merger import EntityMerger
from app.services.confidence_calibrator import ConfidenceCalibrator

log = structlog.get_logger()

class EnsemblePredictor:
    def __init__(self, validator: SemanticValidator = SemanticValidator(), merger: EntityMerger = EntityMerger(), calibrator: ConfidenceCalibrator = ConfidenceCalibrator()):
        self.models = self._load_models()
        self.validator = validator
        self.merger = merger
        self.calibrator = calibrator

    def _load_models(self) -> List[Tuple[AutoModelForTokenClassification, AutoTokenizer]]:
        """Loads all models specified in the configuration."""
        models = []
        log.info("Loading models...", models=settings.ENSEMBLE_MODELS)
        for model_name in settings.ENSEMBLE_MODELS:
            try:
                log.info(f"Loading model: {model_name}")
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                models.append((model, tokenizer))
                log.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                log.error("Failed to load model", model_name=model_name, error=str(e))
                raise
        return models

    async def predict(self, text: str) -> Tuple[List[Dict[str, Any]], bool, float]:
        """
        Enhanced NER prediction with proper character alignment and ensemble voting.
        """
        log.info("Starting ensemble prediction", input_text_length=len(text), num_models=len(self.models))
        if not self.models:
            log.warning("No models loaded, returning empty prediction.")
            return [], False, 0.0

        all_processed_entities = []
        all_uncertainties = []
        model_predictions = []

        # Run prediction for each model
        for model_idx, (model, tokenizer) in enumerate(self.models):
            model_name = settings.ENSEMBLE_MODELS[model_idx]
            log.info("Running prediction for model", model_name=model_name)

            try:
                entities, uncertainty = await self._predict_single_model(model, tokenizer, text, model_name)
                all_processed_entities.extend(entities)
                all_uncertainties.append(uncertainty)
                model_predictions.append(entities)
                log.info("Model prediction complete", model_name=model_name, entities_count=len(entities), uncertainty=uncertainty)
            except Exception as e:
                log.error("Model prediction failed", model_name=model_name, error=str(e))
                continue

        if not all_processed_entities:
            log.warning("No successful predictions from any model")
            return [], True, 1.0

        # Calculate ensemble uncertainty
        overall_uncertainty = self._calculate_ensemble_uncertainty(all_uncertainties, model_predictions)
        requires_review = overall_uncertainty > settings.UNCERTAINTY_THRESHOLD

        log.info("Ensemble prediction complete", total_entities=len(all_processed_entities),
                overall_uncertainty=overall_uncertainty, requires_review=requires_review)

        # Apply consensus and validation
        merged_entities = self._semantic_consensus(all_processed_entities)
        calibrated_entities = self.calibrator.calibrate(merged_entities)

        return calibrated_entities, requires_review, overall_uncertainty

    async def _predict_single_model(self, model, tokenizer, text: str, model_name: str) -> Tuple[List[Dict[str, Any]], float]:
        """Enhanced single model prediction with proper character alignment."""

        # Tokenize with return_offsets_mapping for accurate character alignment
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,
            max_length=512
        )

        offset_mapping = inputs.pop("offset_mapping")[0]  # Remove from inputs, keep for alignment
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=2)

        predictions = torch.argmax(logits, dim=2)

        # Extract entities with proper character alignment
        entities = []
        current_entity = None

        for i, (token_id, offset) in enumerate(zip(predictions[0], offset_mapping)):
            if offset[0] == offset[1]:  # Skip special tokens ([CLS], [SEP], etc.)
                continue

            label = model.config.id2label[token_id.item()]
            score = probabilities[0][i][token_id.item()].item()

            if label.startswith("B-"):
                # Finish previous entity
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, text))

                # Start new entity
                current_entity = {
                    "label": label[2:],
                    "start_char": offset[0].item(),
                    "end_char": offset[1].item(),
                    "scores": [score],
                    "token_indices": [i]
                }
            elif label.startswith("I-") and current_entity and label[2:] == current_entity["label"]:
                # Continue current entity
                current_entity["end_char"] = offset[1].item()
                current_entity["scores"].append(score)
                current_entity["token_indices"].append(i)
            else:
                # End current entity if not continuing
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, text))
                    current_entity = None

        # Don't forget the last entity
        if current_entity:
            entities.append(self._finalize_entity(current_entity, text))

        # Calculate model uncertainty
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)
        avg_entropy = torch.mean(entropy).item()

        # Process entities
        processed_entities = []
        for entity in entities:
            if entity["end_char"] > entity["start_char"]:  # Valid entity
                processed_entities.append({
                    "text": entity["text"],
                    "label": entity["label"],
                    "start_char": entity["start_char"],
                    "end_char": entity["end_char"],
                    "confidence": entity["confidence"],
                    "model": model_name,
                    "token_count": len(entity["token_indices"])
                })

        return processed_entities, avg_entropy

    def _finalize_entity(self, entity_info: Dict, text: str) -> Dict[str, Any]:
        """Finalize entity with text extraction and confidence calculation."""
        start_char = entity_info["start_char"]
        end_char = entity_info["end_char"]

        # Extract actual text
        entity_text = text[start_char:end_char].strip()

        # Calculate average confidence
        avg_confidence = sum(entity_info["scores"]) / len(entity_info["scores"])

        return {
            "text": entity_text,
            "label": entity_info["label"],
            "start_char": start_char,
            "end_char": end_char,
            "confidence": avg_confidence,
            "token_indices": entity_info["token_indices"]
        }

    def _calculate_ensemble_uncertainty(self, uncertainties: List[float], model_predictions: List[List[Dict]]) -> float:
        """Calculate ensemble uncertainty considering model agreement."""
        if not uncertainties:
            return 1.0

        # Base uncertainty from individual models
        avg_uncertainty = sum(uncertainties) / len(uncertainties)

        # Calculate disagreement between models
        disagreement_factor = self._calculate_model_disagreement(model_predictions)

        # Combine uncertainties: higher disagreement = higher uncertainty
        ensemble_uncertainty = avg_uncertainty * (1 + disagreement_factor)

        return min(ensemble_uncertainty, 1.0)  # Cap at 1.0

    def _calculate_model_disagreement(self, model_predictions: List[List[Dict]]) -> float:
        """Calculate disagreement factor between model predictions."""
        if len(model_predictions) < 2:
            return 0.0

        # Simple disagreement: ratio of unique entities to total entities
        all_entities = []
        for predictions in model_predictions:
            for entity in predictions:
                # Create a signature for comparison
                signature = f"{entity['text'].lower()}:{entity['label']}:{entity['start_char']}"
                all_entities.append(signature)

        if not all_entities:
            return 0.0

        unique_entities = len(set(all_entities))
        total_entities = len(all_entities)

        # Disagreement factor: more unique entities relative to total = more disagreement
        disagreement = (unique_entities / total_entities) - (1 / len(model_predictions))
        return max(0.0, disagreement)

    def _semantic_consensus(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges entities based on exact text and label matching, prioritizing higher confidence."""
        log.info("Applying semantic consensus (exact match)", input_entities_count=len(entities))
        
        if not entities:
            return []

        # First, apply the more complex merging logic from EntityMerger
        pre_merged_entities = self.merger.merge_entities(entities)

        # Then, apply exact match consensus on the pre-merged entities
        unique_entities: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for entity in pre_merged_entities:
            key = (entity["text"].lower(), entity["label"])
            if key not in unique_entities:
                unique_entities[key] = entity
            else:
                # If a duplicate is found, keep the one with higher confidence
                if entity["confidence"] > unique_entities[key]["confidence"]:
                    unique_entities[key] = entity
        
        merged_entities = list(unique_entities.values())
        log.info("Semantic consensus complete", output_entities_count=len(merged_entities))
        
        # Validate entities semantically
        validated_entities = self.validator.validate_entities(merged_entities)
        return validated_entities

    
