from typing import List, Tuple, Dict, Any
from app.core.config import settings
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import structlog
from app.services.semantic_validator import SemanticValidator
from app.services.entity_merger import EntityMerger

log = structlog.get_logger()

class EnsemblePredictor:
    def __init__(self, validator: SemanticValidator = SemanticValidator(), merger: EntityMerger = EntityMerger()):
        self.models = self._load_models()
        self.validator = validator
        self.merger = merger

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
        Performs NER prediction using the ensemble of models.
        For now, it uses only the first model.
        """
        log.info("Starting prediction", input_text_length=len(text))
        if not self.models:
            log.warning("No models loaded, returning empty prediction.")
            return [], False

        model, tokenizer = self.models[0]
        model_name = settings.ENSEMBLE_MODELS[0]
        all_processed_entities = []
        all_uncertainties = []

        for model_idx, (model, tokenizer) in enumerate(self.models):
            model_name = settings.ENSEMBLE_MODELS[model_idx]
            log.info("Running prediction for model", model_name=model_name)

            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=2)

            predictions = torch.argmax(logits, dim=2)
            
            entities = []
            current_entity = None

            for i, token_id in enumerate(predictions[0]):
                label = model.config.id2label[token_id.item()]
                score = probabilities[0][i][token_id.item()].item()

                if label.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "label": label[2:],
                        "tokens": [tokens[i]],
                        "scores": [score],
                        "start_token_index": i
                    }
                elif label.startswith("I-") and current_entity and label[2:] == current_entity["label"]:
                    current_entity["tokens"].append(tokens[i])
                    current_entity["scores"].append(score)
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = None
            
            if current_entity:
                entities.append(current_entity)

            processed_entities_for_model = []
            for entity in entities:
                entity_text = tokenizer.convert_tokens_to_string(entity["tokens"])
                start_char = text.find(entity_text)
                end_char = start_char + len(entity_text)
                avg_confidence = sum(entity["scores"]) / len(entity["scores"])

                processed_entities_for_model.append({
                    "text": entity_text,
                    "label": entity["label"],
                    "start_char": start_char,
                    "end_char": end_char,
                    "confidence": avg_confidence,
                    "model": model_name
                })

            # Calculate uncertainty (entropy) for this model
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)
            avg_entropy = torch.mean(entropy).item()

            all_processed_entities.extend(processed_entities_for_model)

            all_uncertainties.append(avg_entropy)
            log.info("Model prediction complete", model_name=model_name, entities_count=len(processed_entities_for_model), average_entropy=avg_entropy)

        # Simple ensemble uncertainty: average of all model uncertainties
        overall_uncertainty = sum(all_uncertainties) / len(all_uncertainties) if all_uncertainties else 0.0
        requires_review = overall_uncertainty > settings.UNCERTAINTY_THRESHOLD
        log.info("Overall ensemble prediction complete", total_entities=len(all_processed_entities), overall_uncertainty=overall_uncertainty, requires_review=requires_review, threshold=settings.UNCERTAINTY_THRESHOLD)

        # For now, we return all entities from all models. Semantic consensus will merge them later.
        merged_entities = self._semantic_consensus(all_processed_entities)
        calibrated_entities = self._calibrate_confidence(merged_entities)

        return calibrated_entities, requires_review, overall_uncertainty

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

    def _calibrate_confidence(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calibrates the confidence scores of the entities."""
        log.info("Calibrating confidence (placeholder)", input_entities_count=len(entities))
        # Placeholder: This would involve adjusting confidence scores based on ensemble agreement
        # or other calibration techniques.
        return entities
