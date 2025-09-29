from typing import List, Dict, Any, Tuple
import numpy as np
import structlog
from datetime import datetime, timedelta

log = structlog.get_logger()

class ConfidenceCalibrator:
    def __init__(self):
        log.info("Initializing Enhanced ConfidenceCalibrator")
        self._initialize_calibration_parameters()

    def _initialize_calibration_parameters(self):
        """Initialize calibration parameters based on legal NER characteristics."""

        # Label-specific confidence adjustments based on difficulty
        self.label_adjustments = {
            "NORMATIVA": 1.1,      # Legal texts usually identify normative sources well
            "ORG": 0.95,           # Organizations can be ambiguous
            "PER": 0.9,            # Person names can be challenging in legal contexts
            "GIURISPRUDENZA": 1.05, # Case law is usually well-structured
            "CONCETTO_GIURIDICO": 0.85,  # Legal concepts can be subjective
            "LOC": 0.9,            # Locations
            "MISC": 0.8            # Miscellaneous entities are often uncertain
        }

        # Entity length impact on confidence
        self.length_calibration = {
            "very_short": (1, 3, 0.8),     # 1-3 chars: usually abbreviations, less reliable
            "short": (4, 10, 0.95),        # 4-10 chars: normal, slight penalty
            "medium": (11, 25, 1.0),       # 11-25 chars: optimal length
            "long": (26, 50, 0.98),        # 26-50 chars: slight penalty for very long
            "very_long": (51, 999, 0.85)   # 50+ chars: likely errors or overly long spans
        }

        # Model-specific reliability factors
        self.model_reliability = {
            "dlicari/distil-ita-legal-bert": 1.05,  # Specialized legal model
            "DeepMount00/Italian_NER_XXL_v2": 0.98,  # General Italian model
            "default": 1.0
        }

        # Ensemble agreement bonus
        self.agreement_bonus = {
            1: 1.0,    # Single model
            2: 1.15,   # Two models agree
            3: 1.25,   # Three models agree
            4: 1.3     # Four or more models agree
        }

    def calibrate(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhanced confidence calibration using multiple factors.
        """
        log.debug("Calibrating confidence", input_entities_count=len(entities))

        if not entities:
            return entities

        # Group entities by text and label for ensemble analysis
        entity_groups = self._group_entities_for_ensemble_analysis(entities)

        calibrated_entities = []

        for group_key, group_entities in entity_groups.items():
            # Calculate ensemble confidence for this group
            calibrated_entity = self._calibrate_entity_group(group_entities)
            if calibrated_entity:
                calibrated_entities.append(calibrated_entity)

        # Apply global normalization
        calibrated_entities = self._apply_global_normalization(calibrated_entities)

        log.debug("Confidence calibration complete",
                 output_entities_count=len(calibrated_entities),
                 avg_confidence=np.mean([e["confidence"] for e in calibrated_entities]) if calibrated_entities else 0)

        return calibrated_entities

    def _group_entities_for_ensemble_analysis(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group entities by text and label for ensemble analysis."""
        groups = {}

        for entity in entities:
            # Create a key based on text, label, and approximate position
            text_key = entity["text"].lower().strip()
            label_key = entity["label"]
            position_bucket = entity["start_char"] // 10  # Group nearby positions

            group_key = f"{text_key}:{label_key}:{position_bucket}"

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(entity)

        return groups

    def _calibrate_entity_group(self, group_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calibrate confidence for a group of similar entities."""
        if not group_entities:
            return None

        # Use the entity with highest original confidence as base
        base_entity = max(group_entities, key=lambda x: x.get("confidence", 0))
        calibrated_entity = base_entity.copy()

        # Calculate ensemble factors
        num_models = len(set(e.get("model", "unknown") for e in group_entities))
        avg_confidence = np.mean([e.get("confidence", 0) for e in group_entities])
        max_confidence = max(e.get("confidence", 0) for e in group_entities)
        confidence_variance = np.var([e.get("confidence", 0) for e in group_entities])

        # Start with average confidence
        calibrated_confidence = avg_confidence

        # Apply label-specific adjustment
        label = base_entity["label"]
        label_factor = self.label_adjustments.get(label, 1.0)
        calibrated_confidence *= label_factor

        # Apply length-based adjustment
        text_length = len(base_entity["text"])
        length_factor = self._get_length_factor(text_length)
        calibrated_confidence *= length_factor

        # Apply model reliability factor
        model_name = base_entity.get("model", "default")
        model_factor = self.model_reliability.get(model_name, self.model_reliability["default"])
        calibrated_confidence *= model_factor

        # Apply ensemble agreement bonus
        agreement_factor = self.agreement_bonus.get(num_models, self.agreement_bonus[4])
        calibrated_confidence *= agreement_factor

        # Penalty for high variance (models disagree on confidence)
        if confidence_variance > 0.1:
            variance_penalty = max(0.8, 1 - (confidence_variance * 2))
            calibrated_confidence *= variance_penalty

        # Apply validation score if available
        validation_score = base_entity.get("validation_score", 1.0)
        if validation_score < 1.0:
            calibrated_confidence *= validation_score

        # Ensure confidence stays within bounds
        calibrated_confidence = max(0.01, min(0.99, calibrated_confidence))

        # Add calibration metadata
        calibrated_entity["confidence"] = calibrated_confidence
        calibrated_entity["original_confidence"] = avg_confidence
        calibrated_entity["ensemble_size"] = num_models
        calibrated_entity["confidence_variance"] = confidence_variance
        calibrated_entity["calibration_factors"] = {
            "label_factor": label_factor,
            "length_factor": length_factor,
            "model_factor": model_factor,
            "agreement_factor": agreement_factor,
            "validation_score": validation_score
        }

        return calibrated_entity

    def _get_length_factor(self, text_length: int) -> float:
        """Get confidence adjustment factor based on entity text length."""
        for category, (min_len, max_len, factor) in self.length_calibration.items():
            if min_len <= text_length <= max_len:
                return factor
        return 0.8  # Default penalty for very unusual lengths

    def _apply_global_normalization(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply global confidence normalization to maintain reasonable distribution."""
        if not entities:
            return entities

        confidences = [e["confidence"] for e in entities]

        # Calculate statistics
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)

        # Apply mild normalization if confidence distribution is too skewed
        if mean_conf > 0.9:  # Too confident
            normalization_factor = 0.85 / mean_conf
            for entity in entities:
                entity["confidence"] *= normalization_factor
                entity["confidence"] = max(0.01, min(0.99, entity["confidence"]))
        elif mean_conf < 0.4:  # Too uncertain
            normalization_factor = 0.6 / mean_conf
            for entity in entities:
                entity["confidence"] *= normalization_factor
                entity["confidence"] = max(0.01, min(0.99, entity["confidence"]))

        return entities

    def get_calibration_stats(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get calibration statistics for analysis."""
        if not entities:
            return {}

        calibrated_confs = [e["confidence"] for e in entities]
        original_confs = [e.get("original_confidence", e["confidence"]) for e in entities]

        return {
            "entity_count": len(entities),
            "avg_confidence_before": np.mean(original_confs),
            "avg_confidence_after": np.mean(calibrated_confs),
            "confidence_std_before": np.std(original_confs),
            "confidence_std_after": np.std(calibrated_confs),
            "confidence_range_before": (min(original_confs), max(original_confs)),
            "confidence_range_after": (min(calibrated_confs), max(calibrated_confs)),
            "label_distribution": self._get_label_distribution(entities),
            "ensemble_sizes": self._get_ensemble_size_distribution(entities)
        }

    def _get_label_distribution(self, entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of entity labels."""
        distribution = {}
        for entity in entities:
            label = entity["label"]
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def _get_ensemble_size_distribution(self, entities: List[Dict[str, Any]]) -> Dict[int, int]:
        """Get distribution of ensemble sizes."""
        distribution = {}
        for entity in entities:
            size = entity.get("ensemble_size", 1)
            distribution[size] = distribution.get(size, 0) + 1
        return distribution
