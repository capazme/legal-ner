"""
Enhanced Ensemble Predictor with Three-Stage Pipeline
=====================================================

Sistema NER avanzato che implementa la strategia a 3 stadi:
1. Stage 1: Italian_NER_XXL_v2 per estrazione entità generiche
2. Stage 2: Italian-legal-bert per strutturazione legal-specific
3. Stage 3: Distil-bert per validation + correlazione semantica

Include integrazione completa con feedback loop e golden dataset.
"""

from typing import List, Dict, Any, Tuple
import structlog
from app.services.three_stage_predictor import ThreeStagePredictor
from app.services.semantic_correlator import SemanticCorrelator
from app.services.feedback_loop import FeedbackLoop

log = structlog.get_logger()

class EnsemblePredictor:
    """
    Ensemble Predictor con pipeline a 3 stadi e correlazione semantica.

    Architettura:
    - ThreeStagePredictor: Pipeline NER specializzata per dominio legale
    - SemanticCorrelator: Correlazione riferimenti normativi distanti
    - FeedbackLoop: Continuous learning con golden dataset

    Questa implementazione sostituisce l'approccio ensemble tradizionale
    con una pipeline specializzata per documenti legali italiani.
    """

    def __init__(self):
        log.info("Inizializzazione Enhanced EnsemblePredictor con pipeline a 3 stadi")

        try:
            # Core pipeline a 3 stadi
            self.three_stage_predictor = ThreeStagePredictor()

            # Correlatore semantico per riferimenti distanti
            self.semantic_correlator = SemanticCorrelator()

            # Sistema feedback per continuous learning
            self.feedback_loop = FeedbackLoop()

            log.info("EnsemblePredictor inizializzato con successo")

        except Exception as e:
            log.error("Errore nell'inizializzazione EnsemblePredictor", error=str(e))
            raise

    async def predict(self, text: str) -> Tuple[List[Dict[str, Any]], bool, float]:
        """
        Esegue predizione completa con pipeline a 3 stadi + correlazione semantica.

        Args:
            text: Testo da analizzare

        Returns:
            Tuple[entities, requires_review, uncertainty]

        Pipeline:
        1. Three-stage NER extraction
        2. Semantic correlation of distant references
        3. Quality assessment e uncertainty calculation
        """
        log.info("Starting enhanced prediction", text_length=len(text))

        try:
            # Stage 1-3: Estrazione NER con pipeline specializzata
            stage_entities, stage_requires_review, stage_uncertainty = await self.three_stage_predictor.predict(text)
            log.info("Three-stage prediction complete",
                    entities_found=len(stage_entities),
                    stage_uncertainty=stage_uncertainty)

            # Stage 4: Correlazione semantica per riferimenti distanti
            enhanced_entities, correlations = await self.semantic_correlator.correlate_legal_references(
                text, stage_entities
            )
            log.info("Semantic correlation complete",
                    enhanced_entities=len(enhanced_entities),
                    correlations_found=len(correlations))

            # Stage 5: Quality assessment finale
            final_entities, final_uncertainty, requires_review = await self._final_quality_assessment(
                enhanced_entities, correlations, stage_uncertainty
            )

            log.info("Enhanced prediction complete",
                    final_entities=len(final_entities),
                    final_uncertainty=final_uncertainty,
                    requires_review=requires_review)

            return final_entities, requires_review, final_uncertainty

        except Exception as e:
            log.error("Errore durante enhanced prediction", error=str(e))
            # Fallback sicuro
            return [], True, 1.0

    async def _final_quality_assessment(
        self,
        entities: List[Dict[str, Any]],
        correlations: List[Any],
        base_uncertainty: float
    ) -> Tuple[List[Dict[str, Any]], float, bool]:
        """
        Assessment finale della qualità e calcolo uncertainty.
        """
        if not entities:
            return [], 1.0, True

        # Calcola quality score per ogni entità
        quality_enhanced_entities = []
        total_quality_score = 0.0

        for entity in entities:
            # Base quality dalla confidence
            quality_score = entity.get("confidence", 0.5)

            # Bonus per entità strutturate
            if entity.get("structured_data"):
                quality_score += 0.1

            # Bonus per entità correlate
            if entity.get("has_correlations"):
                correlation_count = len(entity.get("semantic_correlations", []))
                quality_score += min(0.15, correlation_count * 0.05)

            # Bonus per validation score alta
            validation_score = entity.get("validation_score", 0.5)
            if validation_score > 0.8:
                quality_score += 0.05

            # Assicura range [0.1, 0.99]
            quality_score = max(0.1, min(0.99, quality_score))

            # Aggiorna entità con quality score finale
            entity["final_quality_score"] = quality_score
            quality_enhanced_entities.append(entity)
            total_quality_score += quality_score

        # Calcola uncertainty finale
        avg_quality = total_quality_score / len(entities) if entities else 0.0
        final_uncertainty = 1.0 - avg_quality

        # Factor in correlations per uncertainty
        correlation_factor = min(0.2, len(correlations) * 0.02)
        final_uncertainty = max(0.1, final_uncertainty - correlation_factor)

        # Combina con base uncertainty dal three-stage
        combined_uncertainty = (final_uncertainty * 0.6) + (base_uncertainty * 0.4)

        # Determina se serve review
        requires_review = (
            combined_uncertainty > 0.6 or  # Alta incertezza
            avg_quality < 0.7 or           # Bassa qualità media
            len(entities) == 0             # Nessuna entità trovata
        )

        log.debug("Final quality assessment",
                 avg_quality=avg_quality,
                 final_uncertainty=combined_uncertainty,
                 requires_review=requires_review,
                 correlations_count=len(correlations))

        return quality_enhanced_entities, combined_uncertainty, requires_review

    async def process_feedback(
        self,
        document_id: str,
        user_id: str,
        feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processa feedback dell'utente per continuous learning.

        Args:
            document_id: ID del documento analizzato
            user_id: ID dell'utente che fornisce feedback
            feedback_data: Dati del feedback

        Returns:
            Risultato del processing feedback
        """
        log.info("Processing user feedback",
                document_id=document_id,
                user_id=user_id)

        try:
            result = await self.feedback_loop.process_feedback(
                document_id=document_id,
                user_id=user_id,
                feedback_data=feedback_data
            )

            log.info("Feedback processed successfully", result=result)
            return result

        except Exception as e:
            log.error("Error processing feedback", error=str(e))
            return {"status": "error", "error": str(e)}

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Ottiene statistiche complete del sistema per monitoring.
        """
        try:
            # Stats dal feedback loop
            feedback_stats = await self.feedback_loop.get_feedback_statistics()

            # Stats dal correlatore semantico
            # (Implementato se necessario)

            # Combina tutto
            system_stats = {
                "predictor_type": "three_stage_with_correlation",
                "feedback_stats": feedback_stats,
                "golden_dataset_size": feedback_stats.get("golden_dataset_size", 0),
                "system_accuracy": feedback_stats.get("accuracy_rate", 0.0),
                "status": "operational"
            }

            return system_stats

        except Exception as e:
            log.error("Error getting system stats", error=str(e))
            return {"status": "error", "error": str(e)}

    async def export_golden_dataset(self, format: str = "json") -> str:
        """
        Esporta il golden dataset per analisi o training.
        """
        try:
            return await self.feedback_loop.export_golden_dataset(format=format)
        except Exception as e:
            log.error("Error exporting golden dataset", error=str(e))
            raise

    async def get_training_data(self, min_quality: float = 0.8) -> List[Dict[str, Any]]:
        """
        Ottiene dati per training/retraining dei modelli.
        """
        try:
            return await self.feedback_loop.get_golden_dataset_for_training(
                min_quality_score=min_quality
            )
        except Exception as e:
            log.error("Error getting training data", error=str(e))
            raise