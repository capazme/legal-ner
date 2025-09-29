"""
Feedback Loop and Golden Dataset Management
===========================================

Sistema per gestire feedback utenti e creare/aggiornare golden dataset
per continuous learning del sistema NER legale.

Features:
1. Feedback collection tramite API
2. Golden dataset creation e versioning
3. Quality metrics tracking
4. Automatic model retraining triggers
"""

from typing import List, Dict, Any, Optional, Tuple
import structlog
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path

log = structlog.get_logger()

class FeedbackType(Enum):
    """Tipi di feedback possibili."""
    CORRECT = "correct"              # Entità corretta
    INCORRECT = "incorrect"          # Entità sbagliata
    MISSING = "missing"              # Entità mancante
    WRONG_LABEL = "wrong_label"      # Label sbagliata
    WRONG_BOUNDARY = "wrong_boundary"  # Confini sbagliati
    PARTIAL = "partial"              # Entità parzialmente corretta

@dataclass
class FeedbackEntry:
    """Singolo feedback entry."""
    id: str
    document_id: str
    user_id: str
    timestamp: datetime
    feedback_type: FeedbackType
    original_entity: Optional[Dict[str, Any]]
    corrected_entity: Optional[Dict[str, Any]]
    confidence_score: float
    notes: Optional[str] = None
    validated: bool = False

@dataclass
class GoldenDatasetEntry:
    """Entry nel golden dataset."""
    id: str
    text: str
    entities: List[Dict[str, Any]]
    source: str  # "feedback", "manual", "automatic"
    quality_score: float
    created_at: datetime
    validated_by: Optional[str] = None
    feedback_count: int = 0

class FeedbackLoop:
    """
    Sistema di feedback loop per continuous learning.

    Workflow:
    1. Raccolta feedback tramite API
    2. Validazione e processing feedback
    3. Aggiornamento golden dataset
    4. Quality metrics calculation
    5. Trigger retraining se necessario
    """

    def __init__(self, golden_dataset_path: str = "data/golden_dataset.jsonl"):
        log.info("Inizializzazione FeedbackLoop", dataset_path=golden_dataset_path)
        self.golden_dataset_path = Path(golden_dataset_path)
        self.golden_dataset_path.parent.mkdir(parents=True, exist_ok=True)

        self.feedback_history: List[FeedbackEntry] = []
        self.golden_dataset: List[GoldenDatasetEntry] = []

        # Threshold per quality control
        self.min_feedback_for_golden = 2  # Minimo feedback per aggiungere al golden dataset
        self.quality_threshold = 0.8      # Soglia qualità per golden dataset
        self.retraining_threshold = 50    # Nuove entry per trigger retraining

        self._load_existing_data()

    def _load_existing_data(self):
        """Carica dati esistenti dal filesystem."""
        try:
            if self.golden_dataset_path.exists():
                with open(self.golden_dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        entry = GoldenDatasetEntry(
                            id=data['id'],
                            text=data['text'],
                            entities=data['entities'],
                            source=data['source'],
                            quality_score=data['quality_score'],
                            created_at=datetime.fromisoformat(data['created_at']),
                            validated_by=data.get('validated_by'),
                            feedback_count=data.get('feedback_count', 0)
                        )
                        self.golden_dataset.append(entry)

                log.info("Golden dataset loaded", entries=len(self.golden_dataset))
        except Exception as e:
            log.warning("Could not load existing golden dataset", error=str(e))

    async def process_feedback(
        self,
        document_id: str,
        user_id: str,
        feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processa feedback dell'utente e aggiorna il sistema.

        Args:
            document_id: ID del documento
            user_id: ID dell'utente che fornisce feedback
            feedback_data: Dati del feedback

        Returns:
            Risultato del processing
        """
        log.info("Processing user feedback",
                document_id=document_id,
                user_id=user_id,
                feedback_type=feedback_data.get('type'))

        try:
            # Crea feedback entry
            feedback_entry = FeedbackEntry(
                id=str(uuid.uuid4()),
                document_id=document_id,
                user_id=user_id,
                timestamp=datetime.now(),
                feedback_type=FeedbackType(feedback_data['type']),
                original_entity=feedback_data.get('original_entity'),
                corrected_entity=feedback_data.get('corrected_entity'),
                confidence_score=feedback_data.get('confidence_score', 0.0),
                notes=feedback_data.get('notes')
            )

            # Aggiungi alla history
            self.feedback_history.append(feedback_entry)

            # Process feedback per golden dataset
            await self._process_feedback_for_golden_dataset(feedback_entry)

            # Update quality metrics
            quality_impact = await self._calculate_quality_impact(feedback_entry)

            # Check se trigger retraining
            should_retrain = await self._check_retraining_trigger()

            result = {
                "feedback_id": feedback_entry.id,
                "status": "processed",
                "quality_impact": quality_impact,
                "should_retrain": should_retrain,
                "golden_dataset_size": len(self.golden_dataset)
            }

            log.info("Feedback processed successfully", result=result)
            return result

        except Exception as e:
            log.error("Error processing feedback", error=str(e))
            return {"status": "error", "error": str(e)}

    async def _process_feedback_for_golden_dataset(self, feedback: FeedbackEntry):
        """
        Processa feedback per aggiornare il golden dataset.
        """
        # TODO: Implement robust document matching.
        # This currently uses a simplified and incorrect matching of document_id against entry.text.
        # A robust implementation should:
        # 1. Fetch the document text from the database using feedback.document_id.
        # 2. Find the corresponding entry in self.golden_dataset by matching the document text.
        # 3. If no entry exists, one should be created with the full document text.
        existing_entry = None
        for entry in self.golden_dataset:
            if entry.text == feedback.document_id:  # Simplified - dovrebbe essere text matching
                existing_entry = entry
                break

        if feedback.feedback_type == FeedbackType.CORRECT:
            # Feedback positivo - aumenta quality score
            if existing_entry:
                existing_entry.quality_score = min(1.0, existing_entry.quality_score + 0.1)
                existing_entry.feedback_count += 1

        elif feedback.feedback_type in [FeedbackType.INCORRECT, FeedbackType.WRONG_LABEL]:
            # Feedback negativo - aggiorna o rimuovi
            if feedback.corrected_entity:
                # C'è una correzione - aggiorna il golden dataset
                await self._update_golden_dataset_with_correction(feedback)
            elif existing_entry:
                # Rimuovi entità incorretta
                existing_entry.quality_score = max(0.0, existing_entry.quality_score - 0.2)

        elif feedback.feedback_type == FeedbackType.MISSING:
            # Entità mancante - aggiungi al golden dataset
            if feedback.corrected_entity:
                await self._add_missing_entity_to_golden_dataset(feedback)

        # Salva aggiornamenti
        await self._save_golden_dataset()

    async def _update_golden_dataset_with_correction(self, feedback: FeedbackEntry):
        """
        Aggiorna golden dataset con correzione dal feedback.
        """
        corrected_entity = feedback.corrected_entity

        # TODO: Use the actual document text instead of a placeholder.
        # The `text` field is currently populated with a placeholder string.
        # It should be populated with the full text of the document associated with `feedback.document_id`.
        # This requires fetching the document from the database.
        new_entry = GoldenDatasetEntry(
            id=str(uuid.uuid4()),
            text=f"document_{feedback.document_id}",  # Simplified
            entities=[corrected_entity],
            source="feedback",
            quality_score=0.8,  # Initial score per feedback corrections
            created_at=datetime.now(),
            validated_by=feedback.user_id,
            feedback_count=1
        )

        self.golden_dataset.append(new_entry)
        log.debug("Added corrected entity to golden dataset", entity=corrected_entity)

    async def _add_missing_entity_to_golden_dataset(self, feedback: FeedbackEntry):
        """
        Aggiunge entità mancante al golden dataset.
        """
        missing_entity = feedback.corrected_entity

        # TODO: Use the actual document text instead of a placeholder.
        # The `text` field is currently populated with a placeholder string.
        # It should be populated with the full text of the document associated with `feedback.document_id`.
        # This requires fetching the document from the database.
        new_entry = GoldenDatasetEntry(
            id=str(uuid.uuid4()),
            text=f"document_{feedback.document_id}",
            entities=[missing_entity],
            source="feedback_missing",
            quality_score=0.9,  # High score per missing entities (important catches)
            created_at=datetime.now(),
            validated_by=feedback.user_id,
            feedback_count=1
        )

        self.golden_dataset.append(new_entry)
        log.debug("Added missing entity to golden dataset", entity=missing_entity)

    async def _calculate_quality_impact(self, feedback: FeedbackEntry) -> Dict[str, float]:
        """
        Calcola l'impatto del feedback sulla qualità del sistema.
        """
        # TODO: Implement a more sophisticated quality impact calculation.
        # The current implementation uses fixed placeholder values.
        # A better approach would be to:
        # 1. Maintain running metrics for precision, recall, and F1-score.
        # 2. Update these metrics based on the type of feedback (TP, FP, FN).
        #    - CORRECT = True Positive (if already detected)
        #    - INCORRECT = False Positive
        #    - MISSING = False Negative
        #    - WRONG_LABEL = FP + FN
        # 3. The impact should reflect the change in these metrics.
        impact = {
            "accuracy_impact": 0.0,
            "precision_impact": 0.0,
            "recall_impact": 0.0
        }

        if feedback.feedback_type == FeedbackType.CORRECT:
            impact["accuracy_impact"] = 0.02
            impact["precision_impact"] = 0.01
        elif feedback.feedback_type == FeedbackType.INCORRECT:
            impact["accuracy_impact"] = -0.05
            impact["precision_impact"] = -0.03
        elif feedback.feedback_type == FeedbackType.MISSING:
            impact["recall_impact"] = -0.04

        return impact

    async def _check_retraining_trigger(self) -> bool:
        """
        Verifica se è necessario triggerare il retraining.
        """
        recent_feedback = [
            f for f in self.feedback_history
            if (datetime.now() - f.timestamp).days <= 7
        ]

        negative_feedback_count = sum(
            1 for f in recent_feedback
            if f.feedback_type in [FeedbackType.INCORRECT, FeedbackType.MISSING]
        )

        return negative_feedback_count >= self.retraining_threshold

    async def _save_golden_dataset(self):
        """
        Salva il golden dataset su filesystem.
        """
        try:
            with open(self.golden_dataset_path, 'w', encoding='utf-8') as f:
                for entry in self.golden_dataset:
                    # Converti a dict per serializzazione JSON
                    entry_dict = asdict(entry)
                    entry_dict['created_at'] = entry.created_at.isoformat()
                    f.write(json.dumps(entry_dict, ensure_ascii=False) + '\n')

            log.debug("Golden dataset saved", entries=len(self.golden_dataset))
        except Exception as e:
            log.error("Error saving golden dataset", error=str(e))

    async def get_golden_dataset_for_training(
        self,
        min_quality_score: float = 0.8,
        max_entries: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Ottiene il golden dataset filtrato per training.

        Args:
            min_quality_score: Soglia minima qualità
            max_entries: Numero massimo entries da ritornare

        Returns:
            Lista di entries per training
        """
        # Filtra per qualità
        quality_entries = [
            entry for entry in self.golden_dataset
            if entry.quality_score >= min_quality_score
        ]

        # Ordina per quality score (migliori prima)
        quality_entries.sort(key=lambda x: x.quality_score, reverse=True)

        # Limita numero se richiesto
        if max_entries:
            quality_entries = quality_entries[:max_entries]

        # Converti a formato training
        training_data = []
        for entry in quality_entries:
            training_data.append({
                "text": entry.text,
                "entities": entry.entities,
                "quality_score": entry.quality_score,
                "source": entry.source
            })

        log.info("Golden dataset prepared for training",
                total_entries=len(self.golden_dataset),
                quality_filtered=len(training_data),
                avg_quality=sum(e["quality_score"] for e in training_data) / len(training_data) if training_data else 0)

        return training_data

    async def get_feedback_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Ottiene statistiche sui feedback recenti.
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_feedback = [
            f for f in self.feedback_history
            if f.timestamp >= cutoff_date
        ]

        if not recent_feedback:
            return {"total_feedback": 0}

        feedback_by_type = {}
        for feedback in recent_feedback:
            fb_type = feedback.feedback_type.value
            feedback_by_type[fb_type] = feedback_by_type.get(fb_type, 0) + 1

        accuracy_feedback = [
            f for f in recent_feedback
            if f.feedback_type in [FeedbackType.CORRECT, FeedbackType.INCORRECT]
        ]

        accuracy_rate = (
            sum(1 for f in accuracy_feedback if f.feedback_type == FeedbackType.CORRECT) /
            len(accuracy_feedback) if accuracy_feedback else 0
        )

        return {
            "total_feedback": len(recent_feedback),
            "feedback_by_type": feedback_by_type,
            "accuracy_rate": accuracy_rate,
            "golden_dataset_size": len(self.golden_dataset),
            "avg_golden_quality": sum(e.quality_score for e in self.golden_dataset) / len(self.golden_dataset) if self.golden_dataset else 0,
            "days_analyzed": days
        }

    async def export_golden_dataset(self, format: str = "json") -> str:
        """
        Esporta il golden dataset in formato specificato.
        """
        if format == "json":
            export_data = []
            for entry in self.golden_dataset:
                export_data.append({
                    "id": entry.id,
                    "text": entry.text,
                    "entities": entry.entities,
                    "quality_score": entry.quality_score,
                    "source": entry.source,
                    "created_at": entry.created_at.isoformat()
                })
            return json.dumps(export_data, indent=2, ensure_ascii=False)

        elif format == "conll":
            # TODO: Implement a proper CoNLL-2003 format export.
            # The current implementation is a simplified placeholder.
            # A correct implementation should:
            # 1. Tokenize the document text.
            # 2. For each token, determine its IOB2 tag (B-LABEL, I-LABEL, O).
            # 3. Format the output as "token IOB-tag" on each line.
            # 4. Separate sentences with an empty line.
            # 5. Separate documents with a "-DOCSTART-" line or similar.
            conll_lines = []
            for entry in self.golden_dataset:
                # Simplified CoNLL export
                conll_lines.append(f"# {entry.id}")
                conll_lines.append(f"# {entry.text}")
                for entity in entry.entities:
                    conll_lines.append(f"{entity['text']}\t{entity['label']}")
                conll_lines.append("")  # Empty line tra documenti

            return "\n".join(conll_lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def validate_golden_dataset_entry(
        self,
        entry_id: str,
        validator_user_id: str,
        is_valid: bool,
        notes: Optional[str] = None
    ) -> bool:
        """
        Valida una entry del golden dataset.
        """
        for entry in self.golden_dataset:
            if entry.id == entry_id:
                if is_valid:
                    entry.quality_score = min(1.0, entry.quality_score + 0.1)
                    entry.validated_by = validator_user_id
                else:
                    entry.quality_score = max(0.0, entry.quality_score - 0.3)

                await self._save_golden_dataset()

                log.info("Golden dataset entry validated",
                        entry_id=entry_id,
                        is_valid=is_valid,
                        new_quality_score=entry.quality_score)
                return True

        return False