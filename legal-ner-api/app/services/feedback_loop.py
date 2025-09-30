"""
Feedback Loop and Golden Dataset Management (DATABASE-BACKED)
==============================================================

Sistema per gestire feedback utenti e creare/aggiornare golden dataset
usando il database PostgreSQL invece di file JSONL.

Features:
1. Feedback collection tramite tabelle Annotation
2. Golden dataset built from validated annotations
3. Quality metrics tracking basati su dati reali
4. Automatic model retraining triggers
"""

from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func
import structlog
from datetime import datetime, timedelta
from app.database import models

log = structlog.get_logger()


class FeedbackLoop:
    """
    Sistema di feedback loop per continuous learning basato su database.

    Workflow:
    1. Le annotazioni sono salvate nella tabella `annotations` dal UI
    2. Il golden dataset è costruito on-demand dalle annotazioni validate
    3. Quality metrics calcolate da dati reali del database
    4. Trigger retraining basato su soglie configurabili
    """

    def __init__(self):
        log.info("Inizializzazione FeedbackLoop (Database-backed)")

        # Threshold per quality control
        self.min_feedback_for_golden = 1  # Almeno 1 annotazione per includere nel golden dataset
        self.quality_threshold = 0.8      # Soglia qualità per golden dataset
        self.retraining_threshold = 50    # Nuove annotazioni per trigger retraining

    def get_golden_dataset(self, db: Session, min_feedback_count: int = 1) -> List[Dict[str, Any]]:
        """
        Costruisce il golden dataset dalle annotazioni validate nel database.

        Args:
            db: Database session
            min_feedback_count: Minimo numero di feedback per documento

        Returns:
            Lista di entries per il golden dataset con testo e entità
        """
        log.info("Building golden dataset from database annotations")

        # Query per ottenere documenti con annotazioni
        # Un documento è considerato "golden" se ha almeno min_feedback_count annotazioni
        subquery = (
            db.query(
                models.Annotation.entity_id,
                func.count(models.Annotation.id).label('feedback_count')
            )
            .group_by(models.Annotation.entity_id)
            .having(func.count(models.Annotation.id) >= min_feedback_count)
            .subquery()
        )

        # Ottieni entità con feedback sufficiente
        entities_with_feedback = (
            db.query(models.Entity, subquery.c.feedback_count)
            .join(subquery, models.Entity.id == subquery.c.entity_id)
            .all()
        )

        # Raggruppa per documento
        documents_dict = {}
        for entity, feedback_count in entities_with_feedback:
            doc_id = entity.document_id

            if doc_id not in documents_dict:
                # Carica il documento
                document = db.query(models.Document).filter(models.Document.id == doc_id).first()
                if not document:
                    continue

                documents_dict[doc_id] = {
                    'id': str(doc_id),
                    'text': document.text,
                    'entities': [],
                    'source': 'database',
                    'quality_score': 0.0,
                    'created_at': document.created_at.isoformat(),
                    'feedback_count': 0
                }

            # Aggiungi entità al documento
            # Ottieni le annotazioni per questa entità per decidere se è corretta
            annotations = db.query(models.Annotation).filter(
                models.Annotation.entity_id == entity.id
            ).all()

            # Calcola se l'entità è considerata corretta dalla maggioranza
            correct_count = sum(1 for a in annotations if a.is_correct)
            incorrect_count = len(annotations) - correct_count

            # Include solo entità validate come corrette
            if correct_count > incorrect_count:
                entity_data = {
                    'text': entity.text,
                    'label': entity.label,
                    'start_char': entity.start_char,
                    'end_char': entity.end_char,
                    'confidence': entity.confidence,
                    'model': entity.model,
                    'feedback_count': len(annotations),
                    'validation_score': correct_count / len(annotations) if annotations else 0.0
                }
                documents_dict[doc_id]['entities'].append(entity_data)
                documents_dict[doc_id]['feedback_count'] += len(annotations)

        # Converti in lista e calcola quality scores
        golden_dataset = []
        for doc_id, doc_data in documents_dict.items():
            if not doc_data['entities']:  # Skip documenti senza entità validate
                continue

            # Calcola quality score basato su validation scores delle entità
            validation_scores = [e['validation_score'] for e in doc_data['entities']]
            doc_data['quality_score'] = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0

            golden_dataset.append(doc_data)

        log.info(
            "Golden dataset built from database",
            total_documents=len(golden_dataset),
            total_entities=sum(len(d['entities']) for d in golden_dataset),
            avg_quality=sum(d['quality_score'] for d in golden_dataset) / len(golden_dataset) if golden_dataset else 0.0
        )

        return golden_dataset

    def get_golden_dataset_for_training(
        self,
        db: Session,
        min_quality_score: float = 0.8,
        max_entries: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Ottiene il golden dataset filtrato per training.

        Args:
            db: Database session
            min_quality_score: Soglia minima qualità
            max_entries: Numero massimo entries da ritornare

        Returns:
            Lista di entries per training
        """
        # Ottieni tutto il golden dataset
        golden_dataset = self.get_golden_dataset(db)

        # Filtra per qualità
        quality_entries = [
            entry for entry in golden_dataset
            if entry['quality_score'] >= min_quality_score
        ]

        # Ordina per quality score (migliori prima)
        quality_entries.sort(key=lambda x: x['quality_score'], reverse=True)

        # Limita numero se richiesto
        if max_entries:
            quality_entries = quality_entries[:max_entries]

        # Converti a formato training (solo testo ed entità)
        training_data = []
        for entry in quality_entries:
            training_data.append({
                "text": entry['text'],
                "entities": entry['entities'],
                "quality_score": entry['quality_score'],
                "source": entry['source']
            })

        log.info(
            "Golden dataset prepared for training",
            total_entries=len(golden_dataset),
            quality_filtered=len(training_data),
            avg_quality=sum(e["quality_score"] for e in training_data) / len(training_data) if training_data else 0
        )

        return training_data

    def get_feedback_statistics(self, db: Session, days: int = 30) -> Dict[str, Any]:
        """
        Ottiene statistiche sui feedback recenti dal database.

        Args:
            db: Database session
            days: Numero di giorni da analizzare

        Returns:
            Dizionario con statistiche
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # Conta feedback totali
        total_feedback = db.query(models.Annotation).filter(
            models.Annotation.created_at >= cutoff_date
        ).count()

        if total_feedback == 0:
            return {"total_feedback": 0, "days_analyzed": days}

        # Conta feedback per tipo (correct/incorrect)
        correct_feedback = db.query(models.Annotation).filter(
            models.Annotation.created_at >= cutoff_date,
            models.Annotation.is_correct == True
        ).count()

        incorrect_feedback = total_feedback - correct_feedback

        # Calcola accuracy rate
        accuracy_rate = correct_feedback / total_feedback if total_feedback > 0 else 0.0

        # Conta dimensione golden dataset
        golden_dataset_size = len(self.get_golden_dataset(db))

        # Calcola qualità media del golden dataset
        golden_dataset = self.get_golden_dataset(db)
        avg_golden_quality = (
            sum(e['quality_score'] for e in golden_dataset) / len(golden_dataset)
            if golden_dataset else 0.0
        )

        # Feedback per utente
        user_feedback_counts = {}
        annotations = db.query(models.Annotation).filter(
            models.Annotation.created_at >= cutoff_date
        ).all()

        for annotation in annotations:
            user_id = annotation.user_id
            if user_id not in user_feedback_counts:
                user_feedback_counts[user_id] = {'correct': 0, 'incorrect': 0}

            if annotation.is_correct:
                user_feedback_counts[user_id]['correct'] += 1
            else:
                user_feedback_counts[user_id]['incorrect'] += 1

        return {
            "total_feedback": total_feedback,
            "feedback_by_type": {
                "correct": correct_feedback,
                "incorrect": incorrect_feedback
            },
            "accuracy_rate": accuracy_rate,
            "golden_dataset_size": golden_dataset_size,
            "avg_golden_quality": avg_golden_quality,
            "days_analyzed": days,
            "user_feedback_counts": user_feedback_counts,
            "unique_annotators": len(user_feedback_counts)
        }

    def calculate_quality_metrics(self, db: Session) -> Dict[str, Any]:
        """
        Calcola metriche di qualità reali basate su dati del database.

        Returns:
            Precision, Recall, F1 e altre metriche
        """
        # Ottieni tutte le entità con annotazioni
        entities_with_annotations = (
            db.query(models.Entity)
            .join(models.Annotation, models.Entity.id == models.Annotation.entity_id)
            .all()
        )

        if not entities_with_annotations:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "total_entities": 0,
                "annotated_entities": 0
            }

        # Calcola True Positives, False Positives, False Negatives
        true_positives = 0
        false_positives = 0

        for entity in entities_with_annotations:
            annotations = db.query(models.Annotation).filter(
                models.Annotation.entity_id == entity.id
            ).all()

            # Usa majority voting
            correct_count = sum(1 for a in annotations if a.is_correct)
            total_count = len(annotations)

            if correct_count > total_count / 2:
                true_positives += 1
            else:
                false_positives += 1

        # Per False Negatives, cerchiamo entità mancanti
        # (annotazioni con corrected_label diverse da quella originale)
        false_negatives = db.query(models.Annotation).filter(
            models.Annotation.corrected_label.isnot(None),
            models.Annotation.is_correct == False
        ).count()

        # Calcola metriche
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        total_entities = db.query(models.Entity).count()

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_entities": total_entities,
            "annotated_entities": len(entities_with_annotations)
        }

    def check_retraining_trigger(self, db: Session) -> Dict[str, Any]:
        """
        Verifica se è necessario triggerare il retraining basato su nuove annotazioni.

        Args:
            db: Database session

        Returns:
            Dizionario con informazioni sul trigger
        """
        # Conta annotazioni recenti (ultimi 7 giorni)
        cutoff_date = datetime.now() - timedelta(days=7)
        recent_annotations = db.query(models.Annotation).filter(
            models.Annotation.created_at >= cutoff_date
        ).count()

        # Conta annotazioni negative (incorrect + has corrected_label)
        negative_feedback_count = db.query(models.Annotation).filter(
            models.Annotation.created_at >= cutoff_date,
            models.Annotation.is_correct == False
        ).count()

        # Trigger retraining se:
        # 1. Ci sono abbastanza annotazioni totali
        # 2. C'è feedback negativo significativo
        should_retrain = (
            recent_annotations >= self.retraining_threshold or
            negative_feedback_count >= self.retraining_threshold / 2
        )

        # Ottieni dimensione dataset disponibile per training
        golden_dataset_size = len(self.get_golden_dataset(db))

        return {
            "should_retrain": should_retrain,
            "recent_annotations": recent_annotations,
            "negative_feedback_count": negative_feedback_count,
            "retraining_threshold": self.retraining_threshold,
            "golden_dataset_size": golden_dataset_size,
            "reason": (
                f"Recent annotations ({recent_annotations}) >= threshold ({self.retraining_threshold})"
                if recent_annotations >= self.retraining_threshold
                else f"Negative feedback ({negative_feedback_count}) >= threshold ({self.retraining_threshold / 2})"
                if negative_feedback_count >= self.retraining_threshold / 2
                else "No retraining needed"
            )
        }
