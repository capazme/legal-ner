import structlog
from sqlalchemy.orm import Session
from typing import Optional

from app.database import models
from app.services.specialized_pipeline import LegalSourceExtractionPipeline

log = structlog.get_logger()

class ModelManager:
    """
    A singleton class to manage the lifecycle of the ML pipeline.
    It ensures that the pipeline is loaded once and can be hot-swapped.
    """
    _pipeline: Optional[LegalSourceExtractionPipeline] = None
    _active_model_version: Optional[str] = None

    @classmethod
    def load_active_model(cls, db: Session):
        """
        Loads the model marked as 'is_active' from the database and initializes the pipeline.
        Should be called at application startup.
        """
        log.info("Attempting to load the active model for the pipeline.")
        active_model = db.query(models.TrainedModel).filter(models.TrainedModel.is_active == True).first()

        if active_model:
            log.info("Found active model. Initializing pipeline...", version=active_model.version, path=active_model.path)
            try:
                cls._pipeline = LegalSourceExtractionPipeline(fine_tuned_model_path=active_model.path)
                cls._active_model_version = active_model.version
                log.info("Pipeline initialized successfully with active model.", version=cls._active_model_version)
            except Exception as e:
                log.error("Failed to initialize pipeline with active model. Falling back to default.", error=str(e), path=active_model.path)
                cls._pipeline = LegalSourceExtractionPipeline() # Fallback to default
                cls._active_model_version = "default"
        else:
            log.warning("No active model found in the database. Initializing with default pipeline.")
            cls._pipeline = LegalSourceExtractionPipeline()
            cls._active_model_version = "default"

    @classmethod
    def get_pipeline(cls) -> LegalSourceExtractionPipeline:
        """
        Returns the current instance of the pipeline.
        If not loaded, it will load the default one.
        """
        if cls._pipeline is None:
            log.warning("Pipeline accessed before being loaded. Loading default pipeline.")
            # This should ideally be called from a startup event
            # from app.database.database import SessionLocal
            # db = SessionLocal()
            # cls.load_active_model(db)
            # db.close()
            # For safety, we load the default if it's still None
            if cls._pipeline is None:
                cls._pipeline = LegalSourceExtractionPipeline()
                cls._active_model_version = "default"
        
        return cls._pipeline

    @classmethod
    def reload_pipeline(cls, db: Session, model_version: Optional[str] = None) -> dict:
        """
        Swaps the current pipeline with a new one based on the specified model version.
        If no version is provided, it reloads the currently active model from the DB.
        """
        log.info("Pipeline reload requested.", new_version=model_version)
        target_model = None
        if model_version:
            # Deactivate all other models
            db.query(models.TrainedModel).update({models.TrainedModel.is_active: False})
            # Activate the target model
            target_model = db.query(models.TrainedModel).filter(models.TrainedModel.version == model_version).first()
            if target_model:
                target_model.is_active = True
                db.commit()
            else:
                db.rollback()
                log.error("Model version not found for reloading.", version=model_version)
                return {"status": "error", "message": f"Model version {model_version} not found."}
        else:
            # Find the currently active model in the DB
            target_model = db.query(models.TrainedModel).filter(models.TrainedModel.is_active == True).first()

        if not target_model:
            log.error("No active model found to reload.")
            return {"status": "error", "message": "No active model found to reload."}

        try:
            log.info("Loading new pipeline...", version=target_model.version, path=target_model.path)
            cls._pipeline = LegalSourceExtractionPipeline(fine_tuned_model_path=target_model.path)
            cls._active_model_version = target_model.version
            log.info("Pipeline reloaded successfully.", new_version=cls._active_model_version)
            return {"status": "success", "active_version": cls._active_model_version}
        except Exception as e:
            log.error("Failed to reload pipeline with new model. Reverting to default.", error=str(e))
            cls._pipeline = LegalSourceExtractionPipeline()
            cls._active_model_version = "default"
            return {"status": "error", "message": str(e), "active_version": "default"}

    @classmethod
    def get_active_model_version(cls) -> Optional[str]:
        """
        Returns the version of the currently active model.
        """
        return cls._active_model_version

    @classmethod
    def list_available_models(cls, db: Session) -> list[dict]:
        """
        Lista tutti i modelli disponibili nel database con le loro metriche.

        Returns:
            Lista di dizionari con informazioni sui modelli
        """
        models_list = db.query(models.TrainedModel).order_by(
            models.TrainedModel.created_at.desc()
        ).all()

        result = []
        for model in models_list:
            result.append({
                "id": model.id,
                "version": model.version,
                "model_name": model.model_name,
                "path": model.path,
                "description": model.description,
                "accuracy": model.accuracy,
                "f1_score": model.f1_score,
                "precision": model.precision,
                "recall": model.recall,
                "created_at": model.created_at.isoformat() if model.created_at else None,
                "is_active": model.is_active
            })

        return result

    @classmethod
    def activate_model(cls, db: Session, version: str) -> dict:
        """
        Attiva un modello specifico e ricarica la pipeline.

        Args:
            db: Database session
            version: Versione del modello da attivare

        Returns:
            Risultato dell'operazione
        """
        log.info("Activating model", version=version)

        # Trova il modello
        target_model = db.query(models.TrainedModel).filter(
            models.TrainedModel.version == version
        ).first()

        if not target_model:
            log.error("Model not found", version=version)
            return {"status": "error", "message": f"Model version {version} not found"}

        # Disattiva tutti gli altri modelli
        db.query(models.TrainedModel).update({models.TrainedModel.is_active: False})

        # Attiva il modello target
        target_model.is_active = True
        db.commit()

        # Ricarica la pipeline
        return cls.reload_pipeline(db, version)

    @classmethod
    def deactivate_all_models(cls, db: Session) -> dict:
        """
        Disattiva tutti i modelli e torna alla pipeline di default (rule-based).

        Args:
            db: Database session

        Returns:
            Risultato dell'operazione
        """
        log.info("Deactivating all models, reverting to rule-based pipeline")

        # Disattiva tutti i modelli
        db.query(models.TrainedModel).update({models.TrainedModel.is_active: False})
        db.commit()

        # Ricarica pipeline di default
        cls._pipeline = LegalSourceExtractionPipeline()
        cls._active_model_version = "default"

        log.info("Pipeline reverted to default rule-based")
        return {"status": "success", "active_version": "default", "message": "Reverted to rule-based pipeline"}

    @classmethod
    def compare_models(cls, db: Session, version1: str, version2: str) -> dict:
        """
        Confronta due modelli basandosi sulle metriche salvate.

        Args:
            db: Database session
            version1: Prima versione da confrontare
            version2: Seconda versione da confrontare

        Returns:
            Confronto delle metriche
        """
        model1 = db.query(models.TrainedModel).filter(
            models.TrainedModel.version == version1
        ).first()

        model2 = db.query(models.TrainedModel).filter(
            models.TrainedModel.version == version2
        ).first()

        if not model1 or not model2:
            return {"status": "error", "message": "One or both models not found"}

        comparison = {
            "model1": {
                "version": model1.version,
                "f1_score": model1.f1_score,
                "precision": model1.precision,
                "recall": model1.recall,
                "accuracy": model1.accuracy,
                "created_at": model1.created_at.isoformat() if model1.created_at else None
            },
            "model2": {
                "version": model2.version,
                "f1_score": model2.f1_score,
                "precision": model2.precision,
                "recall": model2.recall,
                "accuracy": model2.accuracy,
                "created_at": model2.created_at.isoformat() if model2.created_at else None
            },
            "winner": None,
            "differences": {}
        }

        # Determina il vincitore basandosi sull'F1 score
        if model1.f1_score and model2.f1_score:
            if model1.f1_score > model2.f1_score:
                comparison["winner"] = model1.version
            elif model2.f1_score > model1.f1_score:
                comparison["winner"] = model2.version
            else:
                comparison["winner"] = "tie"

            comparison["differences"] = {
                "f1_score_diff": model1.f1_score - model2.f1_score,
                "precision_diff": (model1.precision or 0.0) - (model2.precision or 0.0),
                "recall_diff": (model1.recall or 0.0) - (model2.recall or 0.0),
                "accuracy_diff": (model1.accuracy or 0.0) - (model2.accuracy or 0.0)
            }

        return comparison

    @classmethod
    def auto_select_best_model(cls, db: Session, metric: str = "f1_score") -> dict:
        """
        Seleziona automaticamente il modello migliore basandosi su una metrica.

        Args:
            db: Database session
            metric: Metrica da usare per la selezione (f1_score, precision, recall, accuracy)

        Returns:
            Risultato dell'operazione
        """
        log.info("Auto-selecting best model", metric=metric)

        # Mapping delle metriche ai campi del modello
        metric_field_map = {
            "f1_score": models.TrainedModel.f1_score,
            "precision": models.TrainedModel.precision,
            "recall": models.TrainedModel.recall,
            "accuracy": models.TrainedModel.accuracy
        }

        if metric not in metric_field_map:
            return {"status": "error", "message": f"Invalid metric: {metric}"}

        # Trova il modello con il valore massimo della metrica
        best_model = (
            db.query(models.TrainedModel)
            .filter(metric_field_map[metric].isnot(None))
            .order_by(metric_field_map[metric].desc())
            .first()
        )

        if not best_model:
            log.warning("No models found in database")
            return {"status": "error", "message": "No trained models found"}

        log.info(
            "Best model selected",
            version=best_model.version,
            metric=metric,
            value=getattr(best_model, metric)
        )

        # Attiva il modello migliore
        return cls.activate_model(db, best_model.version)

# Instantiate the manager to be imported as a singleton
model_manager = ModelManager()
