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

# Instantiate the manager to be imported as a singleton
model_manager = ModelManager()
