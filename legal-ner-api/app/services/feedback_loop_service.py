import asyncio
import structlog
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.database import models
from app.core.active_learning_config import get_active_learning_config
from app.models.model_trainer import ModelTrainer
from app.core.model_manager import model_manager

log = structlog.get_logger()

class FeedbackLoopService:
    """
    Service to automate the feedback loop of training, evaluating, and deploying models.
    """

    def __init__(self, db: Session):
        self.db = db
        self.config = get_active_learning_config()
        self.model_trainer = ModelTrainer()
        self.is_running = False

    async def run_single_check(self):
        """
        Performs a single check to see if conditions are met for a new training run.
        """
        log.info("Feedback loop: Running check...")

        try:
            # 1. Get the timestamp of the last trained model
            last_model = self.db.query(models.TrainedModel).order_by(models.TrainedModel.created_at.desc()).first()
            last_training_time = last_model.created_at if last_model else datetime.min

            # 2. Count new validated annotations since the last training run
            # (Assuming Annotation.is_correct is not None means it's validated)
            new_annotations_count = self.db.query(models.Annotation).filter(models.Annotation.created_at > last_training_time).filter(models.Annotation.is_correct.isnot(None)).count()
            
            log.info("Feedback loop: Check results", 
                     new_annotations=new_annotations_count, 
                     threshold=self.config.feedback_loop.auto_training_threshold)

            # 3. Check if the threshold is met
            if new_annotations_count < self.config.feedback_loop.auto_training_threshold:
                log.info("Feedback loop: Threshold not met. Skipping training.")
                return

            log.info("Feedback loop: Annotation threshold met. Starting new training cycle.")

            # 4. Trigger training
            version = f"auto-trained-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            output_dir = self.config.get_model_output_dir(version)
            dataset_path = "path/to/latest/dataset.json" # This needs to be determined
            
            # Note: The dataset building logic should be invoked here.
            # For now, we assume a dataset path is available.
            # from app.feedback.dataset_builder import DatasetBuilder
            # dataset_builder = DatasetBuilder()
            # dataset_path = dataset_builder.build_dataset(self.db, version)

            self.model_trainer.train_model(
                db=self.db,
                version=version,
                dataset_path=dataset_path, # This needs a real path
                model_name=self.config.training.base_model,
                output_dir=output_dir
            )

            # 5. Compare and deploy if better
            new_model = self.db.query(models.TrainedModel).filter(models.TrainedModel.version == version).first()
            active_model = self.db.query(models.TrainedModel).filter(models.TrainedModel.is_active == True).first()

            if not new_model:
                log.error("Feedback loop: Training finished but new model not found in DB.", version=version)
                return

            primary_metric = self.config.feedback_loop.primary_evaluation_metric
            new_model_score = getattr(new_model, primary_metric, 0.0) or 0.0
            active_model_score = getattr(active_model, primary_metric, 0.0) or 0.0

            log.info("Feedback loop: Comparing models", 
                     new_model_version=new_model.version,
                     new_model_score=new_model_score,
                     active_model_version=active_model.version if active_model else 'None',
                     active_model_score=active_model_score,
                     metric=primary_metric)

            if new_model_score > active_model_score:
                log.info("Feedback loop: New model is better. Activating it.")
                model_manager.reload_pipeline(self.db, model_version=new_model.version)
            else:
                log.info("Feedback loop: New model is not better than the active one. No change.")

        except Exception as e:
            log.error("Feedback loop: Error during check.", error=str(e))

    async def start_loop(self):
        """
        Starts the continuous feedback loop in the background.
        """
        if self.is_running:
            log.warning("Feedback loop is already running.")
            return

        log.info("Starting automated feedback loop...", check_interval_seconds=self.config.feedback_loop.check_interval_seconds)
        self.is_running = True

        while self.is_running:
            await self.run_single_check()
            await asyncio.sleep(self.config.feedback_loop.check_interval_seconds)

    def stop_loop(self):
        """
        Stops the feedback loop.
        """
        log.info("Stopping automated feedback loop...")
        self.is_running = False
