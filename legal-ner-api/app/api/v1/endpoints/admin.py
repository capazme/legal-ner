"""
Admin API Endpoints

Gestisce operazioni amministrative per database, modelli, label e sistema.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from app.database.database import get_db
from app.database import models
from app.core.dependencies import get_api_key
from app.core.config import settings
from app.core.active_learning_config import get_active_learning_config, load_active_learning_config
from app.core.model_manager import ModelManager
import structlog
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import os
import shutil
from datetime import datetime
import psutil
from minio import Minio
from minio.error import S3Error
import yaml

log = structlog.get_logger()

router = APIRouter()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DatabaseStatsResponse(BaseModel):
    documents: int
    entities: int
    annotations: int
    tasks_pending: int
    tasks_completed: int
    trained_models: int
    dataset_versions: int
    db_size_mb: float

class LabelInfo(BaseModel):
    label: str
    count: int
    accuracy: Optional[float] = None

class ConfigUpdateRequest(BaseModel):
    config_section: str  # "training", "active_learning", etc.
    updates: Dict[str, Any]

class BackupRequest(BaseModel):
    include_minio: bool = True
    include_models: bool = True

class SystemHealthResponse(BaseModel):
    database: str
    minio: str
    disk_usage_percent: float
    memory_usage_percent: float
    cpu_usage_percent: float

class ReapplyNERRequest(BaseModel):
    task_ids: Optional[List[int]] = None  # None = all tasks
    model_version: Optional[str] = None  # None = use legacy pipeline
    replace_existing: bool = True  # Replace existing entities or add new ones

# ============================================================================
# DATABASE MANAGEMENT
# ============================================================================

@router.get("/database/stats", response_model=DatabaseStatsResponse)
async def get_database_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Get comprehensive database statistics."""
    try:
        log.info("Fetching database statistics")

        stats = {
            "documents": db.query(models.Document).count(),
            "entities": db.query(models.Entity).count(),
            "annotations": db.query(models.Annotation).count(),
            "tasks_pending": db.query(models.AnnotationTask).filter(
                models.AnnotationTask.status == "pending"
            ).count(),
            "tasks_completed": db.query(models.AnnotationTask).filter(
                models.AnnotationTask.status == "completed"
            ).count(),
            "trained_models": db.query(models.TrainedModel).count(),
            "dataset_versions": db.query(models.DatasetVersion).count(),
            "db_size_mb": 0.0  # Will be calculated below
        }

        # Get database size
        try:
            result = db.execute(text("SELECT pg_database_size(current_database())")).scalar()
            stats["db_size_mb"] = round(result / (1024 * 1024), 2)
        except Exception as e:
            log.warning("Could not fetch database size", error=str(e))

        log.info("Database statistics fetched", **stats)
        return DatabaseStatsResponse(**stats)

    except Exception as e:
        log.error("Error fetching database stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/database/export")
async def export_database(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Export complete database backup."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"backups/db_backup_{timestamp}"
        os.makedirs(backup_dir, exist_ok=True)

        log.info("Starting database export", backup_dir=backup_dir)

        # Export all tables to JSON
        tables_data = {}

        # Documents
        documents = db.query(models.Document).all()
        tables_data["documents"] = [
            {"id": d.id, "text": d.text, "source": d.source, "created_at": d.created_at.isoformat() if d.created_at else None}
            for d in documents
        ]

        # Entities
        entities = db.query(models.Entity).all()
        tables_data["entities"] = [
            {
                "id": e.id, "document_id": e.document_id, "text": e.text,
                "label": e.label, "start_char": e.start_char, "end_char": e.end_char,
                "confidence": e.confidence, "model": e.model
            }
            for e in entities
        ]

        # Annotations
        annotations = db.query(models.Annotation).all()
        tables_data["annotations"] = [
            {
                "id": a.id, "entity_id": a.entity_id, "is_correct": a.is_correct,
                "user_id": a.user_id, "corrected_label": a.corrected_label,
                "corrected_text": a.corrected_text, "notes": a.notes,
                "created_at": a.created_at.isoformat() if a.created_at else None
            }
            for a in annotations
        ]

        # Tasks
        tasks = db.query(models.AnnotationTask).all()
        tables_data["tasks"] = [
            {
                "id": t.id, "document_id": t.document_id, "status": t.status,
                "priority": t.priority, "created_at": t.created_at.isoformat() if t.created_at else None
            }
            for t in tasks
        ]

        # Models
        trained_models = db.query(models.TrainedModel).all()
        tables_data["trained_models"] = [
            {
                "id": m.id, "version": m.version, "model_name": m.model_name,
                "path": m.path, "accuracy": m.accuracy, "f1_score": m.f1_score,
                "precision": m.precision, "recall": m.recall, "is_active": m.is_active,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in trained_models
        ]

        # Save to file
        backup_file = os.path.join(backup_dir, "database_backup.json")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(tables_data, f, indent=2, ensure_ascii=False)

        log.info("Database exported successfully", backup_file=backup_file)

        return {
            "status": "success",
            "backup_dir": backup_dir,
            "backup_file": backup_file,
            "total_records": sum(len(v) for v in tables_data.values())
        }

    except Exception as e:
        log.error("Error exporting database", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/database/clear-annotations")
async def clear_annotations(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Clear all annotations (keep documents and entities)."""
    try:
        log.warning("Clearing all annotations from database")

        count = db.query(models.Annotation).count()
        db.query(models.Annotation).delete()
        db.commit()

        log.info("Annotations cleared", count=count)

        return {
            "status": "success",
            "message": f"Deleted {count} annotations",
            "deleted_count": count
        }

    except Exception as e:
        db.rollback()
        log.error("Error clearing annotations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/database/clear-all")
async def clear_all_data(
    confirm: bool = False,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """⚠️ DANGER: Clear ALL data from database."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to clear all data"
        )

    try:
        log.warning("⚠️ CLEARING ALL DATABASE DATA")

        # Count before deletion
        stats = {
            "annotations": db.query(models.Annotation).count(),
            "entities": db.query(models.Entity).count(),
            "tasks": db.query(models.AnnotationTask).count(),
            "documents": db.query(models.Document).count(),
            "models": db.query(models.TrainedModel).count(),
            "datasets": db.query(models.DatasetVersion).count()
        }

        # Delete in correct order (respecting foreign keys)
        db.query(models.Annotation).delete()
        db.query(models.Entity).delete()
        db.query(models.AnnotationTask).delete()
        db.query(models.Document).delete()
        db.query(models.TrainedModel).delete()
        db.query(models.DatasetVersion).delete()
        db.commit()

        log.info("All data cleared", stats=stats)

        return {
            "status": "success",
            "message": "All data cleared",
            "deleted": stats
        }

    except Exception as e:
        db.rollback()
        log.error("Error clearing all data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/database/vacuum")
async def vacuum_database(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Optimize and compact database."""
    try:
        log.info("Running database vacuum")

        # VACUUM must run outside transaction
        db.commit()  # Commit any pending transaction
        connection = db.connection()
        connection.execution_options(isolation_level="AUTOCOMMIT")
        connection.execute(text("VACUUM ANALYZE"))

        log.info("Database vacuum completed")

        return {
            "status": "success",
            "message": "Database optimized"
        }

    except Exception as e:
        log.error("Error running vacuum", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LABEL MANAGEMENT
# ============================================================================

@router.get("/labels/list", response_model=List[LabelInfo])
async def list_labels(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """List all labels with usage statistics using centralized label mapping."""
    try:
        log.info("Fetching label statistics from centralized label system")

        # Import centralized label system
        from app.core.label_mapping import get_all_labels

        # Get standardized labels from centralized system
        all_labels = get_all_labels()
        label_stats = []

        for label in all_labels:
            # Count entities with this label (exact match)
            count = db.query(models.Entity).filter(models.Entity.label == label).count()

            # Calculate accuracy (entities marked as correct)
            entities_with_label = db.query(models.Entity).filter(
                models.Entity.label == label
            ).all()

            correct = 0
            total = 0
            for entity in entities_with_label:
                annotations = db.query(models.Annotation).filter(
                    models.Annotation.entity_id == entity.id
                ).all()
                if annotations:
                    total += len(annotations)
                    correct += sum(1 for a in annotations if a.is_correct)

            accuracy = correct / total if total > 0 else None

            # Include ALL labels from centralized system (even with 0 count)
            # This matches the annotation task interface behavior
            label_stats.append(LabelInfo(
                label=label,
                count=count,
                accuracy=accuracy
            ))

        # Sort by count descending (labels with 0 count will be at the end)
        label_stats.sort(key=lambda x: x.count, reverse=True)

        log.info("Label statistics fetched from centralized system", total_labels=len(label_stats))

        return label_stats

    except Exception as e:
        log.error("Error fetching label statistics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: Label adding is handled by /api/v1/labels endpoint (labels.py)
# This ensures consistency with the annotation UI


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

@router.get("/models/list")
async def list_models_admin(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """List all trained models with detailed metrics."""
    try:
        models_list = ModelManager.list_available_models(db)
        return {"models": models_list, "total": len(models_list)}
    except Exception as e:
        log.error("Error listing models", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{version}")
async def delete_model(
    version: str,
    delete_files: bool = True,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Delete a trained model."""
    try:
        log.warning("Deleting model", version=version, delete_files=delete_files)

        model = db.query(models.TrainedModel).filter(
            models.TrainedModel.version == version
        ).first()

        if not model:
            raise HTTPException(status_code=404, detail=f"Model {version} not found")

        # Delete files
        if delete_files and model.path and os.path.exists(model.path):
            shutil.rmtree(model.path)
            log.info("Model files deleted", path=model.path)

        # Delete from DB
        db.delete(model)
        db.commit()

        log.info("Model deleted successfully", version=version)

        return {
            "status": "success",
            "message": f"Model {version} deleted",
            "deleted_files": delete_files
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("Error deleting model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@router.get("/config/training")
async def get_training_config(api_key: str = Depends(get_api_key)):
    """Get current training configuration."""
    try:
        config = get_active_learning_config()
        return {
            "base_model": config.training.base_model,
            "num_train_epochs": config.training.num_train_epochs,
            "per_device_train_batch_size": config.training.per_device_train_batch_size,
            "learning_rate": config.training.learning_rate,
            "eval_split": config.training.eval_split,
            "min_training_samples": config.training.min_training_samples
        }
    except Exception as e:
        log.error("Error fetching training config", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/update")
async def update_config(
    request: ConfigUpdateRequest,
    api_key: str = Depends(get_api_key)
):
    """Update configuration parameters."""
    try:
        log.info("Updating configuration", section=request.config_section)

        config_file = "config/active_learning_config.yaml"

        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Update the specified section
        if request.config_section not in config_data:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid config section: {request.config_section}"
            )

        config_data[request.config_section].update(request.updates)

        # Save config
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        # Clear cache to reload config
        from app.core.active_learning_config import _active_learning_config_cache
        global _active_learning_config_cache
        _active_learning_config_cache = None

        log.info("Configuration updated successfully", section=request.config_section)

        return {
            "status": "success",
            "message": f"Configuration section '{request.config_section}' updated",
            "updates": request.updates
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error("Error updating configuration", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SYSTEM OPERATIONS
# ============================================================================

@router.get("/system/health", response_model=SystemHealthResponse)
async def system_health(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Check system health status."""
    try:
        # Database check
        try:
            db.execute(text("SELECT 1"))
            db_status = "healthy"
        except:
            db_status = "unhealthy"

        # MinIO check
        try:
            config = get_active_learning_config()
            minio_client = Minio(
                config.minio.endpoint,
                access_key=config.minio.access_key,
                secret_key=config.minio.secret_key,
                secure=config.minio.secure
            )
            minio_client.bucket_exists(config.minio.bucket)
            minio_status = "healthy"
        except:
            minio_status = "unhealthy"

        # System resources
        disk = psutil.disk_usage('/')
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)

        return SystemHealthResponse(
            database=db_status,
            minio=minio_status,
            disk_usage_percent=disk.percent,
            memory_usage_percent=memory.percent,
            cpu_usage_percent=cpu
        )

    except Exception as e:
        log.error("Error checking system health", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/clear-cache")
async def clear_cache(api_key: str = Depends(get_api_key)):
    """Clear application cache."""
    try:
        log.info("Clearing application cache")

        # Clear config cache
        from app.core.active_learning_config import _active_learning_config_cache
        global _active_learning_config_cache
        _active_learning_config_cache = None

        log.info("Cache cleared successfully")

        return {
            "status": "success",
            "message": "Cache cleared"
        }

    except Exception as e:
        log.error("Error clearing cache", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DATASET MANAGEMENT
# ============================================================================

@router.get("/datasets/list")
async def list_datasets(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """List all dataset versions."""
    try:
        datasets = db.query(models.DatasetVersion).order_by(
            models.DatasetVersion.created_at.desc()
        ).all()

        return {
            "datasets": [
                {
                    "id": d.id,
                    "version_name": d.version_name,
                    "description": d.description,
                    "created_at": d.created_at.isoformat() if d.created_at else None
                }
                for d in datasets
            ],
            "total": len(datasets)
        }

    except Exception as e:
        log.error("Error listing datasets", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasets/{version_name}")
async def delete_dataset(
    version_name: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Delete a dataset version."""
    try:
        log.warning("Deleting dataset", version_name=version_name)

        dataset = db.query(models.DatasetVersion).filter(
            models.DatasetVersion.version_name == version_name
        ).first()

        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {version_name} not found")

        # Delete from MinIO
        try:
            config = get_active_learning_config()
            minio_client = Minio(
                config.minio.endpoint,
                access_key=config.minio.access_key,
                secret_key=config.minio.secret_key,
                secure=config.minio.secure
            )

            # Delete both JSON and IOB versions
            for suffix in ["", "_iob"]:
                object_name = f"datasets/{version_name}{suffix}.json"
                try:
                    minio_client.remove_object(config.minio.bucket, object_name)
                    log.info("Dataset file deleted from MinIO", object_name=object_name)
                except S3Error as e:
                    log.warning("Could not delete dataset file", object_name=object_name, error=str(e))

        except Exception as e:
            log.warning("Error deleting dataset from MinIO", error=str(e))

        # Delete from DB
        db.delete(dataset)
        db.commit()

        log.info("Dataset deleted successfully", version_name=version_name)

        return {
            "status": "success",
            "message": f"Dataset {version_name} deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("Error deleting dataset", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NER REAPPLICATION
# ============================================================================

@router.post("/reapply-ner")
async def reapply_ner(
    request: ReapplyNERRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Reapply NER to tasks using legacy pipeline or a trained model.

    - task_ids: List of task IDs to reprocess (None = all pending tasks)
    - model_version: Model version to use (None = legacy pipeline)
    - replace_existing: Delete existing entities and create new ones
    """
    try:
        log.info("Reapplying NER",
                 task_count=len(request.task_ids) if request.task_ids else "all",
                 model_version=request.model_version or "legacy",
                 replace_existing=request.replace_existing)

        # Get tasks to process
        if request.task_ids:
            tasks = db.query(models.AnnotationTask).filter(
                models.AnnotationTask.id.in_(request.task_ids)
            ).all()
        else:
            # Get all pending tasks
            tasks = db.query(models.AnnotationTask).filter(
                models.AnnotationTask.status == "pending"
            ).all()

        if not tasks:
            raise HTTPException(status_code=404, detail="No tasks found to process")

        # Load pipeline or model
        if request.model_version:
            # Use trained model
            model = db.query(models.TrainedModel).filter(
                models.TrainedModel.version == request.model_version
            ).first()

            if not model:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model version {request.model_version} not found"
                )

            # Activate model temporarily
            ModelManager.activate_model(db, request.model_version)
            pipeline = ModelManager.get_pipeline()
            log.info("Using trained model", version=request.model_version)
        else:
            # Use legacy rule-based pipeline - usa la stessa pipeline di /predict
            from app.core.dependencies import get_legal_pipeline
            pipeline = get_legal_pipeline()
            log.info("Using specialized pipeline (Italian Legal NER) from get_legal_pipeline")

        # Process all tasks synchronously for now to debug
        if tasks:
            log.info("Processing tasks synchronously for debugging")
            try:
                # Crea una nuova sessione per il processing sincrono
                from app.database.database import SessionLocal
                processing_db = SessionLocal()
                await _reapply_ner_background(tasks, pipeline, request.replace_existing, processing_db)
                processing_db.close()
                log.info("All tasks processed successfully")
            except Exception as e:
                log.error("Error processing tasks", error=str(e))
                raise HTTPException(status_code=500, detail=f"Error processing tasks: {str(e)}")

        return {
            "status": "success",
            "message": f"Started NER reapplication for {len(tasks)} tasks",
            "task_count": len(tasks),
            "model": request.model_version or "specialized_pipeline",
            "processing": "background"
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error("Error reapplying NER", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _reapply_ner_background(
    tasks: List[models.AnnotationTask],
    pipeline,
    replace_existing: bool,
    db: Session
):
    """Background task to reapply NER."""
    log.info("Starting background NER reapplication", task_count=len(tasks))

    processed = 0
    errors = 0

    for task in tasks:
        try:
            # Get document
            document = db.query(models.Document).filter(
                models.Document.id == task.document_id
            ).first()

            if not document:
                log.warning("Document not found for task", task_id=task.id)
                continue

            # Delete existing entities if requested
            if replace_existing:
                db.query(models.Entity).filter(
                    models.Entity.document_id == document.id
                ).delete()
                log.info("Deleted existing entities", document_id=document.id)

            # Run NER pipeline
            entities = await pipeline.extract_legal_sources(document.text)
            log.info("Pipeline extracted entities",
                    entity_count=len(entities),
                    entities_sample=entities[:1] if entities else "none")

            if not entities:
                log.warning("No entities extracted from document",
                          document_id=document.id,
                          text_length=len(document.text))
                # Skip to next task if no entities found
                processed += 1
                continue

            # Save new entities
            entities_saved = 0

            # Usa la mappatura centralizzata
            from app.core.label_mapping import act_type_to_label as convert_act_type_to_label

            for entity_data in entities:
                log.debug("Processing entity", entity_data=entity_data)
                act_type = entity_data.get("act_type", "unknown")
                label = convert_act_type_to_label(act_type)

                # Converti valori a tipi Python nativi per evitare errori con PostgreSQL
                start_char_val = entity_data.get("start_char", 0)
                end_char_val = entity_data.get("end_char", 0)
                confidence_val = entity_data.get("confidence", 0.0)

                # Converti tensori/numpy a int/float Python nativi
                import torch
                import numpy as np

                if isinstance(start_char_val, (torch.Tensor, np.integer)):
                    start_char_val = int(start_char_val.item() if hasattr(start_char_val, 'item') else start_char_val)
                else:
                    start_char_val = int(start_char_val)

                if isinstance(end_char_val, (torch.Tensor, np.integer)):
                    end_char_val = int(end_char_val.item() if hasattr(end_char_val, 'item') else end_char_val)
                else:
                    end_char_val = int(end_char_val)

                if isinstance(confidence_val, (torch.Tensor, np.floating, np.integer)):
                    confidence_val = float(confidence_val.item() if hasattr(confidence_val, 'item') else confidence_val)
                else:
                    confidence_val = float(confidence_val)

                entity = models.Entity(
                    document_id=document.id,
                    text=entity_data.get("text", ""),
                    label=label,
                    start_char=start_char_val,
                    end_char=end_char_val,
                    confidence=confidence_val,
                    model="specialized_pipeline"
                )
                db.add(entity)
                entities_saved += 1
                log.debug("Entity added to session",
                         text=entity_data.get("text"),
                         label=label,
                         start=start_char_val,
                         end=end_char_val)

            # Commit dopo aver aggiunto tutte le entità
            db.commit()
            log.info("Entities saved to database",
                    document_id=document.id,
                    entities_saved=entities_saved,
                    entities_expected=len(entities))

            processed += 1

            log.info("Task reprocessed successfully",
                     task_id=task.id,
                     entities_found=len(entities),
                     entities_saved=entities_saved)

        except Exception as e:
            db.rollback()
            errors += 1
            log.error("Error reprocessing task",
                     task_id=task.id,
                     error=str(e))

    log.info("NER reapplication completed",
             processed=processed,
             errors=errors,
             total=len(tasks))
