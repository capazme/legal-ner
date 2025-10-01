"""
Labels API Endpoint

Provides access to the NER model's label schema.
"""

from fastapi import APIRouter, Depends, HTTPException
import structlog
from typing import List, Dict
import yaml
from pydantic import BaseModel

from app.core.active_learning_config import get_active_learning_config, ACTIVE_LEARNING_CONFIG_PATH
from app.core.dependencies import get_api_key
from app.core.label_mapping import (
    get_all_labels,
    get_label_categories,
    get_act_type_to_label_mapping,
    act_type_to_label as convert_act_type_to_label,
    label_to_act_type as convert_label_to_act_type,
    get_label_category,
    update_label_mapping,
    remove_label_mapping,
    reload_label_config
)

log = structlog.get_logger()
router = APIRouter()

class NewLabelRequest(BaseModel):
    name: str

class UpdateLabelRequest(BaseModel):
    old_label: str
    new_label: str
    category: str = "Altro"

class DeleteLabelRequest(BaseModel):
    label: str

@router.get("/labels", response_model=List[str])
def get_labels_endpoint(api_key: str = Depends(get_api_key)):
    """
    Returns the list of standardized entity labels used by the system.

    These are the display labels (e.g., 'D.LGS', 'CODICE_CIVILE')
    that are shown to users, not the internal act_type values.
    """
    try:
        log.info("Fetching standardized NER labels")

        # Restituisci le label standardizzate dalla mappatura centralizzata
        sorted_labels = get_all_labels()
        log.info(f"Returning {len(sorted_labels)} standardized labels.")
        return sorted_labels

    except Exception as e:
        log.error("Failed to retrieve labels.", error=str(e))
        return []

@router.get("/labels/categories", response_model=Dict[str, List[str]])
def get_labels_by_category_endpoint(api_key: str = Depends(get_api_key)):
    """
    Returns all labels organized by category.

    Returns:
        Dictionary with category names as keys and lists of labels as values.
        Example: {"Decreti": ["D.LGS", "D.L", ...], "Leggi": ["LEGGE", ...]}
    """
    try:
        log.info("Fetching labels organized by category")
        return get_label_categories()

    except Exception as e:
        log.error("Failed to retrieve label categories.", error=str(e))
        return {}

@router.get("/labels/mapping", response_model=Dict[str, str])
def get_label_mapping_endpoint(api_key: str = Depends(get_api_key)):
    """
    Returns the complete act_type -> label mapping.

    Returns:
        Dictionary mapping internal act_type values to display labels.
        Example: {"decreto_legislativo": "D.LGS", "codice_civile": "CODICE_CIVILE", ...}
    """
    try:
        log.info("Fetching label mapping")
        return get_act_type_to_label_mapping()

    except Exception as e:
        log.error("Failed to retrieve label mapping.", error=str(e))
        return {}

@router.post("/labels/reload", status_code=200)
def reload_labels(api_key: str = Depends(get_api_key)):
    """
    Ricarica la configurazione delle label dal file.

    Utile dopo modifiche manuali al file di configurazione.
    """
    try:
        log.info("Reloading label configuration")
        reload_label_config()
        return {"status": "success", "message": "Label configuration reloaded"}

    except Exception as e:
        log.error("Failed to reload label configuration.", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to reload: {str(e)}")

@router.post("/labels", status_code=201)
def add_label(
    request: NewLabelRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Adds a new entity label to BOTH systems:
    1. active_learning_config.yaml (BIO format for training)
    2. Centralized label_mapping (standardized format for display/annotation/pipeline)

    This ensures consistency across specialized_pipeline, admin panel, and annotation tasks.
    """
    log.info("Request to add new label", new_label=request.name)

    # Sanitize - standardized format (uppercase with underscores)
    new_label_standard = request.name.strip().upper().replace(" ", "_")
    # BIO format (lowercase with underscores)
    new_label_bio = request.name.strip().lower().replace(" ", "_")

    if not new_label_bio or not new_label_bio.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail=f"Invalid label name: '{request.name}'. Use letters, numbers, and underscores.")

    try:
        # ===================================================================
        # STEP 1: Add to active_learning_config.yaml (for training - BIO format)
        # ===================================================================
        with open(ACTIVE_LEARNING_CONFIG_PATH, 'r') as f:
            config_data = yaml.safe_load(f)

        labels_config = config_data.get("labels", {})
        label_list = labels_config.get("label_list", [])
        label2id = labels_config.get("label2id", {})

        # Check if label already exists in BIO format
        if f"B-{new_label_bio}" in label_list:
            log.warning("Attempted to add a duplicate label", label=new_label_bio)
            raise HTTPException(status_code=409, detail=f"Label '{new_label_bio}' already exists.")

        # Add the new label to the list (BIO format)
        label_list.append(f"B-{new_label_bio}")
        label_list.append(f"I-{new_label_bio}")

        # Update the ID mappings
        max_id = max(label2id.values()) if label2id else 0
        label2id[f"B-{new_label_bio}"] = max_id + 1
        label2id[f"I-{new_label_bio}"] = max_id + 2

        # Rebuild id2label map for consistency
        id2label = {v: k for k, v in label2id.items()}
        labels_config["id2label"] = id2label
        labels_config["label2id"] = label2id
        labels_config["label_list"] = label_list

        # Write the updated data back to the file
        with open(ACTIVE_LEARNING_CONFIG_PATH, 'w') as f:
            yaml.dump(config_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

        log.info("Added label to active_learning_config.yaml", label_bio=new_label_bio)

        # ===================================================================
        # STEP 2: Add to centralized label_mapping (standardized format)
        # ===================================================================
        # This ensures the label is available in:
        # - specialized_pipeline (when it classifies entities)
        # - annotation UI (when users select labels)
        # - admin panel (when viewing label statistics)
        update_label_mapping(
            act_type=new_label_bio,          # Internal act_type (lowercase)
            label=new_label_standard,        # Display label (uppercase)
            category="Custom"                # Custom category for user-added labels
        )

        log.info("Added label to centralized label_mapping", label_standard=new_label_standard)

        # Return standardized format (what users will see)
        return {
            "status": "success",
            "new_label": new_label_standard,
            "bio_labels": [f"B-{new_label_bio}", f"I-{new_label_bio}"],
            "message": f"Label '{new_label_standard}' added to both systems"
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to add new label", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/labels", status_code=200)
def update_label(
    request: UpdateLabelRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Updates an existing label name and/or category.

    This will:
    1. Update the label in active_learning_config.yaml (BIO format)
    2. Update the label in centralized label_mapping
    3. Update all entities in the database with the new label
    """
    log.info("Request to update label", old_label=request.old_label, new_label=request.new_label)

    # Normalize labels
    old_label_standard = request.old_label.strip().upper().replace(" ", "_")
    new_label_standard = request.new_label.strip().upper().replace(" ", "_")
    old_label_bio = request.old_label.strip().lower().replace(" ", "_")
    new_label_bio = request.new_label.strip().lower().replace(" ", "_")

    if not new_label_bio or not new_label_bio.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail=f"Invalid label name: '{request.new_label}'")

    try:
        # Check if old label exists
        all_labels = get_all_labels()
        if old_label_standard not in all_labels:
            raise HTTPException(status_code=404, detail=f"Label '{old_label_standard}' not found")

        # ===================================================================
        # STEP 1: Update active_learning_config.yaml (BIO format)
        # ===================================================================
        with open(ACTIVE_LEARNING_CONFIG_PATH, 'r') as f:
            config_data = yaml.safe_load(f)

        labels_config = config_data.get("labels", {})
        label_list = labels_config.get("label_list", [])
        label2id = labels_config.get("label2id", {})

        # Remove old BIO labels
        old_b = f"B-{old_label_bio}"
        old_i = f"I-{old_label_bio}"
        new_b = f"B-{new_label_bio}"
        new_i = f"I-{new_label_bio}"

        if old_b in label_list:
            # Get old IDs
            old_b_id = label2id.get(old_b)
            old_i_id = label2id.get(old_i)

            # Update label_list
            label_list[label_list.index(old_b)] = new_b
            label_list[label_list.index(old_i)] = new_i

            # Update label2id
            del label2id[old_b]
            del label2id[old_i]
            label2id[new_b] = old_b_id
            label2id[new_i] = old_i_id

            # Rebuild id2label
            id2label = {v: k for k, v in label2id.items()}
            labels_config["id2label"] = id2label
            labels_config["label2id"] = label2id
            labels_config["label_list"] = label_list

            # Save
            with open(ACTIVE_LEARNING_CONFIG_PATH, 'w') as f:
                yaml.dump(config_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

            log.info("Updated label in active_learning_config.yaml", old=old_label_bio, new=new_label_bio)

        # ===================================================================
        # STEP 2: Update centralized label_mapping
        # ===================================================================
        # Remove old mapping
        remove_label_mapping(old_label_bio)
        # Add new mapping
        update_label_mapping(
            act_type=new_label_bio,
            label=new_label_standard,
            category=request.category
        )

        log.info("Updated label in centralized label_mapping", old=old_label_standard, new=new_label_standard)

        # ===================================================================
        # STEP 3: Update entities in database
        # ===================================================================
        from sqlalchemy.orm import Session
        from app.core.database import get_db
        from app.db import models

        db = next(get_db())
        try:
            updated_count = db.query(models.Entity).filter(
                models.Entity.label == old_label_standard
            ).update({"label": new_label_standard})
            db.commit()
            log.info("Updated entities in database", count=updated_count)
        except Exception as e:
            db.rollback()
            log.error("Failed to update entities", error=str(e))
            raise
        finally:
            db.close()

        return {
            "status": "success",
            "old_label": old_label_standard,
            "new_label": new_label_standard,
            "entities_updated": updated_count,
            "message": f"Label updated from '{old_label_standard}' to '{new_label_standard}'"
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to update label", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/labels", status_code=200)
def delete_label(
    request: DeleteLabelRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Deletes a label from the system.

    WARNING: This will also delete all entities with this label from the database.
    Use with caution!

    This will:
    1. Remove the label from active_learning_config.yaml (BIO format)
    2. Remove the label from centralized label_mapping
    3. Delete all entities with this label from the database
    """
    log.info("Request to delete label", label=request.label)

    # Normalize label
    label_standard = request.label.strip().upper().replace(" ", "_")
    label_bio = request.label.strip().lower().replace(" ", "_")

    try:
        # Check if label exists
        all_labels = get_all_labels()
        if label_standard not in all_labels:
            raise HTTPException(status_code=404, detail=f"Label '{label_standard}' not found")

        # ===================================================================
        # STEP 1: Remove from active_learning_config.yaml (BIO format)
        # ===================================================================
        with open(ACTIVE_LEARNING_CONFIG_PATH, 'r') as f:
            config_data = yaml.safe_load(f)

        labels_config = config_data.get("labels", {})
        label_list = labels_config.get("label_list", [])
        label2id = labels_config.get("label2id", {})

        # Remove BIO labels
        b_label = f"B-{label_bio}"
        i_label = f"I-{label_bio}"

        if b_label in label_list:
            label_list.remove(b_label)
            label_list.remove(i_label)

            # Remove from label2id
            if b_label in label2id:
                del label2id[b_label]
            if i_label in label2id:
                del label2id[i_label]

            # Rebuild id2label
            id2label = {v: k for k, v in label2id.items()}
            labels_config["id2label"] = id2label
            labels_config["label2id"] = label2id
            labels_config["label_list"] = label_list

            # Save
            with open(ACTIVE_LEARNING_CONFIG_PATH, 'w') as f:
                yaml.dump(config_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

            log.info("Removed label from active_learning_config.yaml", label=label_bio)

        # ===================================================================
        # STEP 2: Remove from centralized label_mapping
        # ===================================================================
        remove_label_mapping(label_bio)
        log.info("Removed label from centralized label_mapping", label=label_standard)

        # ===================================================================
        # STEP 3: Delete entities from database
        # ===================================================================
        from sqlalchemy.orm import Session
        from app.core.database import get_db
        from app.db import models

        db = next(get_db())
        try:
            deleted_count = db.query(models.Entity).filter(
                models.Entity.label == label_standard
            ).delete()
            db.commit()
            log.info("Deleted entities from database", count=deleted_count)
        except Exception as e:
            db.rollback()
            log.error("Failed to delete entities", error=str(e))
            raise
        finally:
            db.close()

        return {
            "status": "success",
            "deleted_label": label_standard,
            "entities_deleted": deleted_count,
            "message": f"Label '{label_standard}' and {deleted_count} entities deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to delete label", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
