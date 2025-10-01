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
    Adds a new entity label to the system's configuration.
    This is a critical operation and should be used with care.
    """
    log.info("Request to add new label", new_label=request.name)
    
    # Sanitize the new label name
    new_label_core = request.name.strip().lower().replace(" ", "_")
    if not new_label_core or not new_label_core.isidentifier():
        raise HTTPException(status_code=400, detail=f"Invalid label name: '{request.name}'. Use letters, numbers, and underscores.")

    try:
        # Read the existing YAML file to preserve comments and structure
        with open(ACTIVE_LEARNING_CONFIG_PATH, 'r') as f:
            config_data = yaml.safe_load(f)

        labels_config = config_data.get("labels", {})
        label_list = labels_config.get("label_list", [])
        label2id = labels_config.get("label2id", {})
        
        # Check if label already exists
        if f"B-{new_label_core}" in label_list:
            log.warning("Attempted to add a duplicate label", label=new_label_core)
            raise HTTPException(status_code=409, detail=f"Label '{new_label_core}' already exists.")

        # Add the new label to the list
        label_list.append(f"B-{new_label_core}")
        label_list.append(f"I-{new_label_core}")
        
        # Update the ID mappings
        max_id = max(label2id.values()) if label2id else 0
        label2id[f"B-{new_label_core}"] = max_id + 1
        label2id[f"I-{new_label_core}"] = max_id + 2
        
        # Rebuild id2label map for consistency
        id2label = {v: k for k, v in label2id.items()}
        labels_config["id2label"] = id2label
        labels_config["label2id"] = label2id
        labels_config["label_list"] = label_list

        # Write the updated data back to the file
        with open(ACTIVE_LEARNING_CONFIG_PATH, 'w') as f:
            yaml.dump(config_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

        log.info("Successfully added new label", new_label=new_label_core)
        return {"status": "success", "new_label": new_label_core}

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to add new label", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
