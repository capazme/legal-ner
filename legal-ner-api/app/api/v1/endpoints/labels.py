"""
Labels API Endpoint

Provides access to the NER model's label schema.
"""

from fastapi import APIRouter, Depends, HTTPException
import structlog
from typing import List
import yaml
from pydantic import BaseModel

from app.core.active_learning_config import get_active_learning_config, ACTIVE_LEARNING_CONFIG_PATH
from app.core.dependencies import get_api_key

log = structlog.get_logger()
router = APIRouter()

class NewLabelRequest(BaseModel):
    name: str

@router.get("/labels", response_model=List[str])
def get_labels(api_key: str = Depends(get_api_key)):
    """
    Returns the list of unique entity labels used by the model.
    
    The labels are extracted from the active learning configuration,
    and prefixes (B-, I-) are removed to get the core entity type.
    """
    try:
        log.info("Fetching NER labels")
        config = get_active_learning_config()
        
        raw_labels = config.labels.label_list
        
        # Extract unique core labels (e.g., "codice_civile" from "B-codice_civile")
        core_labels = set()
        for label in raw_labels:
            if label == "O":
                continue
            # Remove B- or I- prefix
            core_label = label[2:]
            core_labels.add(core_label)
            
        sorted_labels = sorted(list(core_labels))
        log.info(f"Returning {len(sorted_labels)} core labels.")
        return sorted_labels
        
    except Exception as e:
        log.error("Failed to retrieve labels from configuration.", error=str(e))
        return []

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
