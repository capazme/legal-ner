"""
Active Learning Configuration Loader

Carica e gestisce la configurazione per il sistema di active learning.
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog

log = structlog.get_logger()

@dataclass
class PathsConfig:
    """Configurazione dei percorsi."""
    models_base_dir: str
    training_logs_dir: str
    temp_datasets_dir: str

@dataclass
class MinIOConfig:
    """Configurazione MinIO."""
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False

@dataclass
class TrainingConfig:
    """Configurazione training."""
    base_model: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    evaluation_strategy: str
    save_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    save_total_limit: int
    eval_split: float
    min_training_samples: int

@dataclass
class ActiveLearningStrategyConfig:
    """Configurazione strategia active learning."""
    selection_strategy: str
    default_batch_size: int
    uncertainty_threshold: float
    high_confidence_threshold: float
    use_fine_tuned_if_available: bool
    fine_tuned_confidence_threshold: float

@dataclass
class LabelsConfig:
    """Configurazione label."""
    label_list: List[str]

    def __post_init__(self):
        """Costruisce label2id e id2label."""
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

@dataclass
class LoggingConfig:
    """Configurazione logging."""
    enabled: bool
    level: str
    format: str
    log_to_file: bool
    log_file_path: str
    log_to_console: bool
    max_bytes: int
    backup_count: int

@dataclass
class DatasetConfig:
    """Configurazione dataset."""
    max_sequence_length: int
    min_entities_per_document: int
    min_document_length: int
    require_validation: bool

@dataclass
class VersioningConfig:
    """Configurazione versioning."""
    version_prefix: str
    timestamp_format: str
    keep_best_n_models: int
    save_metadata: List[str]

@dataclass
class MonitoringConfig:
    """Configurazione monitoring."""
    track_metrics: List[str]
    alert_if_eval_loss_above: float
    alert_if_f1_below: float

@dataclass
class UIConfig:
    """Configurazione UI."""
    port: int
    entities_per_page: int
    annotation_timeout: int

@dataclass
class ActiveLearningConfig:
    """Configurazione completa active learning."""
    paths: PathsConfig
    minio: MinIOConfig
    training: TrainingConfig
    active_learning: ActiveLearningStrategyConfig
    labels: LabelsConfig
    logging: LoggingConfig
    dataset: DatasetConfig
    versioning: VersioningConfig
    monitoring: MonitoringConfig
    ui: UIConfig

    def get_model_output_dir(self, version_name: str) -> str:
        """Restituisce il path completo per un modello."""
        return os.path.join(self.paths.models_base_dir, version_name)

    def get_training_log_dir(self, version_name: str) -> str:
        """Restituisce il path per i log di training."""
        return os.path.join(self.paths.training_logs_dir, version_name)

    def ensure_directories(self):
        """Crea le directory necessarie se non esistono."""
        dirs_to_create = [
            self.paths.models_base_dir,
            self.paths.training_logs_dir,
            self.paths.temp_datasets_dir,
        ]

        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            log.debug("Directory ensured", path=dir_path)


def load_active_learning_config(config_path: Optional[str] = None) -> ActiveLearningConfig:
    """
    Carica la configurazione active learning dal file YAML.

    Args:
        config_path: Path al file di configurazione (opzionale)
                    Default: config/active_learning_config.yaml

    Returns:
        ActiveLearningConfig: Configurazione caricata
    """
    if config_path is None:
        # Usa il path di default
        base_dir = Path(__file__).parent.parent.parent  # legal-ner-api/
        config_path = base_dir / "config" / "active_learning_config.yaml"

    log.info("Loading active learning configuration", config_path=str(config_path))

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Active learning config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    # Costruisci i dataclass dalle sezioni del config
    config = ActiveLearningConfig(
        paths=PathsConfig(**config_data['paths']),
        minio=MinIOConfig(**config_data['minio']),
        training=TrainingConfig(**config_data['training']),
        active_learning=ActiveLearningStrategyConfig(**config_data['active_learning']),
        labels=LabelsConfig(**config_data['labels']),
        logging=LoggingConfig(**config_data['logging']),
        dataset=DatasetConfig(**config_data['dataset']),
        versioning=VersioningConfig(**config_data['versioning']),
        monitoring=MonitoringConfig(**config_data['monitoring']),
        ui=UIConfig(**config_data['ui']),
    )

    # Crea le directory necessarie
    config.ensure_directories()

    log.info("Active learning configuration loaded successfully",
             base_model=config.training.base_model,
             num_labels=len(config.labels.label_list))

    return config


# Singleton per caching della configurazione
_active_learning_config_cache: Optional[ActiveLearningConfig] = None

def get_active_learning_config() -> ActiveLearningConfig:
    """
    Restituisce la configurazione active learning (cached).

    Returns:
        ActiveLearningConfig: Configurazione caricata
    """
    global _active_learning_config_cache

    if _active_learning_config_cache is None:
        _active_learning_config_cache = load_active_learning_config()

    return _active_learning_config_cache