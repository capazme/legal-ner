"""
Sistema di Caricamento Configurazione Pipeline
===========================================

Gestisce il caricamento e la validazione della configurazione YAML
per la pipeline specializzata di Legal-NER.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import structlog

log = structlog.get_logger()

@dataclass
class ModelConfig:
    """Configurazione modelli AI."""
    entity_detector_primary: str
    entity_detector_fallback: str
    entity_detector_max_length: int
    legal_classifier_primary: str
    legal_classifier_max_length: int
    semantic_correlator_model: str

@dataclass
class ConfidenceConfig:
    """Configurazione soglie di confidence."""
    minimum_detection_confidence: float
    rule_based_priority_threshold: float
    semantic_boost_factor: float
    semantic_similarity_scale: float

    # Confidence rule-based
    specific_codes: float
    generic_codes: float
    testo_unico: float
    decreto_legislativo_full: float
    decreto_legislativo_abbrev: float
    dpr_full: float
    dpr_abbrev: float
    legge_full: float
    legge_abbrev: float
    costituzione_full: float
    costituzione_abbrev: float
    convention: float
    institution: float
    direttiva_ue: float
    trattato: float
    generic_article: float
    default: float

@dataclass
class ContextConfig:
    """Configurazione finestre di contesto."""
    # Entity expansion
    left_window: int
    right_window: int
    context_window: int

    # Semantic context
    immediate_context: int
    extended_context: int
    full_context: int

    # Classification context
    classification_context: int

@dataclass
class SpuriousFiltersConfig:
    """Configurazione filtri anti-spurio."""
    min_length: int
    valid_short_terms: List[str]
    spurious_words: List[str]
    spurious_patterns: List[str]
    filter_single_alpha: bool
    min_detection_confidence: float

@dataclass
class OutputConfig:
    """Configurazione output."""
    filter_institutions: bool
    filter_null_values: bool
    enable_debug_logging: bool
    log_pattern_matches: bool
    max_logged_patterns: int

@dataclass
class PipelineConfig:
    """Configurazione completa della pipeline."""
    models: ModelConfig
    confidence: ConfidenceConfig
    context: ContextConfig
    normattiva_mapping: Dict[str, List[str]]
    regex_patterns: Dict[str, List[str]]
    context_patterns: Dict[str, List[str]]
    boundary_expansion: Dict[str, List[str]]
    semantic_prototypes: Dict[str, List[str]]
    parsing_patterns: Dict[str, str]
    spurious_filters: SpuriousFiltersConfig
    legal_context_words: List[str]
    output: OutputConfig

class ConfigLoader:
    """Caricatore e validatore configurazione pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Inizializza il caricatore configurazione.

        Args:
            config_path: Percorso del file di configurazione YAML.
                        Se None, usa il percorso default.
        """
        if config_path is None:
            # Percorso default relativo alla root del progetto
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "pipeline_config.yaml"

        self.config_path = Path(config_path)
        self._raw_config: Optional[Dict[str, Any]] = None
        self._parsed_config: Optional[PipelineConfig] = None

    def load_config(self) -> PipelineConfig:
        """
        Carica e valida la configurazione dal file YAML.

        Returns:
            Configurazione pipeline validata.

        Raises:
            FileNotFoundError: Se il file di configurazione non esiste.
            yaml.YAMLError: Se ci sono errori nel parsing YAML.
            ValueError: Se la configurazione non è valida.
        """
        log.info("Loading pipeline configuration", config_path=str(self.config_path))

        # Verifica esistenza file
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Carica YAML
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            log.error("Failed to parse YAML configuration", error=str(e))
            raise

        # Valida e costruisce configurazione
        try:
            self._parsed_config = self._build_config(self._raw_config)
            log.info("Pipeline configuration loaded successfully")
            return self._parsed_config
        except Exception as e:
            log.error("Failed to validate configuration", error=str(e))
            raise ValueError(f"Invalid configuration: {e}")

    def _build_config(self, raw_config: Dict[str, Any]) -> PipelineConfig:
        """Costruisce e valida la configurazione strutturata."""

        # Modelli
        models_raw = raw_config["models"]
        models = ModelConfig(
            entity_detector_primary=models_raw["entity_detector"]["primary"],
            entity_detector_fallback=models_raw["entity_detector"]["fallback"],
            entity_detector_max_length=models_raw["entity_detector"]["max_length"],
            legal_classifier_primary=models_raw["legal_classifier"]["primary"],
            legal_classifier_max_length=models_raw["legal_classifier"]["embedding_max_length"],
            semantic_correlator_model=models_raw["semantic_correlator"]["model"]
        )

        # Confidence
        conf_raw = raw_config["confidence_thresholds"]
        rule_conf = conf_raw["rule_based_confidence"]
        confidence = ConfidenceConfig(
            minimum_detection_confidence=conf_raw["minimum_detection_confidence"],
            rule_based_priority_threshold=conf_raw["rule_based_priority_threshold"],
            semantic_boost_factor=conf_raw["semantic_boost_factor"],
            semantic_similarity_scale=conf_raw["semantic_similarity_scale"],
            specific_codes=rule_conf["specific_codes"],
            generic_codes=rule_conf["generic_codes"],
            testo_unico=rule_conf["testo_unico"],
            decreto_legislativo_full=rule_conf["decreto_legislativo_full"],
            decreto_legislativo_abbrev=rule_conf["decreto_legislativo_abbrev"],
            dpr_full=rule_conf["dpr_full"],
            dpr_abbrev=rule_conf["dpr_abbrev"],
            legge_full=rule_conf["legge_full"],
            legge_abbrev=rule_conf["legge_abbrev"],
            costituzione_full=rule_conf["costituzione_full"],
            costituzione_abbrev=rule_conf["costituzione_abbrev"],
            convention=rule_conf["convention"],
            institution=rule_conf["institution"],
            direttiva_ue=rule_conf["direttiva_ue"],
            trattato=rule_conf["trattato"],
            generic_article=rule_conf["generic_article"],
            default=rule_conf["default"]
        )

        # Context windows
        ctx_raw = raw_config["context_windows"]
        context = ContextConfig(
            left_window=ctx_raw["entity_expansion"]["left_window"],
            right_window=ctx_raw["entity_expansion"]["right_window"],
            context_window=ctx_raw["entity_expansion"]["context_window"],
            immediate_context=ctx_raw["semantic_context"]["immediate_context"],
            extended_context=ctx_raw["semantic_context"]["extended_context"],
            full_context=ctx_raw["semantic_context"]["full_context"],
            classification_context=ctx_raw["classification_context"]
        )

        # Spurious filters
        spur_raw = raw_config["spurious_filters"]
        spurious_filters = SpuriousFiltersConfig(
            min_length=spur_raw["min_length"],
            valid_short_terms=spur_raw["valid_short_terms"],
            spurious_words=spur_raw["spurious_words"],
            spurious_patterns=spur_raw.get("spurious_patterns", []),
            filter_single_alpha=spur_raw["filter_single_alpha"],
            min_detection_confidence=spur_raw["min_detection_confidence"]
        )

        # Output settings
        out_raw = raw_config["output_settings"]
        output = OutputConfig(
            filter_institutions=out_raw["filter_institutions"],
            filter_null_values=out_raw["filter_null_values"],
            enable_debug_logging=out_raw["enable_debug_logging"],
            log_pattern_matches=out_raw["log_pattern_matches"],
            max_logged_patterns=out_raw["max_logged_patterns"]
        )

        return PipelineConfig(
            models=models,
            confidence=confidence,
            context=context,
            normattiva_mapping=raw_config["normattiva_mapping"],
            regex_patterns=raw_config["regex_patterns"],
            context_patterns=raw_config["context_patterns"],
            boundary_expansion=raw_config["boundary_expansion"],
            semantic_prototypes=raw_config["semantic_prototypes"],
            parsing_patterns=raw_config["parsing_patterns"],
            spurious_filters=spurious_filters,
            legal_context_words=raw_config["legal_context_words"],
            output=output
        )

    def get_normattiva_flat_mapping(self) -> Dict[str, str]:
        """
        Restituisce la mappatura NORMATTIVA in formato piatto.

        Returns:
            Dizionario {abbreviazione: tipo_normalizzato}
        """
        if not self._parsed_config:
            raise ValueError("Configuration not loaded. Call load_config() first.")

        flat_mapping = {}
        for act_type, abbreviations in self._parsed_config.normattiva_mapping.items():
            normalized_type = act_type.replace("_", ".")
            for abbrev in abbreviations:
                flat_mapping[abbrev] = normalized_type

        return flat_mapping

    def get_all_regex_patterns(self) -> List[str]:
        """
        Restituisce tutti i pattern regex in una lista piatta.

        Returns:
            Lista di tutti i pattern regex configurati.
        """
        if not self._parsed_config:
            raise ValueError("Configuration not loaded. Call load_config() first.")

        all_patterns = []
        for pattern_group in self._parsed_config.regex_patterns.values():
            all_patterns.extend(pattern_group)

        return all_patterns

    def get_all_context_patterns(self) -> List[str]:
        """
        Restituisce tutti i pattern contestuali in una lista piatta.

        Returns:
            Lista di tutti i pattern contestuali configurati.
        """
        if not self._parsed_config:
            raise ValueError("Configuration not loaded. Call load_config() first.")

        all_patterns = []
        for pattern_group in self._parsed_config.context_patterns.values():
            all_patterns.extend(pattern_group)

        return all_patterns

    def validate_config(self) -> bool:
        """
        Valida la configurazione caricata.

        Returns:
            True se la configurazione è valida.

        Raises:
            ValueError: Se la configurazione non è valida.
        """
        if not self._parsed_config:
            raise ValueError("Configuration not loaded. Call load_config() first.")

        config = self._parsed_config

        # Valida confidence values (0.0 - 1.0)
        confidence_values = [
            config.confidence.minimum_detection_confidence,
            config.confidence.rule_based_priority_threshold,
            config.confidence.specific_codes,
            config.confidence.generic_codes,
            # ... altri valori
        ]

        for val in confidence_values:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Confidence value out of range [0.0, 1.0]: {val}")

        # Valida context windows (positivi)
        context_values = [
            config.context.left_window,
            config.context.right_window,
            config.context.context_window,
            config.context.immediate_context,
            config.context.extended_context,
            config.context.full_context,
            config.context.classification_context
        ]

        for val in context_values:
            if val <= 0:
                raise ValueError(f"Context window must be positive: {val}")

        # Valida model lengths
        if config.models.entity_detector_max_length <= 0:
            raise ValueError("Entity detector max_length must be positive")

        if config.models.legal_classifier_max_length <= 0:
            raise ValueError("Legal classifier max_length must be positive")

        log.info("Configuration validation passed")
        return True

# Singleton instance
_config_loader: Optional[ConfigLoader] = None
_loaded_config: Optional[PipelineConfig] = None

def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Ottiene l'istanza singleton del config loader.

    Args:
        config_path: Percorso del file di configurazione (solo al primo utilizzo).

    Returns:
        Istanza del config loader.
    """
    global _config_loader

    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)

    return _config_loader

def get_pipeline_config(reload: bool = False) -> PipelineConfig:
    """
    Ottiene la configurazione pipeline caricata.

    Args:
        reload: Se True, ricarica la configurazione dal file.

    Returns:
        Configurazione pipeline validata.
    """
    global _loaded_config

    if _loaded_config is None or reload:
        loader = get_config_loader()
        _loaded_config = loader.load_config()
        loader.validate_config()

    return _loaded_config