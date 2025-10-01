"""
Label Mapping Configuration
============================

Questo modulo centralizza la mappatura tra i tipi di atto normativi (act_type)
usati internamente dalla specialized pipeline e le label standardizzate
visualizzate nell'interfaccia utente e nel database.

La mappatura viene caricata dalla configurazione YAML per permettere modifiche
dinamiche senza dover cambiare il codice.

Questo garantisce coerenza in tutta l'applicazione e facilita la manutenzione.
"""

import yaml
from pathlib import Path
from typing import Dict, List
import structlog

log = structlog.get_logger()

# Path alla configurazione
LABEL_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "label_mapping.yaml"

# Cache per evitare di ricaricare il file ad ogni chiamata
_LABEL_CACHE = None

# Mappatura di default (usata se il file di configurazione non esiste)
DEFAULT_ACT_TYPE_TO_LABEL = {
    # Decreti
    'decreto_legislativo': 'D.LGS',
    'decreto_legge': 'D.L',
    'decreto_presidente_repubblica': 'D.P.R',
    'decreto_ministeriale': 'D.M',
    'decreto_presidente_consiglio': 'D.P.C.M',
    'dpcm': 'D.P.C.M',
    'decreto_dirigenziale': 'D.D',
    'decreto_assessore_regionale': 'D.A.R',

    # Leggi
    'legge': 'LEGGE',
    'legge_costituzionale': 'LEGGE_COST',
    'legge_regionale': 'L.R',
    'legge_provinciale': 'L.P',
    'legge_fallimentare': 'LEGGE_FALLIMENTARE',

    # Costituzione
    'costituzione': 'COSTITUZIONE',

    # Codici
    'codice_civile': 'CODICE_CIVILE',
    'codice_penale': 'CODICE_PENALE',
    'codice_procedura_civile': 'CODICE_PROCEDURA_CIVILE',
    'codice_procedura_penale': 'CODICE_PROCEDURA_PENALE',
    'codice_crisi_impresa': 'CODICE_CRISI_IMPRESA',
    'codice_beni_culturali': 'CODICE_BENI_CULTURALI',
    'generic_codes': 'CODICE',

    # Testi unici
    'testo_unico': 'T.U',
    'testo_unico_bancario': 'T.U.B',
    'testo_unico_enti_locali': 'T.U.E.L',
    'testo_unico_finanza': 'T.U.F',
    'testo_unico_lavoro_pubblica_sicurezza': 'T.U.L.P.S',
    'testo_unico_pubblica_sicurezza': 'T.U.P.S',

    # Normativa UE
    'direttiva_ue': 'DIR_UE',
    'regolamento_ue': 'REG_UE',
    'decisione_ue': 'DEC_UE',
    'raccomandazione_ue': 'RAC_UE',

    # Trattati e convenzioni
    'trattato': 'TRATTATO',
    'trattato_ue': 'TRATTATO_UE',
    'trattato_funzionamento_ue': 'TFUE',
    'convenzione_europea_diritti': 'CEDU',
    'convention': 'CONVENTION',

    # Altro
    'circolare': 'CIRCOLARE',
    'institution': 'ISTITUZIONE',
    'unknown': 'UNKNOWN',
    'fonte non identificata': 'UNKNOWN'
}

DEFAULT_LABEL_CATEGORIES = {
    'Decreti': ['D.LGS', 'D.L', 'D.P.R', 'D.M', 'D.P.C.M', 'D.D', 'D.A.R'],
    'Leggi': ['LEGGE', 'LEGGE_COST', 'L.R', 'L.P', 'LEGGE_FALLIMENTARE'],
    'Codici': ['CODICE_CIVILE', 'CODICE_PENALE', 'CODICE_PROCEDURA_CIVILE',
               'CODICE_PROCEDURA_PENALE', 'CODICE_CRISI_IMPRESA',
               'CODICE_BENI_CULTURALI', 'CODICE'],
    'Testi Unici': ['T.U', 'T.U.B', 'T.U.E.L', 'T.U.F', 'T.U.L.P.S', 'T.U.P.S'],
    'Normativa UE': ['DIR_UE', 'REG_UE', 'DEC_UE', 'RAC_UE', 'TRATTATO_UE', 'TFUE'],
    'Trattati': ['TRATTATO', 'CONVENTION', 'CEDU'],
    'Costituzione': ['COSTITUZIONE'],
    'Altro': ['CIRCOLARE', 'ISTITUZIONE', 'UNKNOWN']
}


def _load_label_config():
    """Carica la configurazione delle label dal file YAML."""
    global _LABEL_CACHE

    if _LABEL_CACHE is not None:
        return _LABEL_CACHE

    try:
        if LABEL_CONFIG_PATH.exists():
            with open(LABEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                _LABEL_CACHE = {
                    'act_type_to_label': config.get('act_type_to_label', DEFAULT_ACT_TYPE_TO_LABEL),
                    'label_categories': config.get('label_categories', DEFAULT_LABEL_CATEGORIES)
                }
                log.info("Label configuration loaded from file", path=str(LABEL_CONFIG_PATH))
        else:
            # Crea il file di configurazione con i valori di default
            _LABEL_CACHE = {
                'act_type_to_label': DEFAULT_ACT_TYPE_TO_LABEL,
                'label_categories': DEFAULT_LABEL_CATEGORIES
            }
            _save_label_config(_LABEL_CACHE)
            log.info("Created default label configuration file", path=str(LABEL_CONFIG_PATH))

    except Exception as e:
        log.error("Failed to load label configuration, using defaults", error=str(e))
        _LABEL_CACHE = {
            'act_type_to_label': DEFAULT_ACT_TYPE_TO_LABEL,
            'label_categories': DEFAULT_LABEL_CATEGORIES
        }

    return _LABEL_CACHE


def _save_label_config(config: Dict):
    """Salva la configurazione delle label nel file YAML."""
    try:
        LABEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LABEL_CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
        log.info("Label configuration saved", path=str(LABEL_CONFIG_PATH))
    except Exception as e:
        log.error("Failed to save label configuration", error=str(e))


def reload_label_config():
    """Ricarica la configurazione delle label dal file."""
    global _LABEL_CACHE
    _LABEL_CACHE = None
    return _load_label_config()


# Funzioni helper per accedere alla configurazione dinamica
def get_act_type_to_label_mapping() -> Dict[str, str]:
    """Restituisce la mappatura act_type -> label."""
    config = _load_label_config()
    return config['act_type_to_label']


def get_label_categories() -> Dict[str, List[str]]:
    """Restituisce le categorie di label."""
    config = _load_label_config()
    return config['label_categories']


def get_label_to_act_type_mapping() -> Dict[str, str]:
    """Restituisce la mappatura inversa label -> act_type."""
    config = _load_label_config()
    return {v: k for k, v in config['act_type_to_label'].items()}


def get_all_labels() -> List[str]:
    """Restituisce la lista di tutte le label disponibili."""
    config = _load_label_config()
    return sorted(set(config['act_type_to_label'].values()))


# Costanti per retrocompatibilità (caricano dinamicamente)
ACT_TYPE_TO_LABEL = get_act_type_to_label_mapping()
LABEL_CATEGORIES = get_label_categories()
LABEL_TO_ACT_TYPE = get_label_to_act_type_mapping()
ALL_LABELS = get_all_labels()


def act_type_to_label(act_type: str) -> str:
    """
    Converte un act_type interno in una label standardizzata.

    Args:
        act_type: Tipo di atto normativo interno (es. 'decreto_legislativo')

    Returns:
        Label standardizzata (es. 'D.LGS')
        Se non trovata nella mappatura, ritorna act_type in maiuscolo
    """
    if not act_type:
        return 'UNKNOWN'

    # Normalizza l'input
    act_type_normalized = act_type.lower().strip()

    # Carica la mappatura aggiornata
    mapping = get_act_type_to_label_mapping()

    # Cerca nella mappatura
    label = mapping.get(act_type_normalized)

    if label:
        return label

    # Fallback: usa act_type maiuscolo
    return act_type.upper().replace(' ', '_')


def label_to_act_type(label: str) -> str:
    """
    Converte una label standardizzata in un act_type interno.

    Args:
        label: Label standardizzata (es. 'D.LGS')

    Returns:
        Tipo di atto normativo interno (es. 'decreto_legislativo')
        Se non trovata, ritorna label in minuscolo
    """
    if not label:
        return 'unknown'

    # Normalizza l'input
    label_normalized = label.upper().strip()

    # Carica la mappatura inversa aggiornata
    mapping = get_label_to_act_type_mapping()

    # Cerca nella mappatura inversa
    act_type = mapping.get(label_normalized)

    if act_type:
        return act_type

    # Fallback: usa label minuscolo
    return label.lower().replace(' ', '_')


def get_label_category(label: str) -> str:
    """
    Restituisce la categoria di appartenenza di una label.

    Args:
        label: Label standardizzata

    Returns:
        Nome della categoria (es. 'Decreti', 'Leggi', ecc.)
        'Altro' se non trovata
    """
    label_normalized = label.upper().strip()

    # Carica le categorie aggiornate
    categories = get_label_categories()

    for category, labels in categories.items():
        if label_normalized in labels:
            return category

    return 'Altro'


def validate_label(label: str) -> bool:
    """
    Verifica se una label è valida.

    Args:
        label: Label da validare

    Returns:
        True se la label è valida, False altrimenti
    """
    all_labels = get_all_labels()
    return label.upper().strip() in all_labels


def get_labels_by_category(category: str) -> list[str]:
    """
    Restituisce tutte le label di una categoria.

    Args:
        category: Nome della categoria

    Returns:
        Lista di label appartenenti alla categoria
    """
    categories = get_label_categories()
    return categories.get(category, [])


def update_label_mapping(act_type: str, label: str, category: str = 'Altro'):
    """
    Aggiunge o aggiorna una mappatura act_type -> label.

    Args:
        act_type: Tipo di atto normativo interno
        label: Label standardizzata
        category: Categoria di appartenenza (default: 'Altro')
    """
    global _LABEL_CACHE

    config = _load_label_config()

    # Aggiorna la mappatura
    config['act_type_to_label'][act_type.lower().strip()] = label.upper().strip()

    # Aggiorna le categorie
    if category not in config['label_categories']:
        config['label_categories'][category] = []

    if label.upper().strip() not in config['label_categories'][category]:
        config['label_categories'][category].append(label.upper().strip())

    # Salva e ricarica
    _save_label_config(config)
    reload_label_config()


def remove_label_mapping(act_type: str):
    """
    Rimuove una mappatura act_type -> label.

    Args:
        act_type: Tipo di atto normativo da rimuovere
    """
    global _LABEL_CACHE

    config = _load_label_config()

    act_type_normalized = act_type.lower().strip()

    if act_type_normalized in config['act_type_to_label']:
        label = config['act_type_to_label'][act_type_normalized]
        del config['act_type_to_label'][act_type_normalized]

        # Rimuovi la label dalle categorie se non è più usata
        if label not in config['act_type_to_label'].values():
            for category, labels in config['label_categories'].items():
                if label in labels:
                    labels.remove(label)

        # Salva e ricarica
        _save_label_config(config)
        reload_label_config()
