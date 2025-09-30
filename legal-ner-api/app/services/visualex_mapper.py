"""
VisuaLex Mapper

Converte entità estratte dal Legal-NER nel formato richiesto dall'API VisuaLex.
"""

import re
import structlog
from typing import Dict, List, Optional, Any
from datetime import datetime

log = structlog.get_logger()

# Mapping tra label NER e act_type VisuaLex
# Basato su NORMATTIVA_SEARCH e FONTI_PRINCIPALI di VisuaLex
NER_TO_VISUALEX_MAPPING = {
    # Costituzione
    "costituzione": "costituzione",

    # Codici
    "codice_civile": "codice civile",
    "codice_penale": "codice penale",
    "codice_procedura_civile": "codice di procedura civile",
    "codice_procedura_penale": "codice di procedura penale",
    "codice_beni_culturali": "codice dei beni culturali e del paesaggio",
    "codice_strada": "codice della strada",
    "codice_processo_amministrativo": "codice del processo amministrativo",
    "codice_processo_tributario": "codice del processo tributario",
    "codice_consumo": "codice del consumo",
    "codice_amministrazione_digitale": "codice dell'amministrazione digitale",
    "codice_privacy": "codice in materia di protezione dei dati personali",
    "codice_comunicazioni_elettroniche": "codice delle comunicazioni elettroniche",
    "codice_navigazione": "codice della navigazione",
    "codice_nautica_diporto": "codice della nautica da diporto",
    "codice_proprieta_industriale": "codice della proprietà industriale",
    "codice_assicurazioni_private": "codice delle assicurazioni private",
    "codice_pari_opportunita": "codice delle pari opportunità",
    "codice_ordinamento_militare": "codice dell'ordinamento militare",
    "codice_turismo": "codice del turismo",
    "codice_antimafia": "codice antimafia",
    "codice_giustizia_contabile": "codice di giustizia contabile",
    "codice_terzo_settore": "codice del Terzo settore",
    "codice_protezione_civile": "codice della protezione civile",
    "codice_crisi_impresa": "codice della crisi d'impresa e dell'insolvenza",
    "codice_contratti_pubblici": "codice dei contratti pubblici",
    "codice_ambiente": "norme in materia ambientale",
    "codice_postale_telecomunicazioni": "codice postale e delle telecomunicazioni",

    # Preleggi e disposizioni attuative
    "preleggi": "preleggi",
    "disposizioni_attuazione_cc": "disposizioni per l'attuazione del Codice civile e disposizioni transitorie",
    "disposizioni_attuazione_cpc": "disposizioni per l'attuazione del Codice di procedura civile e disposizioni transitorie",

    # Decreti
    "decreto_legislativo": "decreto legislativo",
    "decreto_legge": "decreto legge",
    "decreto_presidente_repubblica": "d.p.r.",
    "dpr": "d.p.r.",
    "decreto_ministeriale": "decreto ministeriale",  # Generico ma accettato
    "dpcm": "dpcm",  # Decreto del Presidente del Consiglio dei Ministri
    "regio_decreto": "regio decreto",

    # Leggi
    "legge": "legge",
    "legge_costituzionale": "legge",  # Trattata come legge normale
    "legge_regionale": "legge regionale",

    # Testi Unici (generici)
    "testo_unico": "testo unico",

    # Normativa UE
    "regolamento_ue": "regolamento ue",
    "direttiva_ue": "direttiva ue",
    "trattato_ue": "tue",  # TUE = Trattato sull'Unione Europea
    "tfue": "tfue",  # Trattato sul Funzionamento dell'Unione Europea
    "carta_diritti_fondamentali_ue": "cdfue",  # Carta dei Diritti Fondamentali UE
    "convenzione_europea_diritti": "cdfue",
}

# Pattern regex per estrarre componenti
PATTERNS = {
    # Articolo: art. 123, artt. 45-50, articolo 5
    "article": r"(?:art\.?|artt\.?|articolo|articoli)\s*(\d+(?:-\d+)?(?:\s*(?:bis|ter|quater)?)?)",

    # Numero atto: n. 123, num. 456, 123/2000
    "act_number": r"(?:n\.?|num\.?|numero)?\s*(\d+(?:/\d+)?)",

    # Data: 23/12/2020, 23 dicembre 2020
    "date": r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})|(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4})",
}

MONTH_MAP = {
    "gennaio": "01", "febbraio": "02", "marzo": "03", "aprile": "04",
    "maggio": "05", "giugno": "06", "luglio": "07", "agosto": "08",
    "settembre": "09", "ottobre": "10", "novembre": "11", "dicembre": "12"
}


class VisuaLexMapper:
    """
    Converte entità estratte dal NER nel formato VisuaLex.
    """

    def __init__(self):
        log.info("Initializing VisuaLexMapper")

    def entity_to_visualex(self, entity: Dict[str, Any], context: str = "") -> Optional[Dict[str, Any]]:
        """
        Converte una singola entità nel formato NormaRequest di VisuaLex.

        Args:
            entity: Dizionario con l'entità estratta (deve avere: text, label, start_char, end_char)
            context: Testo completo del documento per estrarre contesto

        Returns:
            Dizionario nel formato NormaRequest o None se non convertibile
        """
        entity_text = entity.get("text", "")
        label = entity.get("label", "")

        log.info("Converting entity to VisuaLex format", label=label, text=entity_text[:50])

        # Mappa il label al tipo VisuaLex
        act_type = NER_TO_VISUALEX_MAPPING.get(label)

        if not act_type:
            log.warning("Entity label not supported by VisuaLex", label=label)
            return None

        # Estrai contesto allargato se disponibile
        if context and "start_char" in entity and "end_char" in entity:
            start = max(0, entity["start_char"] - 100)
            end = min(len(context), entity["end_char"] + 100)
            extended_context = context[start:end]
        else:
            extended_context = entity_text

        # Estrai componenti dal testo
        article = self._extract_article(extended_context)
        act_number = self._extract_act_number(extended_context)
        date = self._extract_date(extended_context)

        # L'articolo è REQUIRED da VisuaLex
        if not article:
            log.warning("No article found, cannot create VisuaLex request", text=entity_text)
            return None

        request = {
            "act_type": act_type,
            "article": article,
        }

        # Aggiungi campi opzionali se presenti
        if act_number:
            request["act_number"] = act_number

        if date:
            request["date"] = date

        log.info("Entity converted successfully",
                 act_type=act_type,
                 article=article,
                 act_number=act_number)

        return request

    def entities_to_visualex_batch(self, entities: List[Dict[str, Any]], context: str = "") -> List[Dict[str, Any]]:
        """
        Converte una lista di entità nel formato VisuaLex.

        Args:
            entities: Lista di entità estratte
            context: Testo completo del documento

        Returns:
            Lista di richieste VisuaLex
        """
        log.info("Converting batch of entities", count=len(entities))

        visualex_requests = []

        for entity in entities:
            request = self.entity_to_visualex(entity, context)
            if request:
                visualex_requests.append(request)

        log.info("Batch conversion completed",
                 original_count=len(entities),
                 converted_count=len(visualex_requests))

        return visualex_requests

    def _extract_article(self, text: str) -> Optional[str]:
        """Estrae il numero dell'articolo dal testo."""
        match = re.search(PATTERNS["article"], text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_act_number(self, text: str) -> Optional[str]:
        """Estrae il numero dell'atto dal testo."""
        match = re.search(PATTERNS["act_number"], text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Estrae la data dal testo e la converte in formato ISO (YYYY-MM-DD)."""
        match = re.search(PATTERNS["date"], text, re.IGNORECASE)
        if match:
            date_str = match.group(0)

            # Formato numerico: 23/12/2020 o 23-12-2020
            if "/" in date_str or "-" in date_str:
                parts = re.split(r"[\/\-]", date_str)
                if len(parts) == 3:
                    day, month, year = parts
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

            # Formato testuale: 23 dicembre 2020
            else:
                parts = date_str.split()
                if len(parts) == 3:
                    day, month_name, year = parts
                    month = MONTH_MAP.get(month_name.lower())
                    if month:
                        return f"{year}-{month}-{day.zfill(2)}"

        return None