"""
Specialized Legal Source Extraction Pipeline (CONFIGURABILE)
==========================================================

Pipeline specializzata dove ogni modello ha un ruolo specifico ottimizzato
per le sue capacità uniche, usando configurazione esterna YAML.

Architettura:
- Stage 1: EntityDetector (Italian_NER_XXL_v2) → Trova candidati potenziali
- Stage 2: LegalClassifier (Italian-legal-bert) → Classifica tipo normativo
- Stage 3: NormativeParser (Distil-legal-bert + rules) → Struttura componenti
- Stage 4: ReferenceResolver → Risolve riferimenti incompleti
- Stage 5: StructureBuilder → Output strutturato finale
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from sentence_transformers import SentenceTransformer
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.core.config_loader import get_pipeline_config, PipelineConfig

log = structlog.get_logger()

class ActType(Enum):
    """Tipi di atti normativi italiani supportati."""
    DECRETO_LEGISLATIVO = "decreto legislativo"
    LEGGE = "legge"
    DPR = "d.p.r."
    DECRETO_MINISTERIALE = "decreto ministeriale"
    CODICE = "codice"
    COSTITUZIONE = "costituzione"
    REGOLAMENTO = "regolamento ue"
    DIRETTIVA_UE = "direttiva ue"
    TRATTATO = "trattato"
    CODICE_CIVILE = "codice civile"
    CODICE_PENALE = "codice penale"
    CODICE_PROCEDURA_CIVILE = "codice di procedura civile"
    CODICE_PROCEDURA_PENALE = "codice di procedura penale"
    TESTO_UNICO = "codice"
    CONVENTION = "regolamento ue"
    INSTITUTION = "istituzione"

@dataclass
class TextSpan:
    """Span di testo con posizione e metadati."""
    text: str
    start_char: int
    end_char: int
    initial_confidence: float
    context_window: Optional[str] = None

@dataclass
class LegalClassification:
    """Risultato della classificazione legale."""
    span: TextSpan
    act_type: ActType
    confidence: float
    semantic_embedding: Optional[np.ndarray] = None

@dataclass
class ParsedNormative:
    """Componenti strutturati di una norma."""
    text: str
    act_type: ActType
    act_number: Optional[str] = None
    date: Optional[str] = None
    article: Optional[str] = None
    comma: Optional[str] = None
    letter: Optional[str] = None
    version: Optional[str] = None
    version_date: Optional[str] = None
    annex: Optional[str] = None
    is_complete_reference: bool = False
    confidence: float = 0.0
    start_char: int = 0
    end_char: int = 0

    def is_complete(self) -> bool:
        """Verifica se il riferimento è completo."""
        return self.act_type and self.act_number and (self.date or self.article)

@dataclass
class ResolvedNormative(ParsedNormative):
    """Norma con riferimenti risolti."""
    resolution_method: str = "direct"
    resolution_confidence: float = 1.0

class EntityDetector:
    """
    Stage 1: Usa Italian_NER_XXL_v2 SOLO per identificare span di testo
    che potrebbero essere riferimenti normativi.
    """

    def __init__(self, config: PipelineConfig):
        """Inizializza con configurazione esterna."""
        self.config = config
        log.info("Initializing EntityDetector with configuration")

        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                config.models.entity_detector_primary
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.models.entity_detector_primary
            )
            log.info("EntityDetector initialized successfully",
                    model=config.models.entity_detector_primary)
        except Exception as e:
            log.warning("Failed to load primary model, using fallback",
                       error=str(e),
                       fallback=config.models.entity_detector_fallback)
            self.model = AutoModelForTokenClassification.from_pretrained(
                config.models.entity_detector_fallback
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.models.entity_detector_fallback
            )

        # Carica mappatura NORMATTIVA dalla configurazione
        self.normattiva_mapping = self._build_flat_normattiva_mapping(config.normattiva_mapping)
        log.info("NORMATTIVA mapping loaded",
                abbreviations_count=len(self.normattiva_mapping))

    def _build_flat_normattiva_mapping(self, normattiva_config: Dict[str, List[str]]) -> Dict[str, str]:
        """Costruisce mappatura piatta da configurazione."""
        flat_mapping = {}
        for act_type, abbreviations in normattiva_config.items():
            normalized_type = act_type.replace("_", ".")
            for abbrev in abbreviations:
                flat_mapping[abbrev] = normalized_type
        return flat_mapping

    def _get_all_regex_patterns(self) -> List[str]:
        """Restituisce tutti i pattern regex in una lista piatta."""
        all_patterns = []
        for pattern_group in self.config.regex_patterns.values():
            all_patterns.extend(pattern_group)
        return all_patterns

    def _get_all_context_patterns(self) -> List[str]:
        """Restituisce tutti i pattern contestuali in una lista piatta."""
        all_patterns = []
        for pattern_group in self.config.context_patterns.values():
            all_patterns.extend(pattern_group)
        return all_patterns

    def detect_candidates(self, text: str) -> List[TextSpan]:
        """
        Trova candidati che potrebbero essere riferimenti normativi.
        Focus su PRECISIONE della posizione, non sulla classificazione.
        """
        if self.config.output.enable_debug_logging:
            log.debug("Detecting legal reference candidates", text_length=len(text))

        # Tokenization con offset mapping per posizione precisa
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,
            max_length=self.config.models.entity_detector_max_length
        )
        offset_mapping = inputs.pop("offset_mapping")[0]

        # Inferenza NER
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_token_class_ids = predictions.argmax(dim=-1)

        # Estrai entità con confidenze
        raw_entities = self._extract_entities_with_offsets(
            predicted_token_class_ids[0],
            predictions[0],
            offset_mapping,
            text
        )

        # Filtra solo candidati legali potenziali
        legal_candidates = []
        for entity in raw_entities:
            if self._is_potential_legal_reference(entity.text, text):
                # Espandi i confini per catturare il riferimento completo
                expanded = self._expand_reference_boundaries(entity, text)
                legal_candidates.append(expanded)

        # Rimuovi duplicati e sovrapposizioni
        cleaned_candidates = self._remove_overlaps(legal_candidates)

        if self.config.output.enable_debug_logging:
            log.debug("Legal candidates detected",
                     raw_entities=len(raw_entities),
                     legal_candidates=len(cleaned_candidates))

        return cleaned_candidates

    def _is_potential_legal_reference(self, entity_text: str, full_text: str) -> bool:
        """
        Determina se un'entità potrebbe essere un riferimento normativo.
        Usa la mappatura NORMATTIVA configurabile + contesto specifico.
        """
        entity_lower = entity_text.lower().strip()

        # 1. Verifica diretta nella mappatura NORMATTIVA
        for abbrev in self.normattiva_mapping.keys():
            if abbrev.lower() in entity_lower:
                return True

        # 2. Pattern specifici dalla configurazione
        all_patterns = self._get_all_regex_patterns()
        for pattern in all_patterns:
            if re.search(pattern, entity_lower, re.IGNORECASE):
                return True

        # 3. Verifica contesto semantico
        entity_pos = full_text.lower().find(entity_lower)
        if entity_pos != -1:
            # Finestra di contesto esteso
            start_context = max(0, entity_pos - self.config.context.extended_context)
            end_context = min(len(full_text), entity_pos + len(entity_text) + self.config.context.extended_context)
            context = full_text[start_context:end_context].lower()

            # Pattern contestuali dalla configurazione
            all_context_patterns = self._get_all_context_patterns()
            for pattern in all_context_patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    return True

        # 4. Pattern numerici isolati con contesto legale
        if re.match(r'^\d+(?:/\d{4})?$', entity_text.strip()):
            if entity_pos != -1:
                immediate_context = full_text[
                    max(0, entity_pos - self.config.context.immediate_context):
                    min(len(full_text), entity_pos + len(entity_text) + self.config.context.immediate_context)
                ]

                for word in self.config.legal_context_words:
                    if word in immediate_context.lower():
                        return True

        return False

    def _expand_reference_boundaries(self, entity: TextSpan, full_text: str) -> TextSpan:
        """
        Espande i confini dell'entità per catturare il riferimento normativo completo.
        Usa pattern configurabili per espansione.
        """
        start_char = entity.start_char
        end_char = entity.end_char

        # Espandi a sinistra per catturare tipo di atto
        window_start = max(0, start_char - self.config.context.left_window)
        left_context = full_text[window_start:start_char]

        # Pattern da cercare a sinistra dalla configurazione
        left_patterns = self.config.boundary_expansion["left_patterns"]
        for pattern in left_patterns:
            match = re.search(pattern, left_context, re.IGNORECASE)
            if match:
                word_start = match.start()
                if word_start == 0 or left_context[word_start-1].isspace():
                    start_char = window_start + word_start
                    break

        # Espandi a destra per catturare data/anno
        window_end = min(len(full_text), end_char + self.config.context.right_window)
        right_context = full_text[end_char:window_end]

        # Pattern da cercare a destra dalla configurazione
        right_patterns = self.config.boundary_expansion["right_patterns"]
        for pattern in right_patterns:
            match = re.search(pattern, right_context, re.IGNORECASE)
            if match:
                end_char = end_char + match.end()
                break

        # Crea il nuovo span espanso e pulisci
        expanded_text = full_text[start_char:end_char].strip()

        # Rimuovi caratteri spuri all'inizio e alla fine
        expanded_text = re.sub(r'^[^\w\s]+', '', expanded_text)
        expanded_text = re.sub(r'[^\w\s.,:;)]+$', '', expanded_text)
        expanded_text = expanded_text.strip()

        # Aggiorna le posizioni dopo la pulizia
        if expanded_text != full_text[start_char:end_char].strip():
            clean_start = full_text.find(expanded_text, start_char)
            if clean_start != -1:
                start_char = clean_start
                end_char = clean_start + len(expanded_text)

        return TextSpan(
            text=expanded_text,
            start_char=start_char,
            end_char=end_char,
            initial_confidence=entity.initial_confidence,
            context_window=full_text[
                max(0, start_char - self.config.context.context_window):
                min(len(full_text), end_char + self.config.context.context_window)
            ]
        )

    def _extract_entities_with_offsets(self, predicted_ids, predictions, offset_mapping, text):
        """Estrae entità con posizioni precise usando offset mapping."""
        entities = []
        current_entity = None

        for i, (token_id, token_probs) in enumerate(zip(predicted_ids, predictions)):
            if i >= len(offset_mapping):
                break

            start_offset, end_offset = offset_mapping[i]

            # Skip token speciali
            if start_offset == end_offset:
                continue

            # Converti ID token in label
            label = self.model.config.id2label.get(token_id.item(), 'O')
            confidence = torch.max(token_probs).item()

            # Gestione BIO encoding
            if label.startswith('B-'):
                # Salva entità precedente se esiste
                if current_entity:
                    entities.append(current_entity)

                # Inizia nuova entità
                current_entity = TextSpan(
                    text=text[start_offset:end_offset],
                    start_char=start_offset,
                    end_char=end_offset,
                    initial_confidence=confidence
                )
            elif label.startswith('I-') and current_entity:
                # Estendi entità corrente
                current_entity.text = text[current_entity.start_char:end_offset]
                current_entity.end_char = end_offset
                current_entity.initial_confidence = (current_entity.initial_confidence + confidence) / 2
            else:
                # O tag - fine entità
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Aggiungi ultima entità se esiste
        if current_entity:
            entities.append(current_entity)

        return entities

    def _remove_overlaps(self, candidates: List[TextSpan]) -> List[TextSpan]:
        """Rimuove candidati sovrapposti, mantenendo quello con confidence maggiore."""
        if not candidates:
            return []

        # Ordina per posizione
        sorted_candidates = sorted(candidates, key=lambda x: x.start_char)

        # Rimuovi sovrapposizioni
        cleaned = [sorted_candidates[0]]

        for candidate in sorted_candidates[1:]:
            last = cleaned[-1]

            # Controlla sovrapposizione
            if candidate.start_char < last.end_char:
                # Mantieni quello con confidence maggiore
                if candidate.initial_confidence > last.initial_confidence:
                    cleaned[-1] = candidate
            else:
                cleaned.append(candidate)

        return cleaned


class LegalClassifier:
    """
    Stage 2: Usa Italian-legal-bert come classificatore semantico per determinare
    il tipo specifico di riferimento normativo.
    """

    def __init__(self, config: PipelineConfig):
        """Inizializza con configurazione esterna."""
        self.config = config
        log.info("Initializing LegalClassifier with configuration")

        try:
            # Carica il modello per embeddings semantici
            self.model = AutoModel.from_pretrained(config.models.legal_classifier_primary)
            self.tokenizer = AutoTokenizer.from_pretrained(config.models.legal_classifier_primary)

            # Inizializza prototipi dalla configurazione
            self._initialize_prototypes()
            log.info("LegalClassifier initialized successfully")
        except Exception as e:
            log.warning("Failed to load Italian-legal-bert for classification", error=str(e))
            self.model = None
            self.tokenizer = None

    def _initialize_prototypes(self):
        """Inizializza prototipi semantici dalla configurazione."""
        if self.model is not None:
            self.prototype_embeddings = {}

            for act_type_str, texts in self.config.semantic_prototypes.items():
                try:
                    # Converti stringa in ActType enum
                    act_type = ActType(act_type_str.replace("_", " "))
                except ValueError:
                    # Gestisci casi speciali di mapping
                    act_type_mapping = {
                        "decreto_legislativo": ActType.DECRETO_LEGISLATIVO,
                        "legge": ActType.LEGGE,
                        "dpr": ActType.DPR,
                        "codice_civile": ActType.CODICE_CIVILE,
                        "codice_penale": ActType.CODICE_PENALE,
                        "codice_procedura_civile": ActType.CODICE_PROCEDURA_CIVILE,
                        "codice_procedura_penale": ActType.CODICE_PROCEDURA_PENALE,
                        "testo_unico": ActType.TESTO_UNICO,
                        "costituzione": ActType.COSTITUZIONE,
                        "convention": ActType.CONVENTION,
                        "institution": ActType.INSTITUTION
                    }
                    act_type = act_type_mapping.get(act_type_str)
                    if not act_type:
                        continue

                embeddings = []
                for text in texts:
                    embedding = self._get_embedding(text)
                    if embedding is not None:
                        embeddings.append(embedding)

                if embeddings:
                    # Media degli embeddings per creare il prototipo
                    self.prototype_embeddings[act_type] = np.mean(embeddings, axis=0)

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Genera embedding per un testo usando Italian-legal-bert."""
        if self.model is None:
            return None

        try:
            # Tokenization
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.models.legal_classifier_max_length,
                padding=True
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Usa il [CLS] token come rappresentazione della frase
                embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]

            return embedding
        except Exception as e:
            log.warning("Failed to generate embedding", text=text[:50], error=str(e))
            return None

    def classify_legal_type(self, text_span: TextSpan, context: str) -> LegalClassification:
        """
        Classifica il tipo di atto normativo usando rule-based prioritario + semantica come supporto.
        """
        # Step 1: Prova classificazione rule-based
        rule_classification = self._classify_by_rules(text_span)

        # Se confidence rule-based è alta, priorità alle regole
        if rule_classification.confidence >= self.config.confidence.rule_based_priority_threshold:
            # Se semantica disponibile e concorda, boost confidence
            semantic_classification = None
            if self.model is not None:
                semantic_classification = self._classify_by_semantics(text_span, context)

            if semantic_classification and semantic_classification.act_type == rule_classification.act_type:
                combined_confidence = min(
                    (rule_classification.confidence + semantic_classification.confidence) / 2 + self.config.confidence.semantic_boost_factor,
                    1.0
                )
                return LegalClassification(
                    span=text_span,
                    act_type=rule_classification.act_type,
                    confidence=combined_confidence,
                    semantic_embedding=semantic_classification.semantic_embedding
                )
            else:
                return rule_classification
        else:
            # Se confidence rule-based è bassa, considera semantica
            semantic_classification = None
            if self.model is not None:
                semantic_classification = self._classify_by_semantics(text_span, context)

            if semantic_classification and semantic_classification.confidence > rule_classification.confidence:
                return semantic_classification
            else:
                return rule_classification

    def _classify_by_semantics(self, text_span: TextSpan, context: str) -> Optional[LegalClassification]:
        """Classificazione semantica come metodo di supporto."""
        # Crea una finestra di contesto per la classificazione
        context_window = self._extract_context_window(text_span, context)

        # Genera embedding per il contesto
        context_embedding = self._get_embedding(context_window)
        if context_embedding is None:
            return None

        # Trova il prototipo più simile
        best_act_type = ActType.LEGGE  # Default
        best_similarity = -1.0

        for act_type, prototype_embedding in self.prototype_embeddings.items():
            similarity = cosine_similarity(
                context_embedding.reshape(1, -1),
                prototype_embedding.reshape(1, -1)
            )[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_act_type = act_type

        confidence = min(best_similarity * self.config.confidence.semantic_similarity_scale, 1.0)

        return LegalClassification(
            span=text_span,
            act_type=best_act_type,
            confidence=confidence,
            semantic_embedding=context_embedding
        )

    def _classify_by_rules(self, text_span: TextSpan) -> LegalClassification:
        """Classificazione rule-based precisa con confidence configurabili."""
        text_lower = text_span.text.lower()

        act_type = ActType.LEGGE  # Default
        confidence = self.config.confidence.default

        # Codici specifici (massima priorità)
        if re.search(r'\bc\.p\.c\.?\b', text_lower):
            act_type = ActType.CODICE_PROCEDURA_CIVILE
            confidence = self.config.confidence.specific_codes
        elif re.search(r'\bc\.p\.p\.?\b', text_lower):
            act_type = ActType.CODICE_PROCEDURA_PENALE
            confidence = self.config.confidence.specific_codes
        elif re.search(r'\bc\.c\.?\b', text_lower):
            act_type = ActType.CODICE_CIVILE
            confidence = self.config.confidence.specific_codes
        elif re.search(r'\bc\.p\.?\b', text_lower) and not re.search(r'\b(?:c\.p\.c\.?|c\.p\.p\.?)\b', text_lower):
            act_type = ActType.CODICE_PENALE
            confidence = self.config.confidence.specific_codes
        elif re.search(r'\b(?:testo unico|t\.u\.|tuf)\b', text_lower):
            act_type = ActType.TESTO_UNICO
            confidence = self.config.confidence.testo_unico
        elif 'codice' in text_lower:
            act_type = ActType.CODICE
            confidence = self.config.confidence.generic_codes

        # Decreto Legislativo
        elif any(pattern in text_lower for pattern in ['decreto legislativo']):
            act_type = ActType.DECRETO_LEGISLATIVO
            confidence = self.config.confidence.decreto_legislativo_full
        elif any(pattern in text_lower for pattern in ['d.lgs.', 'd.lgs', 'dlgs']):
            act_type = ActType.DECRETO_LEGISLATIVO
            confidence = self.config.confidence.decreto_legislativo_abbrev

        # DPR
        elif any(pattern in text_lower for pattern in ['decreto del presidente della repubblica']):
            act_type = ActType.DPR
            confidence = self.config.confidence.dpr_full
        elif any(pattern in text_lower for pattern in ['d.p.r.', 'd.p.r', 'dpr']):
            act_type = ActType.DPR
            confidence = self.config.confidence.dpr_abbrev

        # Legge
        elif 'legge' in text_lower:
            act_type = ActType.LEGGE
            confidence = self.config.confidence.legge_full
        elif text_lower.strip() in ['l.', 'l'] or 'l.' in text_lower:
            act_type = ActType.LEGGE
            confidence = self.config.confidence.legge_abbrev

        # Costituzione
        elif 'costituzione' in text_lower:
            act_type = ActType.COSTITUZIONE
            confidence = self.config.confidence.costituzione_full
        elif 'cost.' in text_lower:
            act_type = ActType.COSTITUZIONE
            confidence = self.config.confidence.costituzione_abbrev

        # Altri tipi
        elif re.search(r'\b(?:c(?:venzione|edu)|protocollo)\b', text_lower):
            act_type = ActType.CONVENTION
            confidence = self.config.confidence.convention

        elif re.search(r'\b(?:corte|tribunale|consiglio di stato|agenzia delle entrate)\b', text_lower, re.IGNORECASE):
            act_type = ActType.INSTITUTION
            confidence = self.config.confidence.institution

        elif re.search(r'\b(?:direttiva\s*\(ue\)|direttiva\s*europea)\b', text_lower):
            act_type = ActType.DIRETTIVA_UE
            confidence = self.config.confidence.direttiva_ue

        elif re.search(r'\b(?:trattato\s+sul\s+funzionamento\s+dell\'unione\s+europea|tfue)\b', text_lower):
            act_type = ActType.TRATTATO
            confidence = self.config.confidence.trattato

        # Articoli generici
        elif text_lower.startswith('art') or text_lower.startswith('articolo'):
            confidence = self.config.confidence.generic_article

        if self.config.output.log_pattern_matches:
            patterns_matched = [p for p in ['decreto legislativo', 'd.lgs', 'legge', 'c.c.', 'costituzione']
                              if p in text_lower][:self.config.output.max_logged_patterns]
            log.debug("Rule-based classification",
                     text=text_span.text[:50],
                     act_type=act_type.value,
                     confidence=confidence,
                     patterns_matched=patterns_matched)

        return LegalClassification(
            span=text_span,
            act_type=act_type,
            confidence=confidence,
            semantic_embedding=None
        )

    def _extract_context_window(self, text_span: TextSpan, full_text: str) -> str:
        """Estrae una finestra di contesto attorno al text span."""
        window_size = self.config.context.classification_context
        start_context = max(0, text_span.start_char - window_size // 2)
        end_context = min(len(full_text), text_span.end_char + window_size // 2)

        return full_text[start_context:end_context]


class NormativeParser:
    """
    Stage 3: Estrae componenti strutturati da riferimenti normativi classificati.
    """
    def __init__(self, config: PipelineConfig):
        """Inizializza con configurazione esterna."""
        self.config = config
        log.info("Initializing NormativeParser with configuration")

        # Pattern dalla configurazione
        self.patterns = config.parsing_patterns.copy()
        # Rimuovi pattern speciali che gestiremo separatamente
        self.patterns.pop("eu_directive", None)
        self.patterns.pop("date_patterns", None)

    def parse(self, legal_classification: LegalClassification) -> ParsedNormative:
        """
        Parses the text of a legal classification to extract structured components.
        """
        text = legal_classification.span.text
        parsed_data = {
            "text": text,
            "act_type": legal_classification.act_type,
            "confidence": legal_classification.confidence,
            "start_char": legal_classification.span.start_char,
            "end_char": legal_classification.span.end_char
        }

        text_lower = text.lower()

        # Pattern speciale per Direttive UE
        eu_directive_pattern = self.config.parsing_patterns["eu_directive"]
        match_eu_directive = re.search(eu_directive_pattern, text_lower)
        if match_eu_directive:
            parsed_data["date"] = match_eu_directive.group(1)
            parsed_data["act_number"] = match_eu_directive.group(2)
        else:
            # Estrai componenti usando pattern configurabili
            for component, pattern in self.patterns.items():
                if component == "date":  # Skip, gestito separatamente
                    continue
                match = re.search(pattern, text_lower)
                if match:
                    parsed_data[component] = match.group(1)

            # Estrazione date con priorità configurabile
            date_patterns = self.config.parsing_patterns["date_patterns"]

            # Pattern primario
            match_date_1 = re.search(date_patterns["primary"], text_lower)
            if match_date_1:
                parsed_data["date"] = match_date_1.group(1)
            else:
                # Pattern secondario
                match_date_2 = re.search(date_patterns["secondary"], text_lower)
                if match_date_2:
                    parsed_data["date"] = match_date_2.group(1)
                else:
                    # Pattern terziario
                    match_date_3 = re.search(date_patterns["tertiary"], text_lower)
                    if match_date_3:
                        parsed_data["date"] = match_date_3.group(1)

        # Determina se è un riferimento completo
        is_complete = parsed_data.get("act_number") and (parsed_data.get("date") or parsed_data.get("article"))
        parsed_data["is_complete_reference"] = is_complete

        return ParsedNormative(**parsed_data)


class ReferenceResolver:
    """
    Stage 4: Risolve riferimenti incompleti o ambigui.
    """
    def __init__(self, config: PipelineConfig):
        """Inizializza con configurazione esterna."""
        self.config = config
        log.info("Initializing ReferenceResolver with configuration")

    def resolve(self, parsed_normative: ParsedNormative, full_text: str) -> ResolvedNormative:
        """
        Resolves incomplete references based on context.
        """
        resolved_data = asdict(parsed_normative)
        resolved_data["resolution_method"] = "direct"
        resolved_data["resolution_confidence"] = 1.0

        # TODO: Implementazioni future configurabili
        return ResolvedNormative(**resolved_data)


class StructureBuilder:
    """
    Stage 5: Costruisce l'output finale strutturato.
    """
    def __init__(self, config: PipelineConfig):
        """Inizializza con configurazione esterna."""
        self.config = config
        log.info("Initializing StructureBuilder with configuration")

    def build(self, resolved_normative: ResolvedNormative) -> Dict[str, Any]:
        """
        Builds the final structured output from a ResolvedNormative object.
        """
        # Filtra istituzioni se configurato
        if (self.config.output.filter_institutions and
            resolved_normative.act_type == ActType.INSTITUTION):
            if self.config.output.enable_debug_logging:
                log.debug(f"Filtering out institution: {resolved_normative.text}")
            return {}

        # Map to schema format
        structured_output = {
            "source_type": resolved_normative.act_type.value if resolved_normative.act_type else None,
            "text": resolved_normative.text,
            "confidence": resolved_normative.confidence,
            "start_char": resolved_normative.start_char,
            "end_char": resolved_normative.end_char,
            "act_type": resolved_normative.act_type.value if resolved_normative.act_type else None,
            "date": resolved_normative.date,
            "act_number": resolved_normative.act_number,
            "article": resolved_normative.article,
            "version": resolved_normative.version,
            "version_date": resolved_normative.version_date,
            "annex": resolved_normative.annex
        }

        # Filtra valori null se configurato
        if self.config.output.filter_null_values:
            return {k: v for k, v in structured_output.items() if v is not None}

        return structured_output


class LegalSourceExtractionPipeline:
    """
    Pipeline principale che coordina tutti gli stage specializzati.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Inizializza pipeline con configurazione esterna."""
        log.info("Initializing Specialized Legal Source Extraction Pipeline")

        # Carica configurazione
        self.config = get_pipeline_config()
        log.info("Configuration loaded successfully")

        # Inizializza tutti gli stage con configurazione
        self.entity_detector = EntityDetector(self.config)
        self.legal_classifier = LegalClassifier(self.config)
        self.normative_parser = NormativeParser(self.config)
        self.reference_resolver = ReferenceResolver(self.config)
        self.structure_builder = StructureBuilder(self.config)

        log.info("Specialized pipeline initialized successfully")

    def _is_spurious_entity(self, candidate: TextSpan) -> bool:
        """
        Filtra entità spurie usando configurazione.
        """
        text = candidate.text.strip()

        # Filtra entità troppo brevi
        if len(text) <= self.config.spurious_filters.min_length:
            if text.lower() not in self.config.spurious_filters.valid_short_terms:
                return True

        # Filtra caratteri isolati
        if len(text) == 1 and text.isalpha() and self.config.spurious_filters.filter_single_alpha:
            return True

        # Filtra parole spurie configurate
        if text.lower() in self.config.spurious_filters.spurious_words:
            return True

        # Filtra usando pattern regex spurie
        for pattern in self.config.spurious_filters.spurious_patterns:
            if re.search(pattern, text.lower(), re.IGNORECASE):
                return True

        # Filtra se confidence troppo bassa
        if candidate.initial_confidence < self.config.spurious_filters.min_detection_confidence:
            return True

        return False

    async def extract_legal_sources(self, text: str) -> List[Dict[str, Any]]:
        """
        Estrae fonti normative usando la pipeline specializzata configurabile.
        """
        log.info("Starting specialized legal source extraction", text_length=len(text))

        try:
            # Stage 1: Detect candidates
            candidates = self.entity_detector.detect_candidates(text)
            log.info("Stage 1 complete - Candidates detected", candidates_count=len(candidates))

            # Stage 2: Classify legal types + filter spurious entities
            classified_entities = []
            for candidate in candidates:
                if self._is_spurious_entity(candidate):
                    if self.config.output.enable_debug_logging:
                        log.debug("Filtering spurious entity",
                                 text=candidate.text,
                                 length=len(candidate.text))
                    continue

                classification = self.legal_classifier.classify_legal_type(candidate, text)
                classified_entities.append(classification)

            log.info("Stage 2 complete - Legal types classified", classified_count=len(classified_entities))

            # Stage 3: Parse normative components
            parsed_normatives = []
            for classification in classified_entities:
                parsed = self.normative_parser.parse(classification)
                parsed_normatives.append(parsed)
            log.info("Stage 3 complete - Normative components parsed", parsed_count=len(parsed_normatives))

            # Stage 4: Resolve incomplete references
            resolved_normatives = []
            for parsed in parsed_normatives:
                resolved = self.reference_resolver.resolve(parsed, text)
                resolved_normatives.append(resolved)
            log.info("Stage 4 complete - Incomplete references resolved", resolved_count=len(resolved_normatives))

            # Stage 5: Build final structured output
            final_results = []
            for resolved in resolved_normatives:
                structured_output = self.structure_builder.build(resolved)
                if structured_output:  # Only append if not empty
                    final_results.append(structured_output)
            log.info("Stage 5 complete - Final structured output built", final_results_count=len(final_results))

            log.info("Specialized extraction complete", results_count=len(final_results))
            return final_results

        except Exception as e:
            log.error("Error in specialized pipeline", error=str(e))
            return []