"""
Specialized Legal Source Extraction Pipeline
===========================================

Pipeline specializzata dove ogni modello ha un ruolo specifico ottimizzato
per le sue capacità uniche, invece di fare tutti NER generico.

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
    semantic_embedding: Optional[np.ndarray] # Reverted to Optional[np.ndarray]

@dataclass
class ParsedNormative:
    """Componenti strutturati di una norma."""
    text: str # Added text field
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

    # Mappatura completa delle abbreviazioni normative italiane
    NORMATTIVA = {
        "d.lgs.": "decreto.legislativo",
        "dpr": "decreto.del.presidente.della.repubblica",
        "rd": "regio.decreto",
        "r.d.": "regio.decreto",
        "regio decreto": "regio.decreto",
        "d.p.r.": "decreto.del.presidente.della.repubblica",
        "decreto legge": "decreto.legge",
        "decreto legislativo": "decreto.legislativo",
        "decreto.legge": "decreto.legge",
        "decreto.legislativo": "decreto.legislativo",
        "dl": "decreto.legge",
        "dlgs": "decreto.legislativo",
        "l": "legge",
        "l.": "legge",
        "legge": "legge",
        "c.c.": "codice.civile",
        "c.p.": "codice.penale",
        "c.p.c": "codice.di.procedura.civile",
        "c.p.p.": "codice.di.procedura.penale",
        "c.c.p": "codice.dei.contratti.pubblici",
        "cad": "codice.dell.amministrazione.digitale",
        "cam": "codice.antimafia",
        "camb": "norme.in.materia.ambientale",
        "cap": "codice.delle.assicurazioni.private",
        "cbc": "codice.dei.beni.culturali.e.del.paesaggio",
        "cc": "codice.civile",
        "cce": "codice.delle.comunicazioni.elettroniche",
        "cci": "codice.della.crisi.d.impresa.e.dell.insolvenza",
        "ccp": "codice.dei.contratti.pubblici",
        "cdc": "codice.del.consumo",
        "cdpc": "codice.della.protezione.civile",
        "cds": "codice.della.strada",
        "cgco": "codice.di.giustizia.contabile",
        "cn": "codice.della.navigazione",
        "cnd": "codice.della.nautica.da.diporto",
        "cost": "costituzione",
        "cost.": "costituzione",
        "costituzione": "costituzione"
    }

    def __init__(self):
        log.info("Initializing EntityDetector with Italian_NER_XXL_v2")
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                "DeepMount00/Italian_NER_XXL_v2"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "DeepMount00/Italian_NER_XXL_v2"
            )
            log.info("EntityDetector initialized successfully")
        except Exception as e:
            log.warning("Failed to load Italian_NER_XXL_v2, using fallback", error=str(e))
            self.model = AutoModelForTokenClassification.from_pretrained(
                "Babelscape/wikineural-multilingual-ner"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Babelscape/wikineural-multilingual-ner"
            )

    def detect_candidates(self, text: str) -> List[TextSpan]:
        """
        Trova candidati che potrebbero essere riferimenti normativi.
        Focus su PRECISIONE della posizione, non sulla classificazione.
        """
        log.debug("Detecting legal reference candidates", text_length=len(text))

        # Tokenization con offset mapping per posizione precisa
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,
            max_length=512
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

        log.debug("Legal candidates detected",
                 raw_entities=len(raw_entities),
                 legal_candidates=len(cleaned_candidates))

        return cleaned_candidates

    def _is_potential_legal_reference(self, entity_text: str, full_text: str) -> bool:
        """
        Determina se un'entità potrebbe essere un riferimento normativo.
        Usa la mappatura NORMATTIVA completa + contesto specifico.
        """
        entity_lower = entity_text.lower().strip()

        # 1. Verifica diretta nella mappatura NORMATTIVA
        for abbrev in self.NORMATTIVA.keys():
            if abbrev.lower() in entity_lower:
                return True

        # 2. Pattern specifici per riferimenti normativi italiani
        legal_patterns = [
            # Tipi di atti con numeri
            r'\b(?:d\.?\s*lgs\.?|decreto\s+legislativo)\s+n?\.?\s*\d+',
            r'\b(?:d\.?\s*p\.?\s*r\.?|dpr)\s+n?\.?\s*\d+',
            r'\b(?:d\.?\s*l\.?|decreto\s+legge)\s+n?\.?\s*\d+',
            r'\b(?:l\.?|legge)\s+n?\.?\s*\d+',
            r'\b(?:r\.?\s*d\.?|regio\s+decreto)\s+n?\.?\s*\d+',

            # Codici (molto specifici)
            r'\bc\.?\s*c\.?\b',  # codice civile
            r'\bc\.?\s*p\.?\b',  # codice penale
            r'\bc\.?\s*p\.?\s*c\.?\b',  # codice procedura civile
            r'\bc\.?\s*p\.?\s*p\.?\b',  # codice procedura penale

            # Pattern numerici normativi
            r'\bn\.?\s*\d+(?:/\d{4})?(?:\s+del\s+\d{4})?',
            r'\b\d+/\d{4}\b',
            r'\b\d+\s+del\s+\d{4}\b',

            # Articoli e suddivisioni
            r'\bart\.?\s*\d+(?:\s*[,-]\s*(?:co\.?|comma)\s*\d+)?',
            r'\barticolo\s+\d+(?:\s*[,-]\s*comma\s*\d+)?',
            r'\bco\.?\s*\d+', r'\bcomma\s+\d+',
            r'\blett\.?\s*[a-z](?:\)|\b)', r'\blettera\s+[a-z]\b',

            # Allegati e appendici
            r'\ballegato\s+[a-z0-9]', r'\bappendice\s+[a-z0-9]',
            r'\bannesso\s+[a-z0-9]', r'\btabella\s+[a-z0-9]',
        ]

        for pattern in legal_patterns:
            if re.search(pattern, entity_lower, re.IGNORECASE):
                return True

        # 3. Verifica contesto semantico (finestra estesa)
        entity_pos = full_text.lower().find(entity_lower)
        if entity_pos != -1:
            # Finestra di contesto di 200 caratteri (100 prima, 100 dopo)
            start_context = max(0, entity_pos - 100)
            end_context = min(len(full_text), entity_pos + len(entity_text) + 100)
            context = full_text[start_context:end_context].lower()

            # Pattern contestuali molto specifici per documenti legali
            legal_context_indicators = [
                # Riferimenti normativi
                r'(?:secondo|ai\s+sensi|in\s+base\s+a|previsto|stabilito)\s+(?:da|dal|dell?|nel)',
                r'(?:disciplina|regola|prevede|stabilisce|dispone|determina)',
                r'(?:modificato|integrato|sostituito|abrogato)\s+(?:da|dal|con)',
                r'(?:entrat[oa]\s+in\s+vigore|vigente|applicabil[ei])',

                # Procedure legali
                r'(?:sentenza|decreto|ordinanza|delibera|risoluzione)\s+n?\.?\s*\d+',
                r'(?:tribunale|corte|tar|consiglio\s+di\s+stato)',
                r'(?:ricorso|appello|opposizione|istanza)',

                # Contesto amministrativo
                r'(?:gazzetta\s+ufficiale|g\.?\s*u\.?)',
                r'(?:ministero|dipartimento|agenzia|ente)',
                r'(?:autorizza\w*|approva\w*|emana\w*|promulga\w*)',

                # Riferimenti a articoli/commi nel contesto
                r'(?:di\s+cui\s+all?|previsto\s+dall?|secondo\s+l?)\s*art',
                r'(?:comma|co\.)\s+precedente',
                r'medesim[oa]\s+(?:articolo|comma|decreto|legge)',
            ]

            for pattern in legal_context_indicators:
                if re.search(pattern, context, re.IGNORECASE):
                    return True

        # 4. Pattern numerici isolati che potrebbero essere riferimenti
        # Solo se nel contesto ci sono indicatori legali
        if re.match(r'^\d+(?:/\d{4})?$', entity_text.strip()):
            # Cerca indicatori nelle vicinanze immediate (50 caratteri)
            if entity_pos != -1:
                immediate_context = full_text[max(0, entity_pos - 50):
                                           min(len(full_text), entity_pos + len(entity_text) + 50)]

                nearby_legal_words = [
                    'decreto', 'legge', 'articolo', 'art', 'comma', 'co',
                    'dlgs', 'dpr', 'norma', 'codice', 'costituzione'
                ]

                for word in nearby_legal_words:
                    if word in immediate_context.lower():
                        return True

        return False

    def _expand_reference_boundaries(self, entity: TextSpan, full_text: str) -> TextSpan:
        """
        Espande i confini dell'entità per catturare il riferimento normativo completo.
        Es: "231" → "decreto legislativo n. 231 del 2001"
        """
        start_char = entity.start_char
        end_char = entity.end_char

        # Espandi a sinistra per catturare tipo di atto
        window_start = max(0, start_char - 100)
        left_context = full_text[window_start:start_char]

        # Pattern da cercare a sinistra (più precisi)
        left_patterns = [
            r'(decreto\s+legislativo\s+n?\.?\s?)$',
            r'(d\.?\s*lgs\.?\s+n?\.?\s?)$',
            r'(legge\s+n?\.?\s?)$',
            r'(l\.?\s+n?\.?\s?)$',
            r'(decreto\s+del\s+presidente\s+della\s+repubblica\s+n?\.?\s?)$',
            r'(d\.?\s*p\.?\s*r\.?\s+n?\.?\s?)$',
            r'(articolo\s+)$', r'(art\.?\s*)$'
        ]

        for pattern in left_patterns:
            match = re.search(pattern, left_context, re.IGNORECASE)
            if match:
                # Trova l'inizio della parola per evitare di includere caratteri spuri
                word_start = match.start()
                # Assicurati che sia effettivamente l'inizio di una parola
                if word_start == 0 or left_context[word_start-1].isspace():
                    start_char = window_start + word_start
                    break

        # Espandi a destra per catturare data/anno
        window_end = min(len(full_text), end_char + 100)
        right_context = full_text[end_char:window_end]

        # Pattern da cercare a destra
        right_patterns = [
            r'^(\s+del\s+\d{1,2}[/-]\d{1,2}[/-]\d{4})',  # del 23/05/2001
            r'^(\s+del\s+\d{4})',                          # del 2001
            r'^(/\d{4})',                                  # /2001
            r'^(\s*,\s*articolo\s+\d+)',                  # , articolo 25
            r'^(\s*,\s*art\.?\s*\d+)',                    # , art. 25
            r'^(\s*,\s*comma\s+\d+)',                     # , comma 2
        ]

        for pattern in right_patterns:
            match = re.search(pattern, right_context, re.IGNORECASE)
            if match:
                end_char = end_char + match.end()
                break

        # Crea il nuovo span espanso e pulisci
        expanded_text = full_text[start_char:end_char].strip()

        # Rimuovi caratteri spuri all'inizio e alla fine
        expanded_text = re.sub(r'^[^\w\s]+', '', expanded_text)  # Rimuovi punctuation all'inizio
        expanded_text = re.sub(r'[^\w\s.,:;)]+$', '', expanded_text)  # Rimuovi punctuation alla fine (tranne alcuni)
        expanded_text = expanded_text.strip()

        # Aggiorna le posizioni dopo la pulizia
        if expanded_text != full_text[start_char:end_char].strip():
            # Trova la posizione reale del testo pulito
            clean_start = full_text.find(expanded_text, start_char)
            if clean_start != -1:
                start_char = clean_start
                end_char = clean_start + len(expanded_text)

        return TextSpan(
            text=expanded_text,
            start_char=start_char,
            end_char=end_char,
            initial_confidence=entity.initial_confidence,
            context_window=full_text[max(0, start_char-50):min(len(full_text), end_char+50)]
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

    def __init__(self):
        log.info("Initializing LegalClassifier with Italian-legal-bert")
        try:
            # Carica il modello per embeddings semantici (non per NER)
            self.model = AutoModel.from_pretrained("dlicari/distil-ita-legal-bert")
            self.tokenizer = AutoTokenizer.from_pretrained("dlicari/distil-ita-legal-bert")

            # Prototipi semantici per ogni tipo di atto normativo
            self._initialize_prototypes()
            log.info("LegalClassifier initialized successfully")
        except Exception as e:
            log.warning("Failed to load Italian-legal-bert for classification", error=str(e))
            self.model = None
            self.tokenizer = None

    def _initialize_prototypes(self):
        """Inizializza prototipi semantici per classificazione."""
        # Template di esempio per ogni tipo di atto
        # Questi verranno usati per generare embeddings di riferimento
        self.prototype_texts = {
            ActType.DECRETO_LEGISLATIVO: [
                "decreto legislativo numero del anno",
                "d.lgs. n. del",
                "dlgs",
                "decreto legislativo che disciplina"
            ],
            ActType.LEGGE: [
                "legge numero del anno",
                "l. n. del",
                "legge che stabilisce",
                "legge concernente"
            ],
            ActType.DPR: [
                "decreto del presidente della repubblica numero del",
                "d.p.r. n. del",
                "dpr",
                "decreto presidente repubblica"
            ],
            ActType.CODICE_CIVILE: [
                "codice civile",
                "c.c."
            ],
            ActType.CODICE_PENALE: [
                "codice penale",
                "c.p."
            ],
            ActType.CODICE_PROCEDURA_CIVILE: [
                "codice di procedura civile",
                "c.p.c."
            ],
            ActType.CODICE_PROCEDURA_PENALE: [
                "codice di procedura penale",
                "c.p.p."
            ],
            ActType.TESTO_UNICO: [
                "testo unico",
                "t.u.",
                "tuf"
            ],
            ActType.COSTITUZIONE: [
                "costituzione italiana",
                "cost.",
                "carta costituzionale",
                "principi costituzionali"
            ],
            ActType.CONVENTION: [
                "convenzione europea dei diritti dell'uomo",
                "c.e.d.u.",
                "protocollo addizionale"
            ],
            ActType.INSTITUTION: [
                "corte costituzionale",
                "corte europea dei diritti dell'uomo",
                "tribunale",
                "consiglio di stato"
            ]
        }

        # Genera embeddings per i prototipi (se il modello è disponibile)
        if self.model is not None:
            self.prototype_embeddings = {}
            for act_type, texts in self.prototype_texts.items():
                embeddings = []
                for text in texts:
                    embedding = self._get_embedding(text)
                    if embedding is not None:
                        embeddings.append(embedding)

                if embeddings:
                    # Media degli embeddings per creare il prototipo
                    self.prototype_embeddings[act_type] = np.mean(embeddings, axis=0)

    def _get_embedding(self, text: str) -> Optional[np.ndarray]: # Reverted return type
        """Genera embedding per un testo usando Italian-legal-bert."""
        if self.model is None:
            return None

        try:
            # Tokenization
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Usa il [CLS] token come rappresentazione della frase
                embedding = outputs.last_hidden_state[:, 0, :].numpy()[0] # Reverted to np.ndarray

            return embedding
        except Exception as e:
            log.warning("Failed to generate embedding", text=text[:50], error=str(e))
            return None

    def classify_legal_type(self, text_span: TextSpan, context: str) -> LegalClassification:
        """
        Classifica il tipo di atto normativo usando rule-based prioritario + semantica come supporto.
        """
        # STRATEGIA: Prima regole deterministiche (più accurate), poi semantica come conferma

        # Step 1: Prova classificazione rule-based (molto accurata per pattern chiari)
        rule_classification = self._classify_by_rules(text_span)

        # If rule-based confidence is high (e.g., >= 0.8), prioritize it strongly.
        # This threshold is chosen to protect ActType.CODICE (0.85) and other strong rules.
        if rule_classification.confidence >= 0.8:
            # If semantic is available and agrees, boost confidence.
            semantic_classification = None
            if self.model is not None:
                semantic_classification = self._classify_by_semantics(text_span, context)

            if semantic_classification and semantic_classification.act_type == rule_classification.act_type:
                combined_confidence = min(
                    (rule_classification.confidence + semantic_classification.confidence) / 2 + 0.1,
                    1.0
                )
                return LegalClassification(
                    span=text_span,
                    act_type=rule_classification.act_type,
                    confidence=combined_confidence,
                    semantic_embedding=semantic_classification.semantic_embedding
                )
            else:
                # If semantic is not available, or disagrees, or is less confident, stick with rule-based.
                return rule_classification
        
        # If rule-based confidence is low (< 0.8), then consider semantic classification more openly.
        else:
            semantic_classification = None
            if self.model is not None:
                semantic_classification = self._classify_by_semantics(text_span, context)

            if semantic_classification and semantic_classification.confidence > rule_classification.confidence:
                # If semantic is more confident, use it.
                return semantic_classification
            else:
                # Otherwise, stick with the (low confidence) rule-based.
                return rule_classification

    def _classify_by_semantics(self, text_span: TextSpan, context: str) -> Optional[LegalClassification]:
        """Classificazione semantica come metodo di supporto."""
        # Crea una finestra di contesto per la classificazione
        context_window = self._extract_context_window(text_span, context, window_size=200)

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

        confidence = min(best_similarity * 1.2, 1.0)  # Scale similarity to confidence

        return LegalClassification(
            span=text_span,
            act_type=best_act_type,
            confidence=confidence,
            semantic_embedding=context_embedding
        )

    def _classify_by_rules(self, text_span: TextSpan) -> LegalClassification:
        """Classificazione rule-based precisa con confidence variabili."""
        text_lower = text_span.text.lower()
        original_text = text_span.text # Keep original text for specific checks

        act_type = ActType.LEGGE  # Default
        confidence = 0.5  # Low confidence per default

        # Specific Code Classifications (most specific first)
        if re.search(r'\bc\.p\.c\.?\b', text_lower): # Made final dot optional
            act_type = ActType.CODICE_PROCEDURA_CIVILE
            confidence = 0.99
        elif re.search(r'\bc\.p\.p\.?\b', text_lower): # Made final dot optional
            act_type = ActType.CODICE_PROCEDURA_PENALE
            confidence = 0.99
        elif re.search(r'\bc\.c\.?\b', text_lower): # Made final dot optional
            act_type = ActType.CODICE_CIVILE
            confidence = 0.99
        elif re.search(r'\bc\.p\.?\b', text_lower) and not re.search(r'\b(?:c\.p\.c\.?|c\.p\.p\.?)\b', text_lower):
            # Match c.p. only if c.p.c. or c.p.p. are NOT present in the text (with optional dots)
            act_type = ActType.CODICE_PENALE
            confidence = 0.99
        elif re.search(r'\b(?:testo unico|t\.u\.|tuf)\b', text_lower):
            act_type = ActType.TESTO_UNICO
            confidence = 0.95
        # Treat generic "codice" as CODICE if not more specific
        elif 'codice' in text_lower:
            act_type = ActType.CODICE
            confidence = 0.85

        # Decreto Legislativo
        elif any(pattern in text_lower for pattern in ['decreto legislativo']):
            act_type = ActType.DECRETO_LEGISLATIVO
            confidence = 0.98
        elif any(pattern in text_lower for pattern in ['d.lgs.', 'd.lgs', 'dlgs']):
            act_type = ActType.DECRETO_LEGISLATIVO
            confidence = 0.95

        # DPR
        elif any(pattern in text_lower for pattern in ['decreto del presidente della repubblica']):
            act_type = ActType.DPR
            confidence = 0.98
        elif any(pattern in text_lower for pattern in ['d.p.r.', 'd.p.r', 'dpr']):
            act_type = ActType.DPR
            confidence = 0.95

        # Legge
        elif 'legge' in text_lower:
            act_type = ActType.LEGGE
            confidence = 0.90
        elif text_lower.strip() in ['l.', 'l'] or 'l.' in text_lower:
            act_type = ActType.LEGGE
            confidence = 0.75

        # Costituzione
        elif 'costituzione' in text_lower:
            act_type = ActType.COSTITUZIONE
            confidence = 0.98
        elif 'cost.' in text_lower:
            act_type = ActType.COSTITUZIONE
            confidence = 0.90
        
        # Conventions (e.g., CEDU)
        elif re.search(r'\b(?:c(?:venzione|edu)|protocollo)\b', text_lower):
            act_type = ActType.CONVENTION
            confidence = 0.90

        # Institutions (e.g., Corte Costituzionale, Agenzia delle Entrate) - these are not legal sources, so lower confidence or specific handling
        elif re.search(r'\b(?:corte|tribunale|consiglio di stato|agenzia delle entrate|banca d\'italia|consob|garante per la protezione dei dati personali|corte di giustizia dell\'ue)\b', text_lower, re.IGNORECASE):
            act_type = ActType.INSTITUTION
            confidence = 0.99 # High confidence for identification, but might be filtered later

        # Direttiva UE
        elif re.search(r'\b(?:direttiva\s*\(ue\)|direttiva\s*europea)\b', text_lower):
            act_type = ActType.DIRETTIVA_UE
            confidence = 0.99

        # Trattato
        elif re.search(r'\b(?:trattato\s+sul\s+funzionamento\s+dell\'unione\s+europea|tfue|trattato\s+dell\'unione\s+europea|tue)\b', text_lower):
            act_type = ActType.TRATTATO
            confidence = 0.99

        # Generic articles (low confidence, as they need context)
        elif text_lower.startswith('art') or text_lower.startswith('articolo'):
            confidence = 0.3

        log.debug("Rule-based classification",
                 text=text_span.text[:50],
                 act_type=act_type.value,
                 confidence=confidence,
                 patterns_matched=[p for p in ['decreto legislativo', 'd.lgs', 'legge', 'c.c.', 'costituzione'] if p in text_lower][:3])

        return LegalClassification(
            span=text_span,
            act_type=act_type,
            confidence=confidence,
            semantic_embedding=None  # No semantic embedding for rule-based classification
        )

    def _extract_context_window(self, text_span: TextSpan, full_text: str, window_size: int = 200) -> str:
        """Estrae una finestra di contesto attorno al text span."""
        start_context = max(0, text_span.start_char - window_size // 2)
        end_context = min(len(full_text), text_span.end_char + window_size // 2)

        return full_text[start_context:end_context]


class NormativeParser:
    """
    Stage 3: Estrae componenti strutturati da riferimenti normativi classificati.
    """
    def __init__(self):
        log.info("Initializing NormativeParser")
        # Regex patterns for common legal components
        self.patterns = {
            "act_number": r"(?:n\.?|numero)\s*(\d+)",
            "date": r"(?:del|in\s+data\s+del)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4})",
            "article": r"(?:art\.?|articolo)\s*(\d+)",
            "comma": r"(?:co\.?|comma)\s*(\d+)",
            "letter": r"(?:lett\.?|lettera)\s*([a-z])",
            "version": r"(?:versione|aggiornato\s+al)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4})",
            "annex": r"(?:allegato|annesso)\s*([a-zA-Z0-9]+)"
        }

    def parse(self, legal_classification: LegalClassification) -> ParsedNormative:
        """
        Parses the text of a legal classification to extract structured components.
        """
        text = legal_classification.span.text # Use original case for storing, lower for parsing
        parsed_data = {
            "text": text, # Pass the original text
            "act_type": legal_classification.act_type,
            "confidence": legal_classification.confidence,
            "start_char": legal_classification.span.start_char,
            "end_char": legal_classification.span.end_char
        }

        # Use lowercased text for regex matching
        text_lower = text.lower()

        # Specific pattern for EU Directives: Direttiva (UE) YYYY/NNNN
        eu_directive_pattern = r"direttiva\s*\(ue\)\s*(\d{4})/(\d{1,4})"
        match_eu_directive = re.search(eu_directive_pattern, text_lower)
        if match_eu_directive:
            parsed_data["date"] = match_eu_directive.group(1)
            parsed_data["act_number"] = match_eu_directive.group(2)
        else:
            # Extract act_number, article, comma, letter, version, annex using existing patterns
            for component, pattern in self.patterns.items():
                if component == "date": # Skip date for now, handle separately
                    continue
                match = re.search(pattern, text_lower)
                if match:
                    parsed_data[component] = match.group(1)

            # --- Date Extraction ---
            # Pattern 1: "del DD/MM/YYYY" or "del YYYY"
            date_pattern_1 = r"(?:del|in\s+data\s+del)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4})"
            match_date_1 = re.search(date_pattern_1, text_lower)
            if match_date_1:
                parsed_data["date"] = match_date_1.group(1)
            else:
                # Pattern 2: "n. XX/YYYY" or "XX/YYYY" (year after a slash)
                # Match the full pattern and then extract the year
                date_pattern_2_full = r"\d{1,4}/(\d{4})" # Captures the year after the slash
                match_date_2 = re.search(date_pattern_2_full, text_lower)
                if match_date_2:
                    parsed_data["date"] = match_date_2.group(1)
                else:
                    # Pattern 3: YYYY at the end of the string (e.g., "Legge 1998")
                    date_pattern_3 = r"(\d{4})$"
                    match_date_3 = re.search(date_pattern_3, text_lower)
                    if match_date_3:
                        parsed_data["date"] = match_date_3.group(1)
            # --- End Date Extraction ---

        # Determine if it's a complete reference (basic check)
        is_complete = parsed_data.get("act_number") and (parsed_data.get("date") or parsed_data.get("article"))
        parsed_data["is_complete_reference"] = is_complete

        return ParsedNormative(**parsed_data)

class ReferenceResolver:
    """
    Stage 4: Risolve riferimenti incompleti o ambigui.
    """
    def __init__(self):
        log.info("Initializing ReferenceResolver")
        # TODO: Implement a knowledge base or a more sophisticated lookup for resolution.
        # For now, a simple heuristic for common codes.
        self.code_abbreviations = {
            "c.c.": "civile",
            "c.p.": "penale",
            "c.p.c.": "procedura civile",
            "c.p.p.": "procedura penale"
        }

    def resolve(self, parsed_normative: ParsedNormative, full_text: str) -> ResolvedNormative:
        """
        Resolves incomplete references based on context.
        """
        resolved_data = asdict(parsed_normative)
        resolved_data["resolution_method"] = "direct"
        resolved_data["resolution_confidence"] = 1.0

        
        
        # TODO: Implement more advanced resolution strategies:
        # - Semantic search against a database of legal acts to find the most likely match.
        # - Handling relative references (e.g., "il precedente articolo" requires tracking previous entities).
        # - Using document-level context to disambiguate.

        return ResolvedNormative(**resolved_data)

class StructureBuilder:
    """
    Stage 5: Costruisce l'output finale strutturato.
    """
    def __init__(self):
        log.info("Initializing StructureBuilder")

    def build(self, resolved_normative: ResolvedNormative) -> Dict[str, Any]:
        """
        Builds the final structured output from a ResolvedNormative object.
        """
        # Handle INSTITUTION type: these are not legal sources for visualex
        if resolved_normative.act_type == ActType.INSTITUTION:
            log.debug(f"Filtering out institution: {resolved_normative.text}")
            return {}

        # Map ResolvedNormative fields to the schema.LegalSource format
        structured_output = {
            "source_type": resolved_normative.act_type.value if resolved_normative.act_type else None,
            "text": resolved_normative.text, # Use the text field from ParsedNormative
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
        # Filter out None values for cleaner output if desired by schema
        return {k: v for k, v in structured_output.items() if v is not None}


class LegalSourceExtractionPipeline:
    """
    Pipeline principale che coordina tutti gli stage specializzati.
    """

    def __init__(self):
        log.info("Initializing Specialized Legal Source Extraction Pipeline")

        # Stage 1: Entity Detection
        self.entity_detector = EntityDetector()

        # Stage 2: Legal Classification
        self.legal_classifier = LegalClassifier()

        # Stage 3: Normative Parser
        self.normative_parser = NormativeParser()

        # Stage 4: Reference Resolver
        self.reference_resolver = ReferenceResolver()

        # Stage 5: Structure Builder
        self.structure_builder = StructureBuilder()

        log.info("Specialized pipeline initialized successfully")

    def _is_spurious_entity(self, candidate: TextSpan) -> bool:
        """
        Filtra entità spurie che non sono riferimenti normativi validi.
        """
        text = candidate.text.strip()

        # Filtra entità troppo brevi (1-2 caratteri) a meno che non siano abbreviazioni note
        if len(text) <= 2:
            # Eccezioni per abbreviazioni valide molto brevi
            valid_short = ['l.', 'l', 'cc', 'cp']
            if text.lower() not in valid_short:
                return True

        # Filtra caratteri isolati che chiaramente non sono riferimenti
        if len(text) == 1:
            # Lettere singole isolate sono quasi sempre spurie
            if text.isalpha():
                return True

        # Filtra articoli determinativi isolati
        spurious_words = ["l'", "il", "la", "lo", "gli", "le", "del", "della", "dei", "delle"]
        if text.lower() in spurious_words:
            return True

        # Filtra se confidence della detection è troppo bassa
        if candidate.initial_confidence < 0.5:
            return True

        return False

    async def extract_legal_sources(self, text: str) -> List[Dict[str, Any]]:
        """
        Estrae fonti normative usando la pipeline specializzata.

        Args:
            text: Testo da analizzare

        Returns:
            Lista di fonti normative strutturate
        """
        log.info("Starting specialized legal source extraction", text_length=len(text))

        try:
            # Stage 1: Detect candidates
            candidates = self.entity_detector.detect_candidates(text)
            log.info("Stage 1 complete - Candidates detected", candidates_count=len(candidates))

            # Stage 2: Classify legal types + filter spurious entities
            classified_entities = []
            for candidate in candidates:
                # Skip overly short or spurious entities
                if self._is_spurious_entity(candidate):
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
                if structured_output: # Only append if not empty (i.e., not an INSTITUTION)
                    final_results.append(structured_output)
            log.info("Stage 5 complete - Final structured output built", final_results_count=len(final_results))

            log.info("Specialized extraction complete", results_count=len(final_results))
            return final_results

        except Exception as e:
            log.error("Error in specialized pipeline", error=str(e))
            return []