from typing import List, Dict, Any, Set
import re
import structlog

log = structlog.get_logger()

class SemanticValidator:
    def __init__(self):
        log.info("Initializing Enhanced SemanticValidator")
        self._build_legal_knowledge_base()

    def _build_legal_knowledge_base(self):
        """Build comprehensive legal knowledge base for Italian legal system."""

        # Legal entities and organizations
        self.legal_organizations = {
            # Courts and judicial institutions
            "corte di cassazione", "cassazione", "sezioni unite", "corte costituzionale",
            "consiglio di stato", "tar", "tribunale", "corte di appello", "pretore",
            "giudice di pace", "corte d'assise", "tribunale per i minorenni",
            "tribunale amministrativo regionale", "consiglio di giustizia amministrativa",

            # Law enforcement and prosecution
            "procura della repubblica", "procura generale", "questura", "prefettura",
            "guardia di finanza", "carabinieri", "polizia di stato", "polizia municipale",
            "vigili del fuoco", "protezione civile",

            # Public institutions
            "ministero", "dicastero", "agenzia delle entrate", "inps", "inail",
            "banca d'italia", "consob", "autorità garante", "agcom", "antitrust",
            "garante della privacy", "autorità per l'energia", "ivass",

            # Administrative bodies
            "regione", "provincia", "comune", "ente locale", "ente pubblico",
            "azienda sanitaria", "asl", "università", "camera di commercio"
        }

        # Legal professionals and roles
        self.legal_persons = {
            # Judicial roles
            "giudice", "magistrato", "pubblico ministero", "pm", "procuratore",
            "presidente", "consigliere", "relatore", "estensore",

            # Legal professionals
            "avvocato", "legale", "consulente legale", "notaio", "commercialista",
            "revisore", "esperto", "perito", "consulente tecnico", "ctu", "ctp",

            # Administrative roles
            "dirigente", "funzionario", "segretario", "sindaco", "assessore",
            "consigliere comunale", "presidente della repubblica", "ministro"
        }

        # Legal document types and normative sources
        self.normative_types = {
            # Primary legislation
            "costituzione", "legge", "decreto legge", "decreto legislativo",
            "decreto del presidente della repubblica", "dpr", "regio decreto",

            # Codes
            "codice civile", "codice penale", "codice di procedura civile",
            "codice di procedura penale", "codice della strada", "codice del consumo",
            "codice dell'amministrazione digitale", "testo unico",

            # Regulations and administrative acts
            "regolamento", "ordinanza", "delibera", "decreto ministeriale",
            "circolare", "direttiva", "linee guida", "istruzione",

            # EU legislation
            "regolamento ue", "direttiva ue", "decisione ue", "raccomandazione ue",
            "trattato", "carta dei diritti fondamentali"
        }

        # Legal procedures and jurisprudence
        self.legal_procedures = {
            # Civil procedures
            "citazione", "comparsa", "ricorso", "appello", "cassazione",
            "opposizione", "revocazione", "riassunzione", "istanza",

            # Criminal procedures
            "denuncia", "querela", "imputazione", "rinvio a giudizio",
            "sentenza di condanna", "sentenza di assoluzione", "archiviazione",

            # Administrative procedures
            "ricorso amministrativo", "ricorso al tar", "appello al consiglio di stato",
            "ricorso straordinario", "autotutela", "silenzio assenso",

            # Jurisprudence types
            "sentenza", "ordinanza", "decreto", "pronuncia", "massima",
            "precedente", "giurisprudenza consolidata", "orientamento"
        }

        # Legal concepts and principles
        self.legal_concepts = {
            # Civil law concepts
            "contratto", "obbligazione", "responsabilità civile", "risarcimento",
            "danno", "inadempimento", "mora", "garanzia", "diritto reale",
            "proprietà", "usufrutto", "servitù", "ipoteca", "pegno",

            # Criminal law concepts
            "reato", "delitto", "contravvenzione", "dolo", "colpa",
            "legittima difesa", "stato di necessità", "prescrizione",

            # Administrative law concepts
            "atto amministrativo", "provvedimento", "concessione", "autorizzazione",
            "licenza", "nullità", "annullabilità", "eccesso di potere",

            # Constitutional concepts
            "diritto fondamentale", "libertà costituzionale", "principio",
            "riserva di legge", "competenza", "sussidiarietà"
        }

        # Legal entity patterns for fuzzy matching
        self.entity_patterns = {
            "ORG": self.legal_organizations,
            "PER": self.legal_persons,
            "NORMATIVA": self.normative_types,
            "GIURISPRUDENZA": self.legal_procedures,
            "CONCETTO_GIURIDICO": self.legal_concepts
        }

        # Build comprehensive lookup dictionary
        self.known_legal_terms = {}
        for label, terms in self.entity_patterns.items():
            for term in terms:
                self.known_legal_terms[term.lower()] = label

        # Add specific case variations and abbreviations
        self._add_abbreviations_and_variations()

    def _add_abbreviations_and_variations(self):
        """Add common abbreviations and variations."""

        abbreviations = {
            # Courts
            "cass.": "ORG", "cass": "ORG", "cc": "ORG", "ss.uu.": "ORG",
            "c. cost.": "ORG", "c.d.s.": "ORG", "t.a.r.": "ORG", "tar": "ORG",

            # Legal documents
            "c.c.": "NORMATIVA", "c.p.": "NORMATIVA", "c.p.c.": "NORMATIVA",
            "c.p.p.": "NORMATIVA", "cost.": "NORMATIVA",

            # Procedures
            "sent.": "GIURISPRUDENZA", "ord.": "GIURISPRUDENZA",
            "dec.": "GIURISPRUDENZA", "ric.": "GIURISPRUDENZA",

            # Professional titles
            "avv.": "PER", "dott.": "PER", "prof.": "PER", "ing.": "PER"
        }

        self.known_legal_terms.update(abbreviations)

        # Add title variations
        title_variations = {
            "presidente della repubblica": "PER",
            "presidente del consiglio": "PER", "premier": "PER",
            "ministro della giustizia": "PER", "guardasigilli": "PER",
            "procuratore generale": "PER", "pg": "PER",
            "presidente della corte": "PER", "presidente": "PER"
        }

        self.known_legal_terms.update(title_variations)

    def validate_entity(self, entity: Dict[str, Any]) -> bool:
        """Enhanced entity validation with fuzzy matching and context awareness."""
        log.debug("Validating entity", entity=entity)

        entity_text = entity["text"].strip()
        entity_text_lower = entity_text.lower()
        entity_label = entity["label"]

        # 1. Exact match validation
        if entity_text_lower in self.known_legal_terms:
            expected_label = self.known_legal_terms[entity_text_lower]
            if expected_label == entity_label:
                log.debug("Entity valid (exact match)", entity=entity)
                entity["validation_score"] = 1.0
                entity["validation_reason"] = "exact_match"
                return True
            else:
                log.debug("Entity label mismatch", entity=entity, expected=expected_label, actual=entity_label)
                entity["validation_score"] = 0.3
                entity["validation_reason"] = f"label_mismatch_expected_{expected_label}"
                return False

        # 2. Partial match validation
        validation_score = self._calculate_semantic_score(entity_text_lower, entity_label)
        entity["validation_score"] = validation_score

        if validation_score >= 0.7:
            entity["validation_reason"] = "high_semantic_score"
            return True
        elif validation_score >= 0.4:
            entity["validation_reason"] = "medium_semantic_score"
            return True  # Accept with caution
        else:
            # 3. Pattern-based validation for specific formats
            if self._validate_legal_patterns(entity_text, entity_label):
                entity["validation_score"] = 0.6
                entity["validation_reason"] = "pattern_match"
                return True

            entity["validation_reason"] = "low_semantic_score"
            log.debug("Entity validation failed", entity=entity, score=validation_score)
            return False

    def _calculate_semantic_score(self, text: str, label: str) -> float:
        """Calculate semantic similarity score for entity validation."""
        score = 0.0

        # Check for partial matches in appropriate category
        if label in self.entity_patterns:
            category_terms = self.entity_patterns[label]

            # Exact substring match
            for term in category_terms:
                if term in text or text in term:
                    score = max(score, 0.8)
                    break

            # Word overlap scoring
            text_words = set(text.split())
            for term in category_terms:
                term_words = set(term.split())
                if text_words and term_words:
                    overlap = len(text_words.intersection(term_words))
                    total_words = len(text_words.union(term_words))
                    word_score = overlap / total_words if total_words > 0 else 0
                    score = max(score, word_score * 0.6)

        # Bonus for legal-sounding patterns
        legal_indicators = [
            r'\b(?:art|articolo)\.\s*\d+', r'\b(?:comma|co)\.\s*\d+',
            r'\b(?:d\.lgs|decreto|legge)\b', r'\b(?:sentenza|ordinanza)\b',
            r'\b(?:tribunale|corte|cassazione)\b', r'\b(?:avv|dott|prof)\.\b'
        ]

        for pattern in legal_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score = max(score, 0.5)
                break

        return score

    def _validate_legal_patterns(self, text: str, label: str) -> bool:
        """Validate entities based on legal document patterns."""

        # Normative references
        if label == "NORMATIVA":
            normative_patterns = [
                r'\b(?:legge|decreto|d\.lgs|dpr)\s*(?:n\.?\s*)?\d+',
                r'\b(?:codice|testo\s+unico)\b',
                r'\bart\.\s*\d+',
                r'\bcostituzione\b'
            ]
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in normative_patterns)

        # Legal entities/organizations
        elif label == "ORG":
            org_patterns = [
                r'\b(?:tribunale|corte|cassazione|tar)\b',
                r'\b(?:ministero|agenzia|autorità)\b',
                r'\b(?:procura|questura|prefettura)\b'
            ]
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in org_patterns)

        # Legal persons
        elif label == "PER":
            person_patterns = [
                r'\b(?:avv|dott|prof|ing)\.\s*\w+',
                r'\b(?:giudice|magistrato|procuratore|presidente)\b',
                r'\b(?:ministro|sindaco|assessore)\b'
            ]
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in person_patterns)

        # Jurisprudence
        elif label == "GIURISPRUDENZA":
            jurisprudence_patterns = [
                r'\b(?:sentenza|ordinanza|decreto|pronuncia)\b',
                r'\b(?:ricorso|appello|cassazione)\b',
                r'\bn\.\s*\d+[\/\-]\d{2,4}'
            ]
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in jurisprudence_patterns)

        return False

    def validate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced entity validation with scoring and filtering."""
        log.info("Validating entities", input_count=len(entities))

        validated_entities = []
        validation_stats = {"exact_match": 0, "semantic_match": 0, "pattern_match": 0, "rejected": 0}

        for entity in entities:
            is_valid = self.validate_entity(entity)

            if is_valid:
                validated_entities.append(entity)
                reason = entity.get("validation_reason", "unknown")
                if "exact" in reason:
                    validation_stats["exact_match"] += 1
                elif "semantic" in reason:
                    validation_stats["semantic_match"] += 1
                elif "pattern" in reason:
                    validation_stats["pattern_match"] += 1
            else:
                validation_stats["rejected"] += 1
                log.debug("Entity rejected", entity=entity["text"], label=entity["label"],
                         score=entity.get("validation_score", 0), reason=entity.get("validation_reason"))

        log.info("Entity validation complete",
                output_count=len(validated_entities),
                validation_stats=validation_stats)

        return validated_entities

    def get_entity_suggestions(self, text: str) -> List[Dict[str, Any]]:
        """Suggest possible entity labels for unrecognized text."""
        text_lower = text.lower()
        suggestions = []

        for label, terms in self.entity_patterns.items():
            score = 0
            for term in terms:
                if term in text_lower or text_lower in term:
                    score = max(score, 0.8)
                elif any(word in term for word in text_lower.split()):
                    score = max(score, 0.4)

            if score > 0.3:
                suggestions.append({
                    "label": label,
                    "confidence": score,
                    "explanation": f"Text matches pattern for {label}"
                })

        return sorted(suggestions, key=lambda x: x["confidence"], reverse=True)[:3]
