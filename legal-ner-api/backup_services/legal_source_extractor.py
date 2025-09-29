import re
from typing import List, Dict, Any, Optional, Tuple
import structlog
from datetime import datetime

log = structlog.get_logger()

class LegalSourceExtractor:
    def __init__(self):
        # Comprehensive regex patterns for Italian legal sources
        self.patterns = {
            # Enhanced normative source patterns
            "normative_full": re.compile(
                r"(legge|decreto[\s-]?legge|d\.?\s*l\.?|decreto\s+legislativo|d\.?\s*lgs\.?|d\.?\s*p\.?\s*r\.?|decreto\s+del\s+presidente\s+della\s+repubblica|regio\s+decreto|r\.?\s*d\.?|costituzione|codice\s+(?:civile|penale|procedura\s+civile|procedura\s+penale)|testo\s+unico|t\.?\s*u\.?)\s*(?:n\.?\s*|numero\s+)?(\d+)(?:[\/\-](\d{2,4}))?\s*(?:del|dell'|della|in\s+data\s+del?)?\s*(\d{1,2}[\s\-\/]\s*(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)[\s\-\/]\s*\d{2,4}|\d{1,2}[\s\-\/]\d{1,2}[\s\-\/]\d{2,4})?",
                re.IGNORECASE | re.MULTILINE
            ),
            # Article references with comma/letter specifications
            "article_full": re.compile(
                r"art(?:icolo|icoli|\.)\s*(\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?(?:\s*[-–]\s*\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)?)\s*(?:comma|commi|co\.)?\s*(\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?(?:\s*(?:e|ed|,)\s*\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)*)?(?:\s*lettera\s+([a-z]+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)(?:\s*(?:e|ed|,)\s*[a-z]+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)*)?",
                re.IGNORECASE
            ),
            # EU directives and regulations
            "eu_directive": re.compile(
                r"(direttiva|regolamento)\s+(?:\(UE\)|CE)?\s*(?:n\.?\s*)?(\d+)[\/\-](\d{2,4})(?:\s+del\s+\d{1,2}\s+\w+\s+\d{4})?",
                re.IGNORECASE
            ),
            # Constitutional articles
            "constitutional": re.compile(
                r"art(?:icolo|\.)\s*(\d+)\s+(?:della\s+)?costituzione",
                re.IGNORECASE
            ),
            # Annexes and attachments
            "annex": re.compile(
                r"allegato\s+([A-Z0-9]+(?:\s*[-–]\s*[A-Z0-9]+)?)",
                re.IGNORECASE
            ),
            # Jurisprudence references
            "jurisprudence": re.compile(
                r"(cassazione|corte\s+di\s+cassazione|sezioni\s+unite|corte\s+costituzionale|consiglio\s+di\s+stato|tar|tribunale)\s*(?:civile|penale|amministrativo)?\s*(?:sezione|sez\.)\s*([IVX]+|[\w\s]+)?\s*(?:sentenza|sent\.|ordinanza|ord\.)\s*(?:n\.?\s*)?(\d+)[\/\-](\d{2,4})",
                re.IGNORECASE
            ),
            # Date patterns
            "date_full": re.compile(
                r"(\d{1,2})[\s\-\/](gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)[\s\-\/](\d{2,4})|(\d{1,2})[\s\-\/](\d{1,2})[\s\-\/](\d{2,4})",
                re.IGNORECASE
            )
        }

        # Normalization mappings
        self.act_type_mappings = {
            "legge": "LEGGE",
            "decreto-legge": "DECRETO_LEGGE",
            "decreto legge": "DECRETO_LEGGE",
            "d.l.": "DECRETO_LEGGE",
            "dl": "DECRETO_LEGGE",
            "decreto legislativo": "DECRETO_LEGISLATIVO",
            "d.lgs.": "DECRETO_LEGISLATIVO",
            "dlgs": "DECRETO_LEGISLATIVO",
            "decreto del presidente della repubblica": "DPR",
            "d.p.r.": "DPR",
            "dpr": "DPR",
            "regio decreto": "REGIO_DECRETO",
            "r.d.": "REGIO_DECRETO",
            "costituzione": "COSTITUZIONE",
            "codice civile": "CODICE_CIVILE",
            "codice penale": "CODICE_PENALE",
            "testo unico": "TESTO_UNICO",
            "t.u.": "TESTO_UNICO"
        }

        self.month_mappings = {
            "gennaio": "01", "febbraio": "02", "marzo": "03", "aprile": "04",
            "maggio": "05", "giugno": "06", "luglio": "07", "agosto": "08",
            "settembre": "09", "ottobre": "10", "novembre": "11", "dicembre": "12"
        }

    def extract_sources(self, text: str) -> List[Dict[str, Any]]:
        """
        Enhanced legal source extraction with proper correlation and context awareness.
        """
        log.info("Extracting legal sources", text_length=len(text))

        # Store all matches with their positions for context correlation
        all_matches = []

        # Extract normative sources with positions
        for match in self.patterns["normative_full"].finditer(text):
            act_type_raw = match.group(1).strip().lower() if match.group(1) else None
            act_number = match.group(2).strip() if match.group(2) else None
            year = match.group(3).strip() if match.group(3) else None
            date_str = match.group(4).strip() if match.group(4) else None

            # Normalize act type
            act_type = self._normalize_act_type(act_type_raw) if act_type_raw else None

            # Parse date
            parsed_date = self._parse_date(date_str) if date_str else None

            source = {
                "type": "normative",
                "act_type": act_type,
                "act_number": act_number,
                "year": year,
                "date": parsed_date,
                "date_original": date_str,
                "start_pos": match.start(),
                "end_pos": match.end(),
                "match_text": match.group(0),
                "article": None,
                "comma": None,
                "letter": None,
                "annex": None,
                "confidence": 0.9
            }
            all_matches.append(source)

        # Extract EU directives/regulations
        for match in self.patterns["eu_directive"].finditer(text):
            source = {
                "type": "eu_normative",
                "act_type": match.group(1).strip().upper() if match.group(1) else None,
                "act_number": match.group(2).strip() if match.group(2) else None,
                "year": match.group(3).strip() if match.group(3) else None,
                "start_pos": match.start(),
                "end_pos": match.end(),
                "match_text": match.group(0),
                "confidence": 0.8
            }
            all_matches.append(source)

        # Extract constitutional references
        for match in self.patterns["constitutional"].finditer(text):
            source = {
                "type": "constitutional",
                "act_type": "COSTITUZIONE",
                "article": match.group(1).strip() if match.group(1) else None,
                "start_pos": match.start(),
                "end_pos": match.end(),
                "match_text": match.group(0),
                "confidence": 0.95
            }
            all_matches.append(source)

        # Extract jurisprudence references
        for match in self.patterns["jurisprudence"].finditer(text):
            source = {
                "type": "jurisprudence",
                "court": match.group(1).strip() if match.group(1) else None,
                "section": match.group(2).strip() if match.group(2) else None,
                "decision_number": match.group(3).strip() if match.group(3) else None,
                "year": match.group(4).strip() if match.group(4) else None,
                "start_pos": match.start(),
                "end_pos": match.end(),
                "match_text": match.group(0),
                "confidence": 0.85
            }
            all_matches.append(source)

        # Extract articles and correlate with nearby normative sources
        for match in self.patterns["article_full"].finditer(text):
            article_num = match.group(1).strip() if match.group(1) else None
            comma_num = match.group(2).strip() if match.group(2) else None
            letter = match.group(3).strip() if match.group(3) else None

            # Find the closest normative source
            closest_normative = self._find_closest_normative(match.start(), all_matches)

            if closest_normative and match.start() - closest_normative["end_pos"] < 200:
                # Associate with existing normative source
                closest_normative["article"] = article_num
                closest_normative["comma"] = comma_num
                closest_normative["letter"] = letter
                closest_normative["confidence"] = min(closest_normative["confidence"] + 0.1, 1.0)
            else:
                # Create standalone article reference
                source = {
                    "type": "article_standalone",
                    "article": article_num,
                    "comma": comma_num,
                    "letter": letter,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "match_text": match.group(0),
                    "confidence": 0.6
                }
                all_matches.append(source)

        # Extract annexes and correlate
        for match in self.patterns["annex"].finditer(text):
            annex_id = match.group(1).strip() if match.group(1) else None

            closest_normative = self._find_closest_normative(match.start(), all_matches)

            if closest_normative and match.start() - closest_normative["end_pos"] < 100:
                closest_normative["annex"] = annex_id
                closest_normative["confidence"] = min(closest_normative["confidence"] + 0.05, 1.0)
            else:
                source = {
                    "type": "annex_standalone",
                    "annex": annex_id,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "match_text": match.group(0),
                    "confidence": 0.7
                }
                all_matches.append(source)

        # Sort by position and remove duplicates/overlaps
        all_matches.sort(key=lambda x: x["start_pos"])
        final_sources = self._merge_overlapping_sources(all_matches)

        # Convert to output format
        result = []
        for source in final_sources:
            output_source = self._format_output_source(source)
            if output_source:
                result.append(output_source)

        log.info("Legal source extraction complete", count=len(result))
        return result

    def _normalize_act_type(self, act_type_raw: str) -> str:
        """Normalize act type to standard format."""
        act_type_clean = re.sub(r'\s+', ' ', act_type_raw.strip().lower())
        return self.act_type_mappings.get(act_type_clean, act_type_clean.upper())

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse Italian date formats to ISO format."""
        if not date_str:
            return None

        # Try to parse different date formats
        for match in self.patterns["date_full"].finditer(date_str):
            if match.group(1) and match.group(2) and match.group(3):
                # Italian format: dd mese yyyy
                day = match.group(1).zfill(2)
                month = self.month_mappings.get(match.group(2).lower())
                year = match.group(3)
                if len(year) == 2:
                    year = "20" + year if int(year) < 50 else "19" + year
                if month:
                    return f"{year}-{month}-{day}"
            elif match.group(4) and match.group(5) and match.group(6):
                # Numeric format: dd/mm/yyyy
                day = match.group(4).zfill(2)
                month = match.group(5).zfill(2)
                year = match.group(6)
                if len(year) == 2:
                    year = "20" + year if int(year) < 50 else "19" + year
                return f"{year}-{month}-{day}"

        return date_str

    def _find_closest_normative(self, position: int, matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the closest normative source before the given position."""
        closest = None
        min_distance = float('inf')

        for match in matches:
            if match.get("type") == "normative" and match["end_pos"] <= position:
                distance = position - match["end_pos"]
                if distance < min_distance:
                    min_distance = distance
                    closest = match

        return closest

    def _merge_overlapping_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping or duplicate sources."""
        if not sources:
            return []

        merged = []
        current = sources[0]

        for source in sources[1:]:
            # Check for overlap
            if current["end_pos"] > source["start_pos"]:
                # Merge if they're compatible
                if current.get("type") == source.get("type") or \
                   (current.get("type") == "normative" and source.get("type") == "article_standalone"):
                    # Merge sources
                    current["end_pos"] = max(current["end_pos"], source["end_pos"])
                    current["match_text"] = current["match_text"] + " " + source["match_text"]

                    # Copy additional attributes
                    for key in ["article", "comma", "letter", "annex"]:
                        if not current.get(key) and source.get(key):
                            current[key] = source[key]

                    current["confidence"] = max(current.get("confidence", 0), source.get("confidence", 0))
                else:
                    # Keep the one with higher confidence
                    if source.get("confidence", 0) > current.get("confidence", 0):
                        merged.append(current)
                        current = source
            else:
                merged.append(current)
                current = source

        merged.append(current)
        return merged

    def _format_output_source(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format source for output."""
        if source.get("type") == "normative":
            return {
                "act_type": source.get("act_type"),
                "act_number": source.get("act_number"),
                "year": source.get("year"),
                "date": source.get("date"),
                "article": source.get("article"),
                "comma": source.get("comma"),
                "letter": source.get("letter"),
                "annex": source.get("annex"),
                "confidence": source.get("confidence", 0.0),
                "source_type": "NORMATIVA"
            }
        elif source.get("type") == "eu_normative":
            return {
                "act_type": source.get("act_type"),
                "act_number": source.get("act_number"),
                "year": source.get("year"),
                "confidence": source.get("confidence", 0.0),
                "source_type": "NORMATIVA_EU"
            }
        elif source.get("type") == "constitutional":
            return {
                "act_type": source.get("act_type"),
                "article": source.get("article"),
                "confidence": source.get("confidence", 0.0),
                "source_type": "COSTITUZIONALE"
            }
        elif source.get("type") == "jurisprudence":
            return {
                "court": source.get("court"),
                "section": source.get("section"),
                "decision_number": source.get("decision_number"),
                "year": source.get("year"),
                "confidence": source.get("confidence", 0.0),
                "source_type": "GIURISPRUDENZA"
            }
        elif source.get("type") in ["article_standalone", "annex_standalone"]:
            return {
                "article": source.get("article"),
                "comma": source.get("comma"),
                "letter": source.get("letter"),
                "annex": source.get("annex"),
                "confidence": source.get("confidence", 0.0),
                "source_type": "RIFERIMENTO_PARZIALE"
            }

        return None