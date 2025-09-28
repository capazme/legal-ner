import re
from typing import List, Dict, Any, Optional
import structlog

log = structlog.get_logger()

class LegalSourceExtractor:
    def __init__(self):
        # Regex patterns for common Italian legal sources
        # These patterns are illustrative and would need extensive testing and refinement
        self.patterns = {
            "act_type_number_date": re.compile(
                r"(legge|decreto-legge|d\.l\.|decreto legislativo|d\.lgs\.|dpr|d.p.r.)\s+n\.?\s*(\d+)(?:\/(\d{4}))?\s*(?:del|in data del|in data)?\s*(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4})?",
                re.IGNORECASE
            ),
            "article": re.compile(
                r"art(?:icolo)?\.?\s*(\d+)(?:\s+comma\s+(\d+))?",
                re.IGNORECASE
            ),
            "version": re.compile(
                r"(?:versione|aggiornata al)\s+(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4})",
                re.IGNORECASE
            ),
            "annex": re.compile(
                r"allegato\s+([a-zA-Z0-9]+)",
                re.IGNORECASE
            )
        }

    def extract_sources(self, text: str) -> List[Dict[str, Any]]:
        log.info("Extracting legal sources", text_length=len(text))
        extracted_sources = []

        # Extract act type, number, and date
        for match in self.patterns["act_type_number_date"].finditer(text):
            source = {
                "act_type": match.group(1).strip() if match.group(1) else None,
                "act_number": match.group(2).strip() if match.group(2) else None,
                "date": match.group(4).strip() if match.group(4) else None, # Group 4 is the full date string
                "article": None,
                "version": None,
                "version_date": None,
                "annex": None
            }
            extracted_sources.append(source)
        
        # If no act type/number/date found, but other elements are present, create a base source
        if not extracted_sources:
            extracted_sources.append({
                "act_type": None,
                "date": None,
                "act_number": None,
                "article": None,
                "version": None,
                "version_date": None,
                "annex": None
            })

        # Extract articles and try to associate them with the last found act
        for match in self.patterns["article"].finditer(text):
            article_num = match.group(1).strip() if match.group(1) else None
            if extracted_sources and extracted_sources[-1]["act_type"] is not None:
                extracted_sources[-1]["article"] = article_num
            else:
                # If no act found, create a new source just for the article
                extracted_sources.append({
                    "act_type": None,
                    "date": None,
                    "act_number": None,
                    "article": article_num,
                    "version": None,
                    "version_date": None,
                    "annex": None
                })

        # Extract version dates
        for match in self.patterns["version"].finditer(text):
            version_date_str = match.group(1).strip() if match.group(1) else None
            if extracted_sources and extracted_sources[-1]["act_type"] is not None:
                extracted_sources[-1]["version_date"] = version_date_str
            else:
                extracted_sources.append({
                    "act_type": None,
                    "date": None,
                    "act_number": None,
                    "article": None,
                    "version": None,
                    "version_date": version_date_str,
                    "annex": None
                })

        # Extract annexes
        for match in self.patterns["annex"].finditer(text):
            annex_id = match.group(1).strip() if match.group(1) else None
            if extracted_sources and extracted_sources[-1]["act_type"] is not None:
                extracted_sources[-1]["annex"] = annex_id
            else:
                extracted_sources.append({
                    "act_type": None,
                    "date": None,
                    "act_number": None,
                    "article": None,
                    "version": None,
                    "version_date": None,
                    "annex": annex_id
                })

        # Filter out empty sources if multiple passes created them
        final_sources = [s for s in extracted_sources if any(s.values())]

        log.info("Legal source extraction complete", count=len(final_sources))
        return final_sources