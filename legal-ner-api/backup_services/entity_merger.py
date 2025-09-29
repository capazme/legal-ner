from typing import List, Dict, Any
import structlog

log = structlog.get_logger()

class EntityMerger:
    def __init__(self):
        log.info("Initializing EntityMerger")

    def merge_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges overlapping or duplicate entities based on textual overlap and labels."""
        log.info("Applying entity merging", input_entities_count=len(entities))

        if not entities:
            return []

        # Sort entities by start_char to simplify overlap detection
        sorted_entities = sorted(entities, key=lambda x: x["start_char"])
        
        merged_results = []

        for entity in sorted_entities:
            # Check if the current entity overlaps with any already merged entity
            merged = False
            for i, merged_entity in enumerate(merged_results):
                # Simple overlap check: if spans intersect
                if max(entity["start_char"], merged_entity["start_char"]) < min(entity["end_char"], merged_entity["end_char"]):
                    # Overlap detected
                    # For now, if labels are the same, merge them by taking the union of spans
                    # and averaging confidence. If labels are different, prioritize higher confidence.
                    if entity["label"] == merged_entity["label"]:
                        # Merge spans
                        new_start = min(entity["start_char"], merged_entity["start_char"])
                        new_end = max(entity["end_char"], merged_entity["end_char"])
                        
                        # Reconstruct text (this is a simplification, ideally would re-extract from original text)
                        # For now, just take the text of the longer entity or the one with higher confidence
                        new_text = entity["text"] if len(entity["text"]) > len(merged_entity["text"]) else merged_entity["text"]
                        if len(entity["text"]) == len(merged_entity["text"]) and entity["confidence"] > merged_entity["confidence"]:
                            new_text = entity["text"]

                        # Average confidence
                        new_confidence = (entity["confidence"] + merged_entity["confidence"]) / 2

                        merged_results[i] = {
                            "text": new_text,
                            "label": entity["label"],
                            "start_char": new_start,
                            "end_char": new_end,
                            "confidence": new_confidence,
                            "model": f"{merged_entity["model"]},{entity["model"]}" # Indicate models involved
                        }
                        merged = True
                        break
                    else:
                        # If labels are different, keep the one with higher confidence
                        if entity["confidence"] > merged_entity["confidence"]:
                            merged_results[i] = entity # Replace with higher confidence entity
                        merged = True
                        break
            
            if not merged:
                merged_results.append(entity)
        
        log.info("Entity merging complete", output_entities_count=len(merged_results))
        return merged_results
