from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to be analyzed for named entities.")

class Entity(BaseModel):
    text: str = Field(..., description="The text of the recognized entity.")
    label: str = Field(..., description="The label of the entity (e.g., NORMATIVA, GIURISPRUDENZA).")
    start_char: int = Field(..., description="Start character index of the entity in the original text.")
    end_char: int = Field(..., description="End character index of the entity in the original text.")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the prediction.")
    model: str = Field(..., description="Model that generated this entity.")

class LegalSource(BaseModel):
    act_type: Optional[str] = None
    date: Optional[str] = None
    act_number: Optional[str] = None
    article: Optional[str] = None
    version: Optional[str] = None
    version_date: Optional[str] = None
    annex: Optional[str] = None

class NERResponse(BaseModel):
    entities: List[Entity] = Field(..., description="List of recognized entities.")
    legal_sources: List[LegalSource] = Field(..., description="List of extracted legal sources.")
    requires_review: bool = Field(..., description="Flag indicating if the prediction requires human review.")
    request_id: str = Field(..., description="Unique identifier for the request.")

class Annotation(BaseModel):
    entity_id: int
    is_correct: bool
    corrected_label: Optional[str] = None

class FeedbackRequest(BaseModel):
    annotations: List[Annotation]
