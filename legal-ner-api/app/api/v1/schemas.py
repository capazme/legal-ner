from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to be analyzed for named entities.")

class Entity(BaseModel):
    text: str = Field(..., description="The text of the recognized entity.")
    label: str = Field(..., description="The label of the entity (e.g., LEGGE, TRIBUNALE, N_SENTENZA).")
    start_char: int = Field(..., description="Start character index of the entity in the original text.")
    end_char: int = Field(..., description="End character index of the entity in the original text.")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the prediction.")
    model: str = Field(..., description="Model that generated this entity.")
    stage: Optional[str] = Field(None, description="Pipeline stage that extracted this entity.")
    structured_data: Optional[Dict[str, Any]] = Field(None, description="Structured components for legal entities.")
    validation_score: Optional[float] = Field(None, description="Semantic validation score.")
    semantic_correlations: Optional[List[Dict[str, Any]]] = Field(None, description="Semantic correlations with other entities.")
    final_quality_score: Optional[float] = Field(None, description="Final quality assessment score.")

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

# Enhanced feedback schemas for new system
class EnhancedFeedbackRequest(BaseModel):
    document_id: str = Field(..., description="ID of the document that was analyzed.")
    feedback_type: str = Field(..., description="Type of feedback: correct, incorrect, missing, wrong_label, wrong_boundary, partial")
    original_entity: Optional[Entity] = Field(None, description="Original entity if providing feedback on existing entity.")
    corrected_entity: Optional[Entity] = Field(None, description="Corrected entity if providing correction.")
    confidence_score: Optional[float] = Field(0.0, description="User confidence in their feedback.")
    notes: Optional[str] = Field(None, description="Additional notes or explanation.")

class FeedbackResponse(BaseModel):
    feedback_id: str = Field(..., description="Unique ID for this feedback.")
    status: str = Field(..., description="Status of feedback processing.")
    quality_impact: Dict[str, float] = Field(..., description="Impact on system quality metrics.")
    should_retrain: bool = Field(..., description="Whether this feedback triggers retraining.")
    golden_dataset_size: int = Field(..., description="Current size of golden dataset.")

class SystemStatsResponse(BaseModel):
    predictor_type: str = Field(..., description="Type of predictor system.")
    feedback_stats: Dict[str, Any] = Field(..., description="Feedback statistics.")
    golden_dataset_size: int = Field(..., description="Size of golden dataset.")
    system_accuracy: float = Field(..., description="Current system accuracy based on feedback.")
    status: str = Field(..., description="System operational status.")

class GoldenDatasetExportResponse(BaseModel):
    format: str = Field(..., description="Export format used.")
    data: str = Field(..., description="Exported dataset content.")
    entry_count: int = Field(..., description="Number of entries exported.")
    export_timestamp: str = Field(..., description="Timestamp of export.")
