"""
LabelGuard AI - Schemas Module (Member 1)
Defines shared Pydantic data schemas, enums, and data contracts.

Shared Status Contract:
- PRESENT: Evidence is clearly visible and extracted from the label image.
- NOT_DETECTED: Feature is not detected in the provided image (NOT proof of noncompliance).
- UNCLEAR: Visible text/graphics are blurry, occluded, or ambiguous.
- NEEDS_REVIEW: Flagged for human verification or requires other sides of the package.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class DetectionStatus(str, Enum):
    PRESENT = "PRESENT"
    NOT_DETECTED = "NOT_DETECTED"
    UNCLEAR = "UNCLEAR"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ExtractedField(BaseModel):
    """Represents a specific extracted field with visible evidence and status."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Name or identifier of the extracted field")
    status: DetectionStatus = Field(
        default=DetectionStatus.NOT_DETECTED,
        description="Detection status: PRESENT, NOT_DETECTED, UNCLEAR, or NEEDS_REVIEW"
    )
    extracted_text: Optional[str] = Field(
        default=None,
        description="Literal text extracted from the label image, if visible"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this extraction between 0.0 and 1.0"
    )
    visible_evidence: Optional[str] = Field(
        default=None,
        description="Verifiable description of visual element (e.g., location, appearance)"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any observations or contextual notes"
    )


class NutritionalItem(BaseModel):
    """Individual nutrient entry parsed from nutritional table if visible."""
    model_config = ConfigDict(extra="ignore")

    nutrient_name: str
    value_per_100g_or_ml: Optional[str] = None
    value_per_serve: Optional[str] = None
    unit: Optional[str] = None


class LabelExtractionResult(BaseModel):
    """
    Complete output schema for Gemma 4 multimodal label inspection.
    Purely captures visible evidence; does NOT evaluate compliance rules.
    """
    model_config = ConfigDict(extra="ignore")

    is_mock: bool = Field(
        default=False,
        description="Explicit flag indicating whether result is simulated/demo fallback"
    )
    is_error: bool = Field(
        default=False,
        description="Flag indicating if live model inference failed (prevents treating failure as extracted data)"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Details of inference failure if is_error is True"
    )
    model_name: str = Field(
        default="gemma-4",
        description="Model name used for extraction"
    )
    image_mime_type: str = Field(
        default="image/jpeg",
        description="MIME type of the processed label image"
    )
    product_name: ExtractedField = Field(
        default_factory=lambda: ExtractedField(name="product_name")
    )
    veg_nonveg_symbol: ExtractedField = Field(
        default_factory=lambda: ExtractedField(name="veg_nonveg_symbol")
    )
    fssai_license: ExtractedField = Field(
        default_factory=lambda: ExtractedField(name="fssai_license")
    )
    ingredients_list: ExtractedField = Field(
        default_factory=lambda: ExtractedField(name="ingredients_list")
    )
    nutritional_info: ExtractedField = Field(
        default_factory=lambda: ExtractedField(name="nutritional_info")
    )
    nutritional_items: List[NutritionalItem] = Field(
        default_factory=list,
        description="List of detected nutritional table entries"
    )
    net_quantity: ExtractedField = Field(
        default_factory=lambda: ExtractedField(name="net_quantity")
    )
    mfg_or_expiry_date: ExtractedField = Field(
        default_factory=lambda: ExtractedField(name="mfg_or_expiry_date")
    )
    manufacturer_details: ExtractedField = Field(
        default_factory=lambda: ExtractedField(name="manufacturer_details")
    )
    allergen_declaration: ExtractedField = Field(
        default_factory=lambda: ExtractedField(name="allergen_declaration")
    )
    raw_summary: str = Field(
        default="",
        description="Summary of visible packaging areas detected by Gemma 4"
    )
    disclaimer: str = Field(
        default="This is screening assistance, NOT legal certification. Missing visible items in a single photo do not constitute proof of regulatory noncompliance.",
        description="Legal and screening scope disclaimer"
    )


class RuleEvaluationResult(BaseModel):
    """
    Output schema for each rule evaluated by the deterministic rules engine.
    """
    model_config = ConfigDict(extra="ignore")

    rule_id: str
    rule_name: str
    category: str
    status: DetectionStatus
    citation: str = Field(description="Exact regulatory section citation")
    summary: str
    extracted_evidence: Optional[str] = None
    action_needed: Optional[str] = None
    is_noncompliance_proof: bool = Field(
        default=False,
        description="Always False: NOT_DETECTED is never proof of noncompliance"
    )
