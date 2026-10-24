"""
LabelGuard AI - Schema Unit Tests (Member 1)
Validates Pydantic data schemas, status contracts, and serialization.
"""

try:
    import pytest
except ImportError:
    class MockPytest:
        class raises:
            def __init__(self, expected_exc):
                self.expected_exc = expected_exc

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {self.expected_exc} but no exception was raised")
                return issubclass(exc_type, self.expected_exc)

    pytest = MockPytest()
from pydantic import ValidationError
from schemas import (
    DetectionStatus,
    ExtractedField,
    NutritionalItem,
    LabelExtractionResult,
    RuleEvaluationResult,
)


def test_shared_statuses_contract():
    """Ensure all required statuses exist in DetectionStatus."""
    expected_statuses = {"PRESENT", "NOT_DETECTED", "UNCLEAR", "NEEDS_REVIEW"}
    actual_statuses = {s.value for s in DetectionStatus}
    assert expected_statuses == actual_statuses, "DetectionStatus must contain exactly the 4 contracted statuses"


def test_extracted_field_defaults():
    """ExtractedField defaults to NOT_DETECTED with 0 confidence."""
    field = ExtractedField(name="fssai_license")
    assert field.name == "fssai_license"
    assert field.status == DetectionStatus.NOT_DETECTED
    assert field.extracted_text is None
    assert field.confidence == 0.0


def test_extracted_field_valid():
    """Valid ExtractedField with PRESENT status and evidence."""
    field = ExtractedField(
        name="fssai_license",
        status=DetectionStatus.PRESENT,
        extracted_text="10019022009876",
        confidence=0.95,
        visible_evidence="14 digits printed below logo"
    )
    assert field.status == DetectionStatus.PRESENT
    assert field.extracted_text == "10019022009876"
    assert field.confidence == 0.95


def test_extracted_field_invalid_status():
    """Invalid status values should raise ValidationError."""
    with pytest.raises(ValidationError):
        ExtractedField(name="test", status="INVALID_STATUS")


def test_extracted_field_confidence_bounds():
    """Confidence must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        ExtractedField(name="test", confidence=1.5)

    with pytest.raises(ValidationError):
        ExtractedField(name="test", confidence=-0.1)


def test_label_extraction_result_defaults():
    """LabelExtractionResult default initialization adheres to schema."""
    result = LabelExtractionResult(is_mock=True, model_name="gemma-4")
    assert result.is_mock is True
    assert result.model_name == "gemma-4"
    assert result.fssai_license.status == DetectionStatus.NOT_DETECTED
    assert "NOT legal certification" in result.disclaimer

    # Test serialization to dict
    as_dict = result.model_dump()
    assert isinstance(as_dict, dict)
    assert as_dict["fssai_license"]["status"] == "NOT_DETECTED"


def test_rule_evaluation_result_schema():
    """RuleEvaluationResult validates correctly and preserves noncompliance safeguard."""
    eval_res = RuleEvaluationResult(
        rule_id="FSSAI-2020-5-7",
        rule_name="FSSAI Logo and License Number",
        category="Licensing",
        status=DetectionStatus.NEEDS_REVIEW,
        citation="FSSAI (Labelling and Display) Regulations, 2020 - Regulation 5(7)",
        summary="Absent in this photograph.",
        is_noncompliance_proof=False
    )
    assert eval_res.is_noncompliance_proof is False
    assert eval_res.status == DetectionStatus.NEEDS_REVIEW
