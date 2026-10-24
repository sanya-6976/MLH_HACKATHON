"""
LabelGuard AI - End-to-End Integration Tests (Member 3)
Verifies contract boundaries, full pipeline integration, diagnostics, and mock labeling.
"""

import json
import os
from pathlib import Path
try:
    import pytest
except ImportError:
    class MockPytest:
        @staticmethod
        def fixture(func):
            return func

    pytest = MockPytest()

from gemma_service import (
    inspect_label,
    diagnose_endpoint,
    classify_api_error,
    _sanitize_message,
)
from rules_engine import evaluate_label
from schemas import DetectionStatus

RULES_FILE = Path(__file__).parent.parent / "rules" / "food_label_rules.json"


@pytest.fixture
def loaded_rules():
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_full_pipeline_contract_compliance(loaded_rules):
    """
    Verifies that the entire pipeline complies with exact function contracts:
    1. inspect_label(image_bytes: bytes, mime_type: str) -> dict
    2. evaluate_label(extracted: dict, rules: dict) -> list[dict]
    """
    dummy_image = b"RIFF....WEBPVP8 ... dummy image content"
    mime_type = "image/webp"

    # Step 1: Gemma 4 Inspection
    extracted = inspect_label(dummy_image, mime_type=mime_type)
    assert isinstance(extracted, dict), "inspect_label must return a dict"
    assert "fssai_license" in extracted
    assert "veg_nonveg_symbol" in extracted
    assert "is_mock" in extracted, "Result must explicitly track mock/demo status"

    # Step 2: Deterministic Rules Engine Evaluation
    evaluated = evaluate_label(extracted, loaded_rules)
    assert isinstance(evaluated, list), "evaluate_label must return a list[dict]"
    assert len(evaluated) == len(loaded_rules["rules"]), "All configured rules must be evaluated"

    # Verify every evaluation result adheres to contracted shared statuses
    valid_statuses = {s.value for s in DetectionStatus}
    for item in evaluated:
        assert isinstance(item, dict)
        assert item["status"] in valid_statuses
        assert item["is_noncompliance_proof"] is False
        assert "citation" in item
        assert "rule_id" in item


def test_mock_fallback_is_explicitly_labeled():
    """Verify that simulated/demo extractions are explicitly marked."""
    dummy_image = b"some_bytes"
    result = inspect_label(dummy_image)
    assert "is_mock" in result
    assert result["is_mock"] is True
    assert "model_name" in result
    assert "gemma-4" in result["model_name"]
    assert "DEMO" in result["model_name"] or "SIMULATED" in result["model_name"]


def test_front_panel_scenario_yields_needs_review(loaded_rules):
    """
    Scenario where front of pack is uploaded (odd length trigger in demo mode).
    Ingredients and FSSAI are not on front panel -> NOT_DETECTED -> evaluation flags NEEDS_REVIEW.
    """
    front_pack_bytes = b"x" * 999
    extracted = inspect_label(front_pack_bytes)

    # In front panel, FSSAI is not visible
    assert extracted["fssai_license"]["status"] == DetectionStatus.NOT_DETECTED.value

    # Rules engine should mark it as NEEDS_REVIEW, NOT noncompliance
    evaluated = evaluate_label(extracted, loaded_rules)
    fssai_eval = next(r for r in evaluated if r["rule_id"] == "FSSAI-2020-5-7")
    assert fssai_eval["status"] == DetectionStatus.NEEDS_REVIEW.value
    assert fssai_eval["is_noncompliance_proof"] is False


def test_model_failure_does_not_pass_mock_data(monkeypatch, loaded_rules):
    """
    TASK 6 & 8: When live model call fails, inspect_label must return an error result
    (is_error=True) and NOT silently pass mock snack data.
    """
    monkeypatch.setenv("GEMMA_API_KEY", "test_invalid_api_key_12345")
    monkeypatch.setenv("FORCE_MOCK_GEMMA", "0")

    def fake_failed_call(image_bytes, mime_type, api_key, model_name):
        return None, {
            "category": "AUTHENTICATION_ERROR",
            "status_code": 401,
            "detail": "Invalid API key provided",
            "suggested_fix": "Verify key in Google AI Studio"
        }

    monkeypatch.setattr("gemma_service._call_live_gemma_4", fake_failed_call)

    result = inspect_label(b"test_image_bytes")
    assert result["is_error"] is True
    assert result["is_mock"] is False
    assert "AUTHENTICATION_ERROR" in result["error_message"]

    # When passed to evaluate_label, evaluation is blocked, not evaluated on fake data
    evaluated = evaluate_label(result, loaded_rules)
    assert len(evaluated) == len(loaded_rules["rules"])
    for item in evaluated:
        assert item["status"] == DetectionStatus.NEEDS_REVIEW.value
        assert "evaluation blocked" in item["summary"].lower()


def test_environment_variables_read_consistently(monkeypatch):
    """
    TASK 3: Ensure GEMMA_API_KEY, GOOGLE_API_KEY, and GEMMA_MODEL_NAME are respected.
    """
    monkeypatch.setenv("GEMMA_MODEL_NAME", "gemma-4-26b-a4b-it")
    monkeypatch.setenv("FORCE_MOCK_GEMMA", "1")

    result = inspect_label(b"test_image_bytes")
    assert result["is_mock"] is True
    assert "gemma-4" in result["model_name"]


def test_live_inference_classification_with_invalid_credentials(monkeypatch):
    """
    TASK 4 & 5: Verify how live inference classifies failure when invalid credentials are provided.
    Never prints or leaks secrets.
    """
    from gemma_service import _call_live_gemma_4
    result, diagnostic = _call_live_gemma_4(
        image_bytes=b"dummy_bytes",
        mime_type="image/jpeg",
        api_key="AIzaSyDummyKeyForTestingOnlyNotReal",
        model_name="gemma-4-31b-it"
    )
    assert result is None
    assert diagnostic is not None
    assert "category" in diagnostic
    # Error message must never leak the raw API key
    assert "AIzaSyDummyKeyForTestingOnlyNotReal" not in diagnostic.get("detail", "")
    assert "AIzaSyDummyKeyForTestingOnlyNotReal" not in diagnostic.get("suggested_fix", "")


def test_classify_api_error_categories():
    """
    TASK 5: Verify structured error diagnostics for Authentication, Model Availability,
    Quota, Network, and Server error categories without exposing credentials.
    """
    fake_key = "secret_key_abcdef_123456"

    # 1. Authentication
    auth_err = classify_api_error(400, {"message": f"API_KEY_INVALID on {fake_key}", "status": "INVALID_ARGUMENT"}, api_key=fake_key)
    assert auth_err["category"] == "AUTHENTICATION_ERROR"
    assert fake_key not in auth_err["detail"]

    # 2. Model Not Found
    model_err = classify_api_error(404, {"message": "models/gemma-4-fake is not found", "status": "NOT_FOUND"})
    assert model_err["category"] == "MODEL_NOT_FOUND"

    # 3. Quota Exceeded
    quota_err = classify_api_error(429, {"message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"})
    assert quota_err["category"] == "QUOTA_EXCEEDED"

    # 4. Service Unavailable
    server_err = classify_api_error(503, {"message": "Backend service unavailable", "status": "UNAVAILABLE"})
    assert server_err["category"] == "SERVICE_UNAVAILABLE"


def test_diagnose_endpoint_unconfigured(monkeypatch):
    """
    TASK 5: Endpoint diagnosis returns UNCONFIGURED when no credentials exist.
    """
    monkeypatch.delenv("GEMMA_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    diag = diagnose_endpoint()
    assert diag["status"] == "UNCONFIGURED"
    assert diag["is_live_ready"] is False
    assert "NO_CREDENTIALS" in diag["category"]


def test_live_extraction_contains_actual_model_output_not_mock(monkeypatch, loaded_rules):
    """
    TASK 7: Verify that a successful model response contains genuine extracted
    data and does NOT contain preloaded mock wafer data.
    """
    monkeypatch.setenv("GEMMA_API_KEY", "valid_simulated_key")
    monkeypatch.setenv("FORCE_MOCK_GEMMA", "0")

    from schemas import ExtractedField, LabelExtractionResult

    custom_model_output = LabelExtractionResult(
        is_mock=False,
        is_error=False,
        model_name="gemma-4-31b-it",
        image_mime_type="image/jpeg",
        product_name=ExtractedField(
            name="product_name",
            status=DetectionStatus.PRESENT,
            extracted_text="Pure Himalayan Organic Honey 500g",
            confidence=0.99,
            visible_evidence="Gold metallic banner centered on front glass jar"
        ),
        veg_nonveg_symbol=ExtractedField(
            name="veg_nonveg_symbol",
            status=DetectionStatus.PRESENT,
            extracted_text="VEGETARIAN",
            confidence=0.98,
            visible_evidence="Green dot inside square below label neck"
        ),
        fssai_license=ExtractedField(
            name="fssai_license",
            status=DetectionStatus.PRESENT,
            extracted_text="11521018000123",
            confidence=0.97,
            visible_evidence="14-digit license number beside barcode"
        ),
        raw_summary="Live genuine inspection: Pure Himalayan Organic Honey."
    )

    def fake_successful_call(image_bytes, mime_type, api_key, model_name):
        return custom_model_output, None

    monkeypatch.setattr("gemma_service._call_live_gemma_4", fake_successful_call)

    result = inspect_label(b"real_honey_jar_image")
    assert result["is_mock"] is False
    assert result["is_error"] is False
    assert result["product_name"]["extracted_text"] == "Pure Himalayan Organic Honey 500g"
    assert result["fssai_license"]["extracted_text"] == "11521018000123"

    # Crucial assertion: Mock potato wafer text must NOT be present
    assert "Masala Crunch" not in str(result)
    assert "Potatoes (60%)" not in str(result)

    # Evaluate against rules
    evaluated = evaluate_label(result, loaded_rules)
    fssai_res = next(r for r in evaluated if r["rule_id"] == "FSSAI-2020-5-7")
    assert fssai_res["status"] == DetectionStatus.PRESENT.value
    assert "11521018000123" in fssai_res["extracted_evidence"]
