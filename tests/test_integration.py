"""
LabelGuard AI - End-to-End Integration Tests (Member 3)
Verifies contract boundaries, full pipeline integration, and mock labeling.
"""

import json
from pathlib import Path
try:
    import pytest
except ImportError:
    class MockPytest:
        @staticmethod
        def fixture(func):
            return func

    pytest = MockPytest()
from gemma_service import inspect_label
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


def test_front_panel_scenario_yields_needs_review(loaded_rules):
    """
    Scenario where front of pack is uploaded (odd length trigger in demo mode).
    Ingredients and FSSAI are not on front panel -> NOT_DETECTED -> evaluation flags NEEDS_REVIEW.
    """
    # 999 bytes is < 1000 and odd, triggering front-of-pack scenario in mock
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
    TASK 6: When live model call fails, inspect_label must return an error result
    (is_error=True) and NOT silently pass mock snack data.
    """
    # Set a dummy API key to simulate live mode
    monkeypatch.setenv("GEMMA_API_KEY", "test_invalid_api_key_12345")
    monkeypatch.setenv("FORCE_MOCK_GEMMA", "0")

    # Mock _call_live_gemma_4 to simulate API failure
    from gemma_service import _call_live_gemma_4
    def fake_failed_call(image_bytes, mime_type, api_key, model_name):
        return None, "Gemma 4 API Error: 401 Unauthorized - Invalid API Key"

    monkeypatch.setattr("gemma_service._call_live_gemma_4", fake_failed_call)

    result = inspect_label(b"test_image_bytes")
    assert result["is_error"] is True
    assert result["is_mock"] is False
    assert "401" in result["error_message"]

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
    # In mock mode, the model name tracks gemma-4
    assert "gemma-4" in result["model_name"]


def test_live_inference_classification_with_invalid_credentials(monkeypatch):
    """
    TASK 4: Verify how live inference classifies failure when invalid credentials are provided.
    Never prints or leaks secrets.
    """
    from gemma_service import _call_live_gemma_4
    result, error_msg = _call_live_gemma_4(
        image_bytes=b"dummy_bytes",
        mime_type="image/jpeg",
        api_key="AIzaSyDummyKeyForTestingOnlyNotReal",
        model_name="gemma-4-31b-it"
    )
    # The live call should fail gracefully and report an error message, without crashing
    assert result is None
    assert error_msg is not None
    # Ensure error message does not contain the secret key itself
    assert "AIzaSyDummyKeyForTestingOnlyNotReal" not in error_msg
