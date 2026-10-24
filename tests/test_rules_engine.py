"""
LabelGuard AI - Deterministic Rules Engine Unit Tests (Member 2)
Verifies FSSAI 2020 deterministic evaluation and noncompliance safeguards.
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
from rules_engine import evaluate_label
from schemas import DetectionStatus

# Load test rules
RULES_FILE = Path(__file__).parent.parent / "rules" / "food_label_rules.json"


@pytest.fixture
def food_rules():
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_evaluate_label_not_detected_safeguard(food_rules):
    """
    CRITICAL RULE: NOT_DETECTED must never be treated as proof of noncompliance.
    When a field is not detected, status must be NEEDS_REVIEW with is_noncompliance_proof=False.
    """
    extracted = {
        "fssai_license": {
            "status": "NOT_DETECTED",
            "extracted_text": None,
            "visible_evidence": None
        }
    }

    results = evaluate_label(extracted, food_rules)
    assert isinstance(results, list)
    assert len(results) > 0

    fssai_res = next(r for r in results if r["rule_id"] == "FSSAI-2020-5-7")
    assert fssai_res["status"] == DetectionStatus.NEEDS_REVIEW.value
    assert fssai_res["is_noncompliance_proof"] is False
    assert "not visible" in fssai_res["summary"].lower() or "not detected" in fssai_res["summary"].lower()


def test_evaluate_label_valid_fssai_license(food_rules):
    """Valid 14-digit FSSAI number evaluates to PRESENT."""
    extracted = {
        "fssai_license": {
            "status": "PRESENT",
            "extracted_text": "Lic. No. 10019022009876",
            "visible_evidence": "14-digit number below FSSAI logo"
        }
    }

    results = evaluate_label(extracted, food_rules)
    fssai_res = next(r for r in results if r["rule_id"] == "FSSAI-2020-5-7")
    assert fssai_res["status"] == DetectionStatus.PRESENT.value
    assert fssai_res["action_needed"] is None


def test_evaluate_label_invalid_fssai_license_length(food_rules):
    """FSSAI license with wrong number of digits (e.g. 8 digits) flags NEEDS_REVIEW."""
    extracted = {
        "fssai_license": {
            "status": "PRESENT",
            "extracted_text": "Lic. No. 12345678",  # only 8 digits
            "visible_evidence": "License number below logo"
        }
    }

    results = evaluate_label(extracted, food_rules)
    fssai_res = next(r for r in results if r["rule_id"] == "FSSAI-2020-5-7")
    assert fssai_res["status"] == DetectionStatus.NEEDS_REVIEW.value
    assert "14" in fssai_res["summary"] or "14" in (fssai_res["action_needed"] or "")


def test_evaluate_label_unclear_status(food_rules):
    """Blurry or occluded items preserve UNCLEAR status and request clearer photo."""
    extracted = {
        "fssai_license": {
            "status": "UNCLEAR",
            "extracted_text": "1001???????",
            "visible_evidence": "Faded ink on seal"
        }
    }

    results = evaluate_label(extracted, food_rules)
    fssai_res = next(r for r in results if r["rule_id"] == "FSSAI-2020-5-7")
    assert fssai_res["status"] == DetectionStatus.UNCLEAR.value
    assert "focus" in fssai_res["action_needed"].lower() or "resolution" in fssai_res["summary"].lower()


def test_evaluate_veg_symbol_valid(food_rules):
    """Valid vegetarian symbol evaluates to PRESENT."""
    extracted = {
        "veg_nonveg_symbol": {
            "status": "PRESENT",
            "extracted_text": "VEGETARIAN",
            "visible_evidence": "Green circle inside square"
        }
    }
    results = evaluate_label(extracted, food_rules)
    veg_res = next(r for r in results if r["rule_id"] == "FSSAI-2020-5-4")
    assert veg_res["status"] == DetectionStatus.PRESENT.value


def test_evaluate_label_model_failure_safeguard(food_rules):
    """
    TASK 6 SAFEGUARD: When extraction has is_error=True, rule evaluation must be
    blocked and must NOT treat failure as genuine extracted data.
    """
    extracted_error = {
        "is_mock": False,
        "is_error": True,
        "error_message": "Gemma 4 API Error: HTTP 403 Forbidden",
        "model_name": "gemma-4-31b-it [LIVE CALL FAILED]"
    }
    results = evaluate_label(extracted_error, food_rules)
    assert len(results) == len(food_rules["rules"])
    for res in results:
        assert res["status"] == DetectionStatus.NEEDS_REVIEW.value
        assert res["is_noncompliance_proof"] is False
        assert "evaluation blocked" in res["summary"].lower()


def test_all_rules_not_detected_safeguard(food_rules):
    """
    TASK 7 SAFEGUARD: When every rule receives NOT_DETECTED, is_noncompliance_proof
    must be False across 100% of the rules.
    """
    empty_extracted = {
        rule["target_field"]: {"status": "NOT_DETECTED", "extracted_text": None, "visible_evidence": None}
        for rule in food_rules["rules"]
    }
    results = evaluate_label(empty_extracted, food_rules)
    assert len(results) == len(food_rules["rules"])
    for res in results:
        assert res["status"] == DetectionStatus.NEEDS_REVIEW.value
        assert res["is_noncompliance_proof"] is False


def test_verified_fssai_citations(food_rules):
    """
    TASK 7: Verify all rule citations match FSSAI (Labelling and Display) Regulations, 2020.
    """
    assert "FSSAI (Labelling and Display) Regulations, 2020" in food_rules["framework"]
    for rule in food_rules["rules"]:
        assert "FSSAI (Labelling and Display) Regulations, 2020" in rule["citation"]
        assert "Regulation 5" in rule["citation"]
