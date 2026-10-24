"""
LabelGuard AI - Deterministic Rules Engine (Member 2)
Evaluates extracted label data against configurable regulatory rules.

CRITICAL PRINCIPLES:
1. Screening assistance only — NOT legal certification.
2. NEVER treat NOT_DETECTED as proof of regulatory noncompliance.
   Package labels are 3D and multi-panel; a single camera angle only captures
   visible panels. NOT_DETECTED simply means evidence was not observed in the provided image.
3. MODEL FAILURE ISOLATION:
   If model extraction encountered a failure (is_error=True), rule evaluation
   is explicitly blocked rather than pretending unextracted data is genuine evidence.
4. Function Contract:
   evaluate_label(extracted: dict, rules: dict) -> list[dict]
"""

import re
from typing import Dict, Any, List
from schemas import DetectionStatus, RuleEvaluationResult


def _extract_field_data(extracted: dict, target_field: str) -> dict:
    """Helper to extract field dict or default structure from extracted data."""
    val = extracted.get(target_field)
    if isinstance(val, dict):
        return val
    return {"status": DetectionStatus.NOT_DETECTED.value, "extracted_text": None, "visible_evidence": None}


def evaluate_label(extracted: dict, rules: dict) -> List[dict]:
    """
    Evaluates extracted label information against deterministic rules.

    Args:
        extracted: Dictionary representing extracted label information, conforming
                   to LabelExtractionResult schema.
        rules: Dictionary containing the rules configuration loaded from JSON.

    Returns:
        List of dictionaries conforming to RuleEvaluationResult schema.
    """
    rule_list = rules.get("rules", [])
    results: List[dict] = []

    # Safeguard: If the extraction encountered a model failure, do NOT treat
    # failure as genuine label data. Explicitly report evaluation as blocked.
    if extracted.get("is_error"):
        error_msg = extracted.get("error_message", "Model extraction failed")
        for rule in rule_list:
            res = RuleEvaluationResult(
                rule_id=rule.get("rule_id", "UNKNOWN"),
                rule_name=rule.get("rule_name", "Unknown Rule"),
                category=rule.get("category", "General"),
                status=DetectionStatus.NEEDS_REVIEW,
                citation=rule.get("citation", "FSSAI Regulations, 2020"),
                summary=f"Rule evaluation blocked: Model extraction failed ({error_msg}). Cannot evaluate compliance on unextracted data.",
                extracted_evidence=None,
                action_needed="Resolve model credentials or network connection, or switch to Demo mode.",
                is_noncompliance_proof=False
            )
            results.append(res.model_dump())
        return results

    for rule in rule_list:
        rule_id = rule.get("rule_id", "UNKNOWN_RULE")
        rule_name = rule.get("rule_name", "Unnamed Rule")
        category = rule.get("category", "General")
        citation = rule.get("citation", "Unspecified Regulation")
        target_field = rule.get("target_field", "")
        validation_type = rule.get("validation_type", "default")

        field_data = _extract_field_data(extracted, target_field)
        field_status = field_data.get("status", DetectionStatus.NOT_DETECTED.value)
        extracted_text = field_data.get("extracted_text") or ""
        visible_evidence = field_data.get("visible_evidence") or ""

        # Normalize field_status to string
        if isinstance(field_status, DetectionStatus):
            field_status = field_status.value

        evidence_str = f"Text: '{extracted_text}'" if extracted_text else ""
        if visible_evidence:
            evidence_str = f"{evidence_str} | Visual: {visible_evidence}".strip(" |")

        # Handle NOT_DETECTED:
        # Crucial FSSAI / Hackathon Principle: Absence in single photo != noncompliance!
        if field_status == DetectionStatus.NOT_DETECTED.value:
            res = RuleEvaluationResult(
                rule_id=rule_id,
                rule_name=rule_name,
                category=category,
                status=DetectionStatus.NEEDS_REVIEW,
                citation=citation,
                summary=rule.get(
                    "on_not_detected",
                    "Not visible on this label view. Please check remaining package panels."
                ),
                extracted_evidence=None,
                action_needed="Inspect remaining package sides. A single photo does not prove absence.",
                is_noncompliance_proof=False
            )
            results.append(res.model_dump())
            continue

        # Handle UNCLEAR:
        if field_status == DetectionStatus.UNCLEAR.value:
            res = RuleEvaluationResult(
                rule_id=rule_id,
                rule_name=rule_name,
                category=category,
                status=DetectionStatus.UNCLEAR,
                citation=citation,
                summary=rule.get(
                    "on_unclear",
                    "Image section is blurred, occluded, or low resolution."
                ),
                extracted_evidence=evidence_str or "Section marked unclear by extractor",
                action_needed="Re-take photograph with better illumination, reduced glare, and sharp focus.",
                is_noncompliance_proof=False
            )
            results.append(res.model_dump())
            continue

        # Handle NEEDS_REVIEW directly from extractor:
        if field_status == DetectionStatus.NEEDS_REVIEW.value:
            res = RuleEvaluationResult(
                rule_id=rule_id,
                rule_name=rule_name,
                category=category,
                status=DetectionStatus.NEEDS_REVIEW,
                citation=citation,
                summary="Extractor flagged this element for human verification.",
                extracted_evidence=evidence_str,
                action_needed="Verify details directly on physical packaging.",
                is_noncompliance_proof=False
            )
            results.append(res.model_dump())
            continue

        # Handle PRESENT: Run deterministic validation based on rule type
        status = DetectionStatus.PRESENT
        summary = rule.get("on_present_valid", "Requirement satisfied based on visible evidence.")
        action_needed = None

        if validation_type == "regex_license_14_digit":
            pattern = rule.get("validation_pattern", r"\b\d{14}\b")
            clean_text = re.sub(r"[\s\-]", "", extracted_text)
            has_14_digits = bool(re.search(r"\b\d{14}\b", extracted_text) or re.search(r"\d{14}", clean_text))
            if not has_14_digits:
                status = DetectionStatus.NEEDS_REVIEW
                summary = rule.get(
                    "on_present_invalid",
                    "FSSAI license number visible but did not match expected 14-digit format."
                )
                action_needed = "Verify that the FSSAI license consists of exactly 14 numeric digits."

        elif validation_type == "symbol_presence":
            allowed = rule.get("allowed_values", ["VEGETARIAN", "NON_VEGETARIAN", "VEG", "NON_VEG"])
            val_upper = extracted_text.upper().replace("-", "_").replace(" ", "_")
            if not any(token in val_upper for token in allowed) and not ("VEG" in val_upper or "GREEN" in val_upper or "BROWN" in val_upper):
                status = DetectionStatus.NEEDS_REVIEW
                summary = rule.get("on_present_invalid", "Symbol detected but dietary classification is ambiguous.")
                action_needed = "Confirm whether symbol is Green (Veg) or Brown (Non-Veg)."

        elif validation_type == "text_minimum_length":
            min_len = rule.get("min_length", 2)
            if len(extracted_text.strip()) < min_len:
                status = DetectionStatus.NEEDS_REVIEW
                summary = rule.get("on_present_invalid", "Text field detected but appears incomplete.")
                action_needed = "Confirm full disclosure on packaging."

        elif validation_type == "regex_units":
            pattern = rule.get("validation_pattern", r"\b\d+(\.\d+)?\s*(g|kg|ml|l|gm|grams)\b")
            if not re.search(pattern, extracted_text, re.IGNORECASE):
                status = DetectionStatus.NEEDS_REVIEW
                summary = rule.get("on_present_invalid", "Quantity detected but standard metric unit could not be confirmed.")
                action_needed = "Check whether weight/volume is declared in standard units (g, kg, ml, l)."

        elif validation_type == "nutritional_panel":
            # Check mandatory nutritional items if present
            mandatory = rule.get("mandatory_nutrients", ["energy", "protein", "carbohydrate", "sugar", "fat", "sodium"])
            combined_text = f"{extracted_text} {str(extracted.get('nutritional_items', []))}".lower()
            missing = [m for m in mandatory if m not in combined_text]
            if missing:
                status = DetectionStatus.NEEDS_REVIEW
                summary = (
                    f"Nutritional table detected. Some mandatory nutrients may be unverified: {', '.join(missing)}."
                )
                action_needed = "Verify if complete nutritional panel (energy, carbs, sugars, fat, sodium) is present."

        res = RuleEvaluationResult(
            rule_id=rule_id,
            rule_name=rule_name,
            category=category,
            status=status,
            citation=citation,
            summary=summary,
            extracted_evidence=evidence_str or "Visible on label",
            action_needed=action_needed,
            is_noncompliance_proof=False
        )
        results.append(res.model_dump())

    return results
