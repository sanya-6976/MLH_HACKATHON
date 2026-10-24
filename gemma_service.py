"""
LabelGuard AI - Gemma 4 Multimodal Inspection Service (Member 1)
Extracts visible evidence from food packaging images using Google Gemma 4.

AUDIT & SAFETY PRINCIPLES:
1. Pure extraction: Does NOT evaluate compliance or regulatory rules.
2. Honest detection: If an item is not in the camera frame, status is NOT_DETECTED.
3. Resilient execution:
   - When API credentials are absent or demo mode is active, provides an
     EXPLICITLY LABELED mock/demo result so the app remains fully testable.
   - When API credentials ARE provided but the model call fails, returns an
     explicit error result (is_error=True). NEVER silently falls back to mock
     data, ensuring model failures cannot be mistaken for genuine label evidence.
4. Model Integrity: Targets Google Gemma 4 multimodal models (e.g., gemma-4-31b-it, gemma-4-26b-a4b-it).
5. Secret Hygiene: Never prints, logs, or echoes API keys in error diagnostics or outputs.
6. Function Contract:
   inspect_label(image_bytes: bytes, mime_type: str) -> dict
"""

import os
import json
import base64
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load local environment if available
load_dotenv()

from schemas import (
    DetectionStatus,
    ExtractedField,
    NutritionalItem,
    LabelExtractionResult,
)

# Standard Gemma 4 multimodal instruction-tuned model ID (Google AI Studio / Gemini API)
# Official supported IDs: gemma-4-31b-it (dense 31B), gemma-4-26b-a4b-it (MoE 26B), gemma-4-4b-it
DEFAULT_MODEL_NAME = os.getenv("GEMMA_MODEL_NAME", "gemma-4-26b-a4b-it")


def _sanitize_message(message: str, api_key: Optional[str] = None) -> str:
    """Removes any occurrence of the raw API key from error strings."""
    if not message:
        return ""
    if api_key and api_key.strip():
        message = message.replace(api_key.strip(), "[REDACTED_API_KEY]")
    return message


def classify_api_error(
    status_code: int,
    error_data: Optional[dict] = None,
    raw_text: str = "",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classifies API errors into standard diagnostic categories without exposing credentials.
    Categories:
      - AUTHENTICATION_ERROR: Invalid, expired, or missing API key
      - MODEL_NOT_FOUND: Model identifier not supported on endpoint
      - QUOTA_EXCEEDED: Rate limit or credit quota exhausted
      - SERVICE_UNAVAILABLE: Google endpoint temporary 5xx error
      - HTTP_ERROR: Other HTTP errors
    """
    error_data = error_data or {}
    status_name = error_data.get("status", "")
    message = error_data.get("message", raw_text)
    message = _sanitize_message(message, api_key)

    # 1. Authentication & Permission
    if (
        status_code in (400, 401, 403)
        and (
            "API_KEY" in message
            or "API key" in message
            or "UNAUTHENTICATED" in status_name
            or "PERMISSION_DENIED" in status_name
            or ("INVALID_ARGUMENT" in status_name and "key" in message.lower())
        )
        or status_code in (401, 403)
    ):
        return {
            "category": "AUTHENTICATION_ERROR",
            "status_code": status_code,
            "detail": f"Authentication failed ({status_name or f'HTTP {status_code}'}): The provided API key is invalid or lacks permission for Gemma 4.",
            "suggested_fix": "Verify GEMMA_API_KEY in your .env file or Streamlit sidebar. Generate an active key in Google AI Studio."
        }

    # 2. Model Availability
    if status_code == 404 or "NOT_FOUND" in status_name or "not found" in message.lower():
        return {
            "category": "MODEL_NOT_FOUND",
            "status_code": status_code,
            "detail": f"Model was not found on this endpoint ({message[:120] if message else '404 Not Found'}).",
            "suggested_fix": "Check GEMMA_MODEL_NAME. Verified hosted model IDs are 'gemma-4-31b-it' and 'gemma-4-26b-a4b-it'."
        }

    # 3. Quota & Rate Limiting
    if status_code == 429 or "RESOURCE_EXHAUSTED" in status_name or "quota" in message.lower():
        return {
            "category": "QUOTA_EXCEEDED",
            "status_code": status_code,
            "detail": f"Rate limit or quota exhausted ({status_name or 'HTTP 429'}).",
            "suggested_fix": "Wait before retrying, or review quota and billing settings in Google AI Studio."
        }

    # 4. Service / Server Errors
    if status_code in (500, 502, 503, 504):
        return {
            "category": "SERVICE_UNAVAILABLE",
            "status_code": status_code,
            "detail": f"Temporary Google endpoint error (HTTP {status_code}).",
            "suggested_fix": "The remote endpoint is temporarily unavailable. Retry the inspection shortly."
        }

    return {
        "category": f"HTTP_{status_code}_ERROR",
        "status_code": status_code,
        "detail": f"Request failed with HTTP {status_code}: {message[:120]}",
        "suggested_fix": "Check request payload parameters and network connection."
    }


def _get_mock_label_data(mime_type: str = "image/jpeg", scenario: str = "standard_snack") -> LabelExtractionResult:
    """
    Returns an explicitly labeled mock/demo extraction result for local testing
    or when Gemma 4 API is unconfigured.
    Clearly marked with is_mock=True and descriptive model tags.
    """
    if scenario == "front_of_pack_only":
        return LabelExtractionResult(
            is_mock=True,
            is_error=False,
            model_name="gemma-4 [SIMULATED DEMO: Front-of-Pack Mock Profile]",
            image_mime_type=mime_type,
            product_name=ExtractedField(
                name="product_name",
                status=DetectionStatus.PRESENT,
                extracted_text="Masala Crunch Potato Wafers",
                confidence=0.98,
                visible_evidence="Large bold title centered on front display panel"
            ),
            veg_nonveg_symbol=ExtractedField(
                name="veg_nonveg_symbol",
                status=DetectionStatus.PRESENT,
                extracted_text="VEGETARIAN",
                confidence=0.96,
                visible_evidence="Green circle inside green square visible on top-left of front panel"
            ),
            fssai_license=ExtractedField(
                name="fssai_license",
                status=DetectionStatus.NOT_DETECTED,
                extracted_text=None,
                confidence=0.0,
                visible_evidence=None,
                notes="Not visible on front panel. Expected on back panel."
            ),
            ingredients_list=ExtractedField(
                name="ingredients_list",
                status=DetectionStatus.NOT_DETECTED,
                extracted_text=None,
                confidence=0.0,
                visible_evidence=None,
                notes="Not visible on front panel."
            ),
            nutritional_info=ExtractedField(
                name="nutritional_info",
                status=DetectionStatus.NOT_DETECTED,
                extracted_text=None,
                confidence=0.0,
                visible_evidence=None
            ),
            nutritional_items=[],
            net_quantity=ExtractedField(
                name="net_quantity",
                status=DetectionStatus.PRESENT,
                extracted_text="75 g",
                confidence=0.94,
                visible_evidence="Bottom-right corner of front panel"
            ),
            mfg_or_expiry_date=ExtractedField(
                name="mfg_or_expiry_date",
                status=DetectionStatus.UNCLEAR,
                extracted_text="MFD 10/26",
                confidence=0.65,
                visible_evidence="Faint inkjet stamp near bottom crimp seal; expiry date cut off"
            ),
            manufacturer_details=ExtractedField(
                name="manufacturer_details",
                status=DetectionStatus.NOT_DETECTED,
                extracted_text=None,
                confidence=0.0
            ),
            allergen_declaration=ExtractedField(
                name="allergen_declaration",
                status=DetectionStatus.NOT_DETECTED,
                extracted_text=None,
                confidence=0.0
            ),
            raw_summary="[DEMO MOCK] Front display panel visible. Product name, veg logo, and net weight extracted. Rear panel information not captured in photo."
        )

    # Standard snack label back-of-pack scenario
    return LabelExtractionResult(
        is_mock=True,
        is_error=False,
        model_name="gemma-4 [SIMULATED DEMO: Back Panel Full Mock Profile]",
        image_mime_type=mime_type,
        product_name=ExtractedField(
            name="product_name",
            status=DetectionStatus.PRESENT,
            extracted_text="Masala Crunch Potato Wafers",
            confidence=0.97,
            visible_evidence="Upper heading on back panel"
        ),
        veg_nonveg_symbol=ExtractedField(
            name="veg_nonveg_symbol",
            status=DetectionStatus.PRESENT,
            extracted_text="VEGETARIAN",
            confidence=0.95,
            visible_evidence="Green dot inside green square outline near product name"
        ),
        fssai_license=ExtractedField(
            name="fssai_license",
            status=DetectionStatus.PRESENT,
            extracted_text="Lic. No. 10019022009876",
            confidence=0.96,
            visible_evidence="FSSAI logo with 14-digit license number clearly printed below address"
        ),
        ingredients_list=ExtractedField(
            name="ingredients_list",
            status=DetectionStatus.PRESENT,
            extracted_text="Potatoes (60%), Edible Vegetable Oil (Palmolein), Spices and Condiments (Chilli, Coriander, Cumin, Black Pepper), Iodised Salt, Acidity Regulator (INS 330).",
            confidence=0.93,
            visible_evidence="Paragraph under 'INGREDIENTS:' heading"
        ),
        nutritional_info=ExtractedField(
            name="nutritional_info",
            status=DetectionStatus.PRESENT,
            extracted_text="Energy 540 kcal, Protein 7.2g, Carbohydrate 52g, Total Sugars 2.5g, Added Sugars 0.5g, Total Fat 34g, Saturated Fat 15g, Trans Fat 0.1g, Sodium 680mg",
            confidence=0.91,
            visible_evidence="Nutritional Information table visible per 100g"
        ),
        nutritional_items=[
            NutritionalItem(nutrient_name="Energy", value_per_100g_or_ml="540", unit="kcal"),
            NutritionalItem(nutrient_name="Protein", value_per_100g_or_ml="7.2", unit="g"),
            NutritionalItem(nutrient_name="Carbohydrate", value_per_100g_or_ml="52", unit="g"),
            NutritionalItem(nutrient_name="Total Sugars", value_per_100g_or_ml="2.5", unit="g"),
            NutritionalItem(nutrient_name="Added Sugars", value_per_100g_or_ml="0.5", unit="g"),
            NutritionalItem(nutrient_name="Total Fat", value_per_100g_or_ml="34", unit="g"),
            NutritionalItem(nutrient_name="Saturated Fat", value_per_100g_or_ml="15", unit="g"),
            NutritionalItem(nutrient_name="Trans Fat", value_per_100g_or_ml="0.1", unit="g"),
            NutritionalItem(nutrient_name="Sodium", value_per_100g_or_ml="680", unit="mg"),
        ],
        net_quantity=ExtractedField(
            name="net_quantity",
            status=DetectionStatus.PRESENT,
            extracted_text="75 g",
            confidence=0.94,
            visible_evidence="Printed near barcode"
        ),
        mfg_or_expiry_date=ExtractedField(
            name="mfg_or_expiry_date",
            status=DetectionStatus.PRESENT,
            extracted_text="Mfg Date: 12/10/2026 | Best Before 6 Months from Packaging",
            confidence=0.88,
            visible_evidence="Laser stamped in white rectangle"
        ),
        manufacturer_details=ExtractedField(
            name="manufacturer_details",
            status=DetectionStatus.PRESENT,
            extracted_text="Manufactured by: GoodBite Foods India Pvt. Ltd., Plot 42, Sector 18, Gurugram, Haryana - 122001",
            confidence=0.92,
            visible_evidence="Address section on bottom left"
        ),
        allergen_declaration=ExtractedField(
            name="allergen_declaration",
            status=DetectionStatus.PRESENT,
            extracted_text="Contains Soy. Manufactured on equipment that also processes Milk and Tree Nuts.",
            confidence=0.90,
            visible_evidence="Bold advisory box immediately below ingredients list"
        ),
        raw_summary="[DEMO MOCK] Back information panel visible. All core regulatory fields, nutrition table, and FSSAI 14-digit license extracted."
    )


def _call_live_gemma_4(
    image_bytes: bytes,
    mime_type: str,
    api_key: str,
    model_name: str
) -> Tuple[Optional[LabelExtractionResult], Optional[Dict[str, Any]]]:
    """
    Calls the Google Gemma 4 multimodal endpoint with strict JSON extraction prompt.
    Returns (result, None) on success, or (None, diagnostic_dict) on failure.
    Safeguard: Never logs or prints the API key.
    """
    try:
        import requests

        prompt = """You are an OCR and label inspection model specializing in Indian packaged food products.
Examine this image and extract all visible evidence. Do NOT evaluate regulatory compliance.
For each item, specify status as exactly one of: "PRESENT", "NOT_DETECTED", or "UNCLEAR".
Return ONLY valid JSON matching this schema:
{
  "product_name": {"status": "PRESENT|NOT_DETECTED|UNCLEAR", "extracted_text": "...", "confidence": 0.0-1.0, "visible_evidence": "..."},
  "veg_nonveg_symbol": {"status": "...", "extracted_text": "VEGETARIAN|NON_VEGETARIAN|...", "confidence": 0.0-1.0, "visible_evidence": "..."},
  "fssai_license": {"status": "...", "extracted_text": "...", "confidence": 0.0-1.0, "visible_evidence": "..."},
  "ingredients_list": {"status": "...", "extracted_text": "...", "confidence": 0.0-1.0, "visible_evidence": "..."},
  "nutritional_info": {"status": "...", "extracted_text": "...", "confidence": 0.0-1.0, "visible_evidence": "..."},
  "net_quantity": {"status": "...", "extracted_text": "...", "confidence": 0.0-1.0, "visible_evidence": "..."},
  "mfg_or_expiry_date": {"status": "...", "extracted_text": "...", "confidence": 0.0-1.0, "visible_evidence": "..."},
  "manufacturer_details": {"status": "...", "extracted_text": "...", "confidence": 0.0-1.0, "visible_evidence": "..."},
  "allergen_declaration": {"status": "...", "extracted_text": "...", "confidence": 0.0-1.0, "visible_evidence": "..."},
  "raw_summary": "Short overview of visible panels"
}
"""

        # Encode image to base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Endpoint structure for Google GenAI / Gemma 4
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        }

        resp = requests.post(url, json=payload, timeout=75)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None, {
                    "category": "MALFORMED_RESPONSE",
                    "status_code": 200,
                    "detail": "Model response contained no candidates.",
                    "suggested_fix": "Verify that the uploaded image contains visible text."
                }

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts or "text" not in parts[0]:
                return None, {
                    "category": "MALFORMED_RESPONSE",
                    "status_code": 200,
                    "detail": "Model candidate returned no text payload.",
                    "suggested_fix": "Check image encoding and format."
                }

            raw_text = parts[0]["text"]
            try:
                parsed_json = json.loads(raw_text)
            except json.JSONDecodeError as jde:
                return None, {
                    "category": "MALFORMED_RESPONSE",
                    "status_code": 200,
                    "detail": f"Model response could not be parsed as JSON: {str(jde)[:100]}",
                    "suggested_fix": "Ensure model temperature is low (0.0 - 0.2) and prompt enforces strict JSON."
                }

            def make_field(name: str) -> ExtractedField:
                val = parsed_json.get(name, {})
                status_str = val.get("status", "NOT_DETECTED")
                try:
                    status = DetectionStatus(status_str)
                except ValueError:
                    status = DetectionStatus.UNCLEAR
                return ExtractedField(
                    name=name,
                    status=status,
                    extracted_text=val.get("extracted_text"),
                    confidence=float(val.get("confidence", 0.85)),
                    visible_evidence=val.get("visible_evidence")
                )

            res = LabelExtractionResult(
                is_mock=False,
                is_error=False,
                model_name=model_name,
                image_mime_type=mime_type,
                product_name=make_field("product_name"),
                veg_nonveg_symbol=make_field("veg_nonveg_symbol"),
                fssai_license=make_field("fssai_license"),
                ingredients_list=make_field("ingredients_list"),
                nutritional_info=make_field("nutritional_info"),
                nutritional_items=[],
                net_quantity=make_field("net_quantity"),
                mfg_or_expiry_date=make_field("mfg_or_expiry_date"),
                manufacturer_details=make_field("manufacturer_details"),
                allergen_declaration=make_field("allergen_declaration"),
                raw_summary=parsed_json.get("raw_summary", f"Live extraction via Google {model_name}")
            )
            return res, None

        # Capture specific API error message and classify
        try:
            err_json = resp.json().get("error", {})
        except Exception:
            err_json = {}

        diagnostic = classify_api_error(
            status_code=resp.status_code,
            error_data=err_json,
            raw_text=resp.text,
            api_key=api_key
        )
        return None, diagnostic

    except requests.exceptions.Timeout:
        return None, {
            "category": "NETWORK_ERROR",
            "status_code": None,
            "detail": "Request to Gemma 4 endpoint timed out after 75 seconds.",
            "suggested_fix": "Check your internet connection latency."
        }
    except requests.exceptions.ConnectionError:
        return None, {
            "category": "NETWORK_ERROR",
            "status_code": None,
            "detail": "DNS resolution or socket connection failure reaching generativelanguage.googleapis.com.",
            "suggested_fix": "Verify that your machine has outgoing internet access and valid DNS."
        }
    except Exception as exc:
        return None, {
            "category": "INTERNAL_CLIENT_ERROR",
            "status_code": None,
            "detail": f"{type(exc).__name__}: {_sanitize_message(str(exc), api_key)}",
            "suggested_fix": "Check application runtime and client dependencies."
        }


def diagnose_endpoint(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Lightweight diagnostic utility to verify Gemma 4 endpoint readiness.
    Never exposes API secrets.
    """
    api_key = api_key or os.getenv("GEMMA_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = model_name or os.getenv("GEMMA_MODEL_NAME") or DEFAULT_MODEL_NAME

    if not api_key:
        return {
            "status": "UNCONFIGURED",
            "category": "NO_CREDENTIALS",
            "model_name": model_name,
            "is_live_ready": False,
            "detail": "No GEMMA_API_KEY or GOOGLE_API_KEY found in environment or .env file.",
            "suggested_fix": "Set GEMMA_API_KEY=... in .env or enter it in the Streamlit sidebar."
        }

    # Generate a minimal 1x1 test JPEG image
    from PIL import Image
    import io
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    test_bytes = buf.getvalue()

    result, diagnostic = _call_live_gemma_4(
        image_bytes=test_bytes,
        mime_type="image/jpeg",
        api_key=api_key,
        model_name=model_name
    )

    if result is not None:
        return {
            "status": "SUCCESS",
            "category": "AUTHENTICATED",
            "model_name": model_name,
            "is_live_ready": True,
            "detail": f"Successfully connected to Google {model_name} and verified multimodal response.",
            "suggested_fix": "Endpoint is fully operational."
        }

    return {
        "status": "FAILED",
        "category": diagnostic.get("category", "UNKNOWN_ERROR"),
        "model_name": model_name,
        "is_live_ready": False,
        "detail": diagnostic.get("detail", "Unknown diagnostic detail"),
        "suggested_fix": diagnostic.get("suggested_fix", "Check endpoint settings.")
    }


def inspect_label(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Contract implementation:
    inspect_label(image_bytes: bytes, mime_type: str) -> dict

    Extracts visible label information using Gemma 4.
    - If API key is configured and mock mode is not forced: executes live inference.
    - If live inference fails: returns an explicit failure result (is_error=True)
      with structured diagnostic category, preventing unextracted data from polluting the rules engine.
    - If API key is not configured or mock mode is forced: returns an explicitly
      labeled mock extraction result.
    """
    # Read API credentials and model configurations consistently
    api_key = os.getenv("GEMMA_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMMA_MODEL_NAME") or DEFAULT_MODEL_NAME
    force_mock = os.getenv("FORCE_MOCK_GEMMA", "0") in ("1", "true", "True")

    # Live Model Inference Path
    if api_key and not force_mock:
        live_result, diagnostic = _call_live_gemma_4(image_bytes, mime_type, api_key, model_name)
        if live_result is not None:
            return live_result.model_dump()

        # Task 6 & 8 Safeguard: When credentials exist but call fails, DO NOT silently
        # substitute mock data as if it were genuine extraction!
        err_cat = diagnostic.get("category", "INFERENCE_ERROR")
        err_detail = diagnostic.get("detail", "Model call failed")
        err_fix = diagnostic.get("suggested_fix", "")

        failed_result = LabelExtractionResult(
            is_mock=False,
            is_error=True,
            error_message=f"[{err_cat}] {err_detail} (Fix: {err_fix})",
            model_name=f"{model_name} [LIVE CALL FAILED: {err_cat}]",
            image_mime_type=mime_type,
            raw_summary=f"Live model inference failed with {err_cat}: {err_detail}."
        )
        return failed_result.model_dump()

    # Explicit Demo / Mock Fallback Path
    scenario = "front_of_pack_only" if len(image_bytes) < 1000 and len(image_bytes) % 2 == 1 else "standard_snack"
    mock_result = _get_mock_label_data(mime_type=mime_type, scenario=scenario)
    return mock_result.model_dump()


if __name__ == "__main__":
    import sys
    print("LabelGuard AI - Gemma 4 Multimodal Service Diagnostic")
    diag = diagnose_endpoint()
    print(json.dumps(diag, indent=2))
