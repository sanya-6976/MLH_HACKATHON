"""
LabelGuard AI - Command-Line Label Inspector (CLI)
Allows running inspect_label() and evaluate_label() on any real JPG or PNG image.

Usage:
    python cli_inspect.py path/to/image.jpg
    python cli_inspect.py path/to/image.png
    python cli_inspect.py --diagnose
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment is loaded
load_dotenv()

from gemma_service import inspect_label, diagnose_endpoint, DEFAULT_MODEL_NAME
from rules_engine import evaluate_label
from schemas import DetectionStatus

RULES_FILE = Path(__file__).parent / "rules" / "food_label_rules.json"


def load_rules() -> dict:
    if RULES_FILE.exists():
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"rules": []}


def inspect_image_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    ext = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    if ext not in mime_map:
        print(f"Error: Unsupported image extension '{ext}'. Supported: .jpg, .jpeg, .png, .webp")
        sys.exit(1)

    mime_type = mime_map[ext]
    image_bytes = path.read_bytes()

    print("=" * 65)
    print("LabelGuard AI - Packaged Food Label Inspection (Gemma 4)")
    print("=" * 65)
    print(f"File: {path.name} ({len(image_bytes):,} bytes, {mime_type})")

    api_key = os.getenv("GEMMA_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMMA_MODEL_NAME", DEFAULT_MODEL_NAME)
    force_mock = os.getenv("FORCE_MOCK_GEMMA", "0") in ("1", "true", "True")

    if api_key and not force_mock:
        print(f"Mode: LIVE GEMMA 4 INFERENCE (Model: {model_name})")
    else:
        print("Mode: SIMULATED DEMO FALLBACK (No API key or FORCE_MOCK_GEMMA=1)")

    print("-" * 65)
    print("Running inspect_label()...")
    extracted = inspect_label(image_bytes, mime_type=mime_type)

    is_error = extracted.get("is_error", False)
    is_mock = extracted.get("is_mock", False)

    if is_error:
        print(f"\n[!] INFERENCE FAILED: {extracted.get('error_message')}")
        print("Note: Model failures are isolated; rules engine evaluation will be safely blocked.\n")
    elif is_mock:
        print(f"\n[*] Demo Mock Profile Active: {extracted.get('model_name')}\n")
    else:
        print(f"\n[+] Live Model Extraction Successful: {extracted.get('model_name')}\n")

    # Display Extracted Fields
    print("--- Extracted Visible Evidence ---")
    fields = [
        ("product_name", "Product Name"),
        ("veg_nonveg_symbol", "Dietary Symbol"),
        ("fssai_license", "FSSAI License"),
        ("net_quantity", "Net Quantity"),
        ("ingredients_list", "Ingredients List"),
        ("nutritional_info", "Nutritional Table"),
        ("mfg_or_expiry_date", "Mfg / Expiry Date"),
        ("manufacturer_details", "Manufacturer"),
        ("allergen_declaration", "Allergen Advisory"),
    ]

    for key, label in fields:
        fld = extracted.get(key, {})
        status = fld.get("status", "NOT_DETECTED")
        status_val = status.value if hasattr(status, "value") else str(status)
        text = fld.get("extracted_text")
        evidence = fld.get("visible_evidence")
        print(f"  [{status_val:<12}] {label:<20}: {text or '(None visible)'}")
        if evidence:
            print(f"                  Evidence: {evidence}")

    # Evaluate Rules
    print("\n--- FSSAI (Labelling and Display) Regulations 2020 Screening ---")
    rules_data = load_rules()
    evaluated = evaluate_label(extracted, rules_data)

    for item in evaluated:
        st_raw = item["status"]
        st_val = st_raw.value if hasattr(st_raw, "value") else str(st_raw)
        rule_name = item["rule_name"]
        citation = item["citation"]
        summary = item["summary"]
        print(f"  [{st_val:<12}] {rule_name}")
        print(f"                  Citation: {citation}")
        print(f"                  Finding : {summary}")
        if item.get("action_needed"):
            print(f"                  Action  : {item['action_needed']}")

    print("\n" + "=" * 65)
    print("DISCLAIMER: Screening assistance only, NOT legal certification.")
    print("NOT_DETECTED is absence in this photo view, NEVER proof of noncompliance.")
    print("=" * 65)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--diagnose":
        print("Running Gemma 4 endpoint diagnostic...")
        diag = diagnose_endpoint()
        print(json.dumps(diag, indent=2))
        sys.exit(0)

    inspect_image_file(sys.argv[1])


if __name__ == "__main__":
    main()
