"""
LabelGuard AI - Streamlit Web Interface (Member 3)
Indian Packaged Food Label Screening powered by Google Gemma 4 & Deterministic FSSAI Rules.

DISCLAIMER:
Screening assistance only — NOT legal certification. Missing visible items in a single photo
do not constitute proof of regulatory noncompliance.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import streamlit as st
from PIL import Image
import io
from dotenv import load_dotenv

# Ensure local environment variables are loaded
load_dotenv()

from gemma_service import inspect_label
from rules_engine import evaluate_label
from schemas import DetectionStatus
from evaluate_real_ocr import (
    DEFAULT_MANIFEST,
    TASK_DETECTION,
    TASK_RECOGNITION,
    _markdown_report,
    evaluate_records,
)

# Page Configuration
st.set_page_config(
    page_title="LabelGuard AI | Gemma 4 Label Screening",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Rules Configuration
RULES_PATH = Path(__file__).parent / "rules" / "food_label_rules.json"


@st.cache_data
def load_rules_config() -> dict:
    if RULES_PATH.exists():
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"rules": []}


rules_config = load_rules_config()

# Sidebar: Controls & Model Settings
st.sidebar.title("🛡️ LabelGuard AI")
st.sidebar.caption("Hackathon Track: Best Use of Gemma 4")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Configuration")

# Model ID Selector (consistent with GEMMA_MODEL_NAME)
current_model_env = os.getenv("GEMMA_MODEL_NAME", "gemma-4-31b-it")
model_options = ["gemma-4-31b-it", "gemma-4-26b-a4b-it", "gemma-4-4b-it"]
default_idx = model_options.index(current_model_env) if current_model_env in model_options else 0

selected_model = st.sidebar.selectbox(
    "Gemma 4 Multimodal Model ID",
    options=model_options,
    index=default_idx,
    help="Target Google Gemma 4 multimodal model identifier."
)
os.environ["GEMMA_MODEL_NAME"] = selected_model

# API Key Input (never logged, updates GEMMA_API_KEY)
current_api_key = os.getenv("GEMMA_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "Gemma 4 / Google Gemini API Key",
    value=current_api_key,
    type="password",
    help="Optional. If left blank or invalid, the app runs in explicit Demo/Mock mode."
)
if api_key_input:
    os.environ["GEMMA_API_KEY"] = api_key_input
    os.environ["GOOGLE_API_KEY"] = api_key_input
elif "GEMMA_API_KEY" in os.environ and not current_api_key:
    os.environ.pop("GEMMA_API_KEY", None)

demo_mode_toggle = st.sidebar.checkbox(
    "Force Demo / Mock Mode",
    value=(os.getenv("FORCE_MOCK_GEMMA", "0") in ("1", "true", "True")),
    help="Force simulated responses for offline hackathon testing."
)
os.environ["FORCE_MOCK_GEMMA"] = "1" if demo_mode_toggle else "0"

if not api_key_input and not demo_mode_toggle:
    st.sidebar.info("💡 No API key configured: Application is running in **Simulated Demo Mode**.")
elif api_key_input and not demo_mode_toggle:
    st.sidebar.success("🔑 API Key configured: Live Gemma 4 inference active.")

with st.sidebar.expander("🔌 Test Live Gemma 4 Connection"):
    if st.button("Run Endpoint Diagnostic", key="diag_btn"):
        from gemma_service import diagnose_endpoint
        with st.spinner("Checking Gemma 4 endpoint connectivity..."):
            diag = diagnose_endpoint(api_key=api_key_input, model_name=selected_model)
        if diag.get("status") == "SUCCESS":
            st.success(f"✅ {diag.get('detail')}")
        elif diag.get("status") == "UNCONFIGURED":
            st.info(f"ℹ️ {diag.get('detail')}\n\n👉 {diag.get('suggested_fix')}")
        else:
            st.error(f"❌ **{diag.get('category')}**: {diag.get('detail')}\n\n👉 **Suggested Fix**: {diag.get('suggested_fix')}")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Preloaded Demo Scenarios")
demo_scenario = st.sidebar.radio(
    "Test with Sample Packaging:",
    options=[
        "Upload Custom Photo",
        "Demo 1: Full Back Panel (Snack Wafers - All Visible)",
        "Demo 2: Front-of-Pack Only (Ingredients & FSSAI on other side)",
    ],
    index=0
)

# Main Header & Statutory Disclaimer
st.title("🛡️ LabelGuard AI")
st.markdown("### Automated Indian Packaged Food Label Screening Assistance")

st.warning(
    "⚠️ **LEGAL & REGULATORY DISCLAIMER**: This application provides initial screening assistance "
    "and evidence extraction only. It does **NOT** provide legal certification or compliance guarantees. "
    "Packaged food containers have multiple sides (front, back, sides, crimp folds). An item flagged as "
    "**NOT_DETECTED** indicates it is not visible in the uploaded image, and is **NEVER** proof of regulatory noncompliance."
)

# File Uploader
uploaded_file = None
if demo_scenario == "Upload Custom Photo":
    uploaded_file = st.file_uploader(
        "Upload a clear photograph of an Indian packaged-food label:",
        type=["png", "jpg", "jpeg", "webp"],
        help="Ensure text, FSSAI logo, and markings are reasonably clear."
    )
else:
    st.info(f"Using {demo_scenario}. (Mock label stream active for rapid hackathon evaluation)")

# Process Image
image_bytes = None
mime_type = "image/jpeg"

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type or "image/jpeg"
    image_display = Image.open(io.BytesIO(image_bytes))
elif demo_scenario == "Demo 1: Full Back Panel (Snack Wafers - All Visible)":
    # Synthetic placeholder image for demo 1 (even byte length for standard snack)
    img = Image.new("RGB", (600, 400), color=(240, 244, 248))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()
    image_display = img
elif demo_scenario == "Demo 2: Front-of-Pack Only (Ingredients & FSSAI on other side)":
    # Synthetic placeholder image with odd byte length for front-of-pack scenario
    img = Image.new("RGB", (600, 400), color=(254, 243, 199))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw = buf.getvalue()
    image_bytes = raw if len(raw) % 2 == 1 else raw + b" "
    image_display = img

if image_bytes:
    # Execute Pipeline
    with st.spinner(f"Extracting visible evidence using {os.getenv('GEMMA_MODEL_NAME', 'gemma-4')}..."):
        extracted_data = inspect_label(image_bytes, mime_type=mime_type)

    with st.spinner("Running deterministic FSSAI 2020 rules engine..."):
        evaluation_results = evaluate_label(extracted_data, rules_config)

    # Check for live model failure (Task 6)
    is_error = extracted_data.get("is_error", False)
    if is_error:
        st.error(
            f"❌ **Live Gemma 4 Model Inference Failed**: {extracted_data.get('error_message')}. "
            "Model failures are strictly isolated and are never passed as genuine extracted label data. "
            "Please check your API key, model selection, or enable 'Force Demo / Mock Mode' in the sidebar."
        )

    # Check for mock mode (Task 5)
    is_mock = extracted_data.get("is_mock", False)
    if is_mock:
        st.warning(
            f"⚠️ **SIMULATED DEMO OUTPUT**: Running in mock mode ({extracted_data.get('model_name')}). "
            "This output is sample demonstration data and is **NOT live model output**."
        )

    # Two-Column Results Display
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📸 Packaging Image & Visual Evidence")
        st.image(image_display, caption="Captured Food Label", use_container_width=True)

        st.markdown("#### Gemma 4 Visible Extractions")
        st.caption(f"Model: `{extracted_data.get('model_name')}`")

        fields_to_show = [
            ("Product Name", extracted_data.get("product_name")),
            ("Dietary Symbol (Veg/Non-Veg)", extracted_data.get("veg_nonveg_symbol")),
            ("FSSAI Logo & License", extracted_data.get("fssai_license")),
            ("Net Quantity", extracted_data.get("net_quantity")),
            ("Ingredients List", extracted_data.get("ingredients_list")),
            ("Nutritional Table", extracted_data.get("nutritional_info")),
            ("Date of Mfg / Expiry", extracted_data.get("mfg_or_expiry_date")),
            ("Manufacturer Details", extracted_data.get("manufacturer_details")),
            ("Allergen Declaration", extracted_data.get("allergen_declaration")),
        ]

        for label, fld in fields_to_show:
            if not fld:
                continue
            status = fld.get("status", "NOT_DETECTED")
            text = fld.get("extracted_text")
            evidence = fld.get("visible_evidence")
            confidence = fld.get("confidence", 0.0)

            # Badge color mapping
            badge_icon = "⚪"
            if status == "PRESENT":
                badge_icon = "🟢"
            elif status == "UNCLEAR":
                badge_icon = "🟠"
            elif status == "NEEDS_REVIEW":
                badge_icon = "🟡"

            with st.expander(f"{badge_icon} **{label}**: `{status}`", expanded=(status != "NOT_DETECTED")):
                if text:
                    st.markdown(f"**Extracted Text**: {text}")
                if evidence:
                    st.markdown(f"**Visual Evidence**: {evidence}")
                if confidence > 0:
                    st.caption(f"Extraction Confidence: {confidence * 100:.1f}%")

    with col_right:
        st.subheader("⚖️ FSSAI (Labelling & Display) Screening")
        st.caption(
            "Configured screening checks referencing FSSAI labelling provisions. "
            "These findings are not legal verification or compliance certification."
        )

        # Summary counts
        present_count = sum(1 for r in evaluation_results if r.get("status") == "PRESENT")
        review_count = sum(1 for r in evaluation_results if r.get("status") == "NEEDS_REVIEW")
        unclear_count = sum(1 for r in evaluation_results if r.get("status") == "UNCLEAR")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Checks marked PRESENT", present_count)
        metric_col2.metric("Needs Review / Other Sides", review_count)
        metric_col3.metric("Unclear / Blurry", unclear_count)

        st.markdown("---")

        for r in evaluation_results:
            rule_id = r.get("rule_id")
            rule_name = r.get("rule_name")
            category = r.get("category")
            status = r.get("status")
            citation = r.get("citation")
            summary = r.get("summary")
            action = r.get("action_needed")
            evidence = r.get("extracted_evidence")

            icon = "🟢" if status == "PRESENT" else ("🟠" if status == "UNCLEAR" else "🟡")

            with st.container(border=True):
                st.markdown(f"### {icon} {rule_name}")
                st.caption(f"**Category**: {category} | **Rule ID**: `{rule_id}`")
                st.markdown(f"**Citation**: *{citation}*")
                st.markdown(f"**Screening Finding**: {summary}")

                if evidence:
                    st.info(f"**Observed Evidence**: {evidence}")

                if action:
                    st.warning(f"👉 **Action / Note**: {action}")

    # Export Report
    st.markdown("---")
    st.subheader("📥 Export Screening Summary")
    report_data = {
        "disclaimer": "Screening assistance only, NOT legal certification. NOT_DETECTED is not noncompliance.",
        "gemma_extraction": extracted_data,
        "rules_evaluation": evaluation_results,
    }
    st.download_button(
        label="Download JSON Audit Report",
        data=json.dumps(report_data, indent=2),
        file_name="labelguard_screening_report.json",
        mime="application/json"
    )
else:
    st.info("👆 Please upload a food label image or choose a preloaded demo scenario from the sidebar to begin screening.")


# Member 2's offline OCR evaluator is separate from the Gemma field-extraction flow.
st.markdown("---")
st.header("Generic OCR Evaluation")
st.caption(
    "Evaluate genuine JSONL predictions against the selected generic OCR test subset. "
    "This does not measure ingredient extraction, nutrition extraction, expiry dates, or compliance. "
    "The uploaded predictions are evaluated locally and are not sent to Gemma."
)

dataset_root_value = st.text_input(
    "Local dataset root (contains the manifest-relative dataset paths)",
    value=os.getenv("LABELGUARD_OCR_DATASET_ROOT", str(Path(__file__).parent)),
    key="ocr_dataset_root",
)
ocr_predictions_file = st.file_uploader(
    "Upload genuine UTF-8 OCR predictions (.jsonl)",
    type=["jsonl"],
    key="ocr_predictions_jsonl",
)

if st.button("Evaluate OCR predictions", key="evaluate_ocr_predictions"):
    if ocr_predictions_file is None:
        st.warning("Upload a genuine model predictions JSONL file first. Mock or illustrative predictions are not accepted as model results.")
    else:
        dataset_root = Path(dataset_root_value).expanduser()
        try:
            manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
            missing_data = []
            for entry in manifest["entries"]:
                for relative_path in (entry["original_relative_path"], entry["annotation_path"]):
                    if not (dataset_root / relative_path).is_file():
                        missing_data.append(relative_path)
            if missing_data:
                st.error(
                    "The selected dataset images/annotations are not available under this root. "
                    "Set the dataset root to the local directory that contains the manifest paths. "
                    f"Missing files include: {', '.join(missing_data[:4])}"
                )
                st.session_state.pop("ocr_evaluation_report", None)
            else:
                raw_jsonl = ocr_predictions_file.getvalue().decode("utf-8-sig")
                records = []
                parse_errors = []
                for line_number, line in enumerate(raw_jsonl.splitlines(), start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append((line_number, json.loads(line)))
                    except json.JSONDecodeError as error:
                        parse_errors.append(
                            {"line": line_number, "error": f"malformed JSON: {error.msg} (column {error.colno})"}
                        )
                st.session_state["ocr_evaluation_report"] = evaluate_records(
                    manifest, records, dataset_root, parse_errors
                )
        except (UnicodeDecodeError, OSError, json.JSONDecodeError, KeyError, ValueError) as error:
            st.error(f"OCR evaluation could not start: {error}")
            st.session_state.pop("ocr_evaluation_report", None)

ocr_report = st.session_state.get("ocr_evaluation_report")
if ocr_report:
    detection = ocr_report["metrics"][TASK_DETECTION]
    recognition = ocr_report["metrics"][TASK_RECOGNITION]
    st.subheader("Text Detection")
    det_cols = st.columns(3)
    for column, label, value in zip(
        det_cols,
        ("Precision", "Recall", "F1"),
        (detection["precision"], detection["recall"], detection["f1"]),
    ):
        column.metric(label, "n/a" if value is None else f"{value:.4f}")
    st.caption(
        f"Scored {detection['scored_samples']} / {detection['expected_samples']} images · "
        f"TP {detection['tp']} · FP {detection['fp']} · FN {detection['fn']} · "
        f"IoU threshold {detection['iou_threshold']:.2f}"
    )

    st.subheader("Text Recognition")
    rec_cols = st.columns(3)
    for column, label, value in zip(
        rec_cols,
        ("Exact match", "CER", "WER"),
        (recognition["exact_match_accuracy"], recognition["cer"], recognition["wer"]),
    ):
        column.metric(label, "n/a" if value is None else f"{value:.4f}")
    st.caption(
        f"Scored {recognition['scored_samples']} / {recognition['expected_samples']} crops · "
        f"Missing {recognition['missing_samples']} · Failed {recognition['failed_samples']} · "
        f"Invalid {recognition['invalid_samples']}"
    )
    if ocr_report["invalid_records"] or ocr_report["input_errors"] or ocr_report["missing_sample_ids"]:
        st.warning("Some prediction records were invalid or missing. Review the report details before interpreting metrics.")
    with st.expander("Sample-level results and validation errors"):
        st.json({
            "invalid_records": ocr_report["invalid_records"],
            "input_errors": ocr_report["input_errors"],
            "missing_sample_ids": ocr_report["missing_sample_ids"],
            "detection_samples": detection["samples"],
            "recognition_samples": recognition["samples"],
        })
    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download JSON evaluation report",
        data=json.dumps(ocr_report, ensure_ascii=False, indent=2, allow_nan=False),
        file_name="real_ocr_report.json",
        mime="application/json",
        key="download_ocr_json",
    )
    download_cols[1].download_button(
        "Download Markdown evaluation report",
        data=_markdown_report(ocr_report),
        file_name="real_ocr_report.md",
        mime="text/markdown",
        key="download_ocr_markdown",
    )
