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

from gemma_service import inspect_label
from rules_engine import evaluate_label
from schemas import DetectionStatus

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
st.sidebar.info("Target Model: **Google Gemma 4** (Multimodal)")

api_key_input = st.sidebar.text_input(
    "Gemma 4 / Gemini API Key",
    value=os.getenv("GEMMA_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
    type="password",
    help="Optional. If left blank, the app runs in deterministic Demo/Mock mode."
)
if api_key_input:
    os.environ["GEMMA_API_KEY"] = api_key_input

demo_mode_toggle = st.sidebar.checkbox(
    "Force Demo / Mock Mode",
    value=(os.getenv("FORCE_MOCK_GEMMA", "0") == "1"),
    help="Force simulated responses for offline hackathon testing."
)
os.environ["FORCE_MOCK_GEMMA"] = "1" if demo_mode_toggle else "0"

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
    with st.spinner("Extracting visible evidence using Gemma 4 multimodal model..."):
        extracted_data = inspect_label(image_bytes, mime_type=mime_type)

    with st.spinner("Running deterministic FSSAI 2020 rules engine..."):
        evaluation_results = evaluate_label(extracted_data, rules_config)

    # Check if mock mode was triggered
    is_mock = extracted_data.get("is_mock", False)
    if is_mock:
        st.info("ℹ️ **DEMO / MOCK EXTRACTION**: Running in simulated mode using Gemma 4 test profile.")

    # Two-Column Results Display
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📸 Packaging Image & Visual Evidence")
        st.image(image_display, caption="Captured Food Label", use_container_width=True)

        st.markdown("#### Gemma 4 Visible Extractions")
        st.caption("Extracted strictly from visible areas of the packaging:")

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
        st.caption("Deterministic evaluation against Food Safety and Standards Regulations, 2020:")

        # Summary counts
        present_count = sum(1 for r in evaluation_results if r.get("status") == "PRESENT")
        review_count = sum(1 for r in evaluation_results if r.get("status") == "NEEDS_REVIEW")
        unclear_count = sum(1 for r in evaluation_results if r.get("status") == "UNCLEAR")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Present & Verified", present_count)
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
