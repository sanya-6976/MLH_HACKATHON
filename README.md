# LabelGuard AI 🛡️
**Automated Indian Packaged Food Label Screening Assistance**  
*Built for Hackathon Track: Best Use of Gemma 4*

---

## ⚠️ Important Regulatory & Screening Disclaimer
> **This application provides screening assistance only — NOT legal certification.**  
> Packaged food products have multiple sides (front-of-pack, back-of-pack, side panels, crimp seals). A single camera angle only captures visible panels. **`NOT_DETECTED` is NEVER proof of regulatory noncompliance.** When an element is not detected in a photo, LabelGuard flags it for review across the remaining physical panels.

---

## 🎯 Project Overview
LabelGuard AI assists consumers, reviewers, and quality teams in screening packaged-food labels sold in India against verifiable provisions of the **Food Safety and Standards (Labelling and Display) Regulations, 2020**:

1. **Multimodal Visual Evidence Extraction**: Google **Gemma 4** inspects label imagery to identify what is actually visible on the packaging (text, logos, symbols, dates, and tables).
2. **Deterministic Rules Evaluation**: A separate, verifiable Python rules engine evaluates extracted evidence against configurable rules in `rules/food_label_rules.json`.
3. **Transparent Uncertainty**: Explicitly presents statuses (`PRESENT`, `NOT_DETECTED`, `UNCLEAR`, `NEEDS_REVIEW`) and visual evidence so users can understand and verify every finding.

---

## 🤝 Team Contract & Module Ownership

| Role | Owner | Primary Modules | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Member 1** | Model & Schemas | `gemma_service.py`, `schemas.py`, `tests/test_schema.py` | Multimodal Gemma 4 extraction, Pydantic data schemas, status contracts |
| **Member 2** | Rules & Verification | `rules_engine.py`, `rules/food_label_rules.json`, `tests/test_rules_engine.py` | Deterministic FSSAI 2020 checks, regex & nutrient validation, noncompliance safeguards |
| **Member 3** | UI & Integration | `app.py`, `tests/test_integration.py` | Streamlit interactive dashboard, demo scenarios, end-to-end integration tests |

### Strict Function Contracts
- **`inspect_label(image_bytes: bytes, mime_type: str) -> dict`**  
  *Implemented in `gemma_service.py`.* Extracts visible label features without evaluating compliance rules.
- **`evaluate_label(extracted: dict, rules: dict) -> list[dict]`**  
  *Implemented in `rules_engine.py`.* Deterministically screens extracted dictionary against rule specifications.

### Shared Statuses
- **`PRESENT`**: Evidence is clearly detected and verified on visible label.
- **`NOT_DETECTED`**: Element not found in this image view (flags for multi-panel review).
- **`UNCLEAR`**: Text or graphic is blurry, faint, or partially occluded.
- **`NEEDS_REVIEW`**: Ambiguous classification or formatting requiring manual check.

---

## 📁 Project Structure

```text
labelguard-ai/
├── app.py                      # Streamlit interactive UI & demo runner
├── gemma_service.py            # Google Gemma 4 multimodal extraction
├── schemas.py                  # Pydantic schemas and shared status enums
├── rules_engine.py             # Deterministic FSSAI 2020 rule evaluator
├── rules/
│   └── food_label_rules.json   # Configurable FSSAI 2020 regulations & citations
├── tests/
│   ├── test_schema.py          # Schema and validation unit tests
│   ├── test_rules_engine.py    # Deterministic rule engine tests
│   └── test_integration.py     # End-to-end pipeline & contract tests
├── requirements.txt            # Project dependencies
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore rules
└── README.md                   # Hackathon documentation and guide
```

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Configure Environment (Optional for Live API)
Copy the example environment configuration:
```bash
cp .env.example .env
```
Add your Google API key to `.env` if testing live Gemma 4 extraction:
```ini
GEMMA_API_KEY=your_gemini_api_key_here
GEMMA_MODEL_NAME=gemma-4
```
> **Offline / Demo Resilience**: If no API key is configured or the model endpoint is offline, LabelGuard AI automatically falls back to an **explicitly labeled mock/demo mode** so hackathon judges and evaluators can immediately test all scenarios without setup friction.

### 3. Run Test Suite
```bash
pytest -v
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```

---

## 📜 Verifiable Regulatory References
All rules in `rules/food_label_rules.json` reference actual statutory clauses under the **Food Safety and Standards (Labelling and Display) Regulations, 2020**:
- **Regulation 5(1)**: List of Ingredients in descending order
- **Regulation 5(2)**: Mandatory Nutritional Information panel per 100g/ml
- **Regulation 5(3)(ix)**: Food Allergen Declarations
- **Regulation 5(4)**: Vegetarian (Green dot) / Non-Vegetarian (Brown triangle) logo
- **Regulation 5(5)**: Net Quantity and Metric measurements
- **Regulation 5(6)**: Name & physical address of manufacturer/packer
- **Regulation 5(7)**: FSSAI Logo and 14-digit License Number
- **Regulation 5(10)**: Date of manufacture/packaging and Best Before / Expiry dates
