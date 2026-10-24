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

1. **Multimodal Visual Evidence Extraction**: Google **Gemma 4** (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`) inspects label imagery to identify what is actually visible on the packaging (text, logos, symbols, dates, and tables).
2. **Deterministic Rules Evaluation**: A separate, verifiable Python rules engine evaluates extracted evidence against configurable statutory rules in `rules/food_label_rules.json`.
3. **Model Failure Isolation**: If live model inference fails or is throttled, rules evaluation is safely blocked with `NEEDS_REVIEW` instead of injecting false compliance findings or hallucinated data.
4. **Transparent Uncertainty**: Explicitly presents statuses (`PRESENT`, `NOT_DETECTED`, `UNCLEAR`, `NEEDS_REVIEW`) and visual evidence so users can understand and verify every finding.

---

## 🤝 Team Contract & Module Ownership

| Role | Owner | Primary Modules | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Member 1** | Model & Schemas | `gemma_service.py`, `schemas.py`, `tests/test_schema.py` | Multimodal Gemma 4 extraction, Pydantic data schemas, status contracts, diagnostics |
| **Member 2** | Rules & Verification | `rules_engine.py`, `rules/food_label_rules.json`, `tests/test_rules_engine.py` | Deterministic FSSAI 2020 checks, regex & nutrient validation, noncompliance safeguards |
| **Member 3** | UI & Integration | `app.py`, `cli_inspect.py`, `tests/test_integration.py` | Streamlit interactive dashboard, CLI inspector, demo scenarios, integration tests |

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
├── cli_inspect.py              # Command-line tool to test real JPG/PNG images
├── gemma_service.py            # Google Gemma 4 multimodal extraction & diagnostics
├── schemas.py                  # Pydantic schemas and shared status enums
├── rules_engine.py             # Deterministic FSSAI 2020 rule evaluator
├── rules/
│   └── food_label_rules.json   # Configurable FSSAI 2020 regulations & citations
├── tests/
│   ├── test_schema.py          # Schema, error fields, and validation unit tests
│   ├── test_rules_engine.py    # Deterministic rule engine & noncompliance safeguard tests
│   └── test_integration.py     # End-to-end pipeline, contract, and diagnostic tests
├── requirements.txt            # Project dependencies
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore rules (protects secrets like .env)
└── README.md                   # Hackathon documentation and guide
```

---

## 🚀 Running Live Tests Locally

### Step 1: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Configure Your API Key
Create a `.env` file in the project root:
```ini
GEMMA_API_KEY=your_google_ai_studio_api_key_here
GEMMA_MODEL_NAME=gemma-4-31b-it
```
*(Note: `.env` is gitignored to ensure API keys are never committed).*

### Step 3: Run Endpoint Diagnostic
Verify your credentials and endpoint connectivity:
```bash
python cli_inspect.py --diagnose
```
This test will classify your status:
- **`AUTHENTICATED`**: Key is valid and endpoint is reachable.
- **`NO_CREDENTIALS`**: No key set (app runs in simulated demo mode).
- **`AUTHENTICATION_ERROR` / `MODEL_NOT_FOUND` / `QUOTA_EXCEEDED`**: Reports exact failure category without exposing keys.

### Step 4: Run Live Inference on a Local Image (CLI)
You can inspect any real food packaging image directly from your terminal:
```bash
python cli_inspect.py path/to/packaged_food_label.jpg
```
The CLI will output the extraction source (Live Gemma 4 vs Demo Mock), all visible fields, and the deterministic FSSAI screening findings.

### Step 5: Run the Streamlit Web Application
```bash
streamlit run app.py
```
1. Open the local URL displayed in the terminal (default: `http://localhost:8501`).
2. Use the sidebar to enter your API key or test connection diagnostics.
3. Upload any packaged food label (`.jpg`, `.jpeg`, `.png`, `.webp`) or select a preloaded demo scenario.
4. Review the extracted evidence and download the JSON audit report.

### Step 6: Run Full Automated Test Suite
```bash
pytest -v
```

---

## 📜 Verifiable Regulatory References
All rules in `rules/food_label_rules.json` reference actual statutory clauses under the **Food Safety and Standards (Labelling and Display) Regulations, 2020 (as amended)**:
- **Regulation 5(1)**: Name of the Food (on Principal Display Panel)
- **Regulation 5(2)**: List of Ingredients in descending order of in-going weight/volume
- **Regulation 5(3)**: Mandatory Nutritional Information panel per 100g/ml or per serve
- **Regulation 5(3)(ix)**: Food Allergen Declarations
- **Regulation 5(4)**: Vegetarian (Green dot) / Non-Vegetarian (Brown triangle) logo
- **Regulation 5(6)**: Name & physical address of manufacturer/packer
- **Regulation 5(7)**: Official FSSAI Logo and 14-digit License Number
- **Regulation 5(8)**: Net Quantity and Metric measurements (Legal Metrology Rules)
- **Regulation 5(10)**: Date of manufacture/packaging and Best Before / Expiry dates
