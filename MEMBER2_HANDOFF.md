# Member 2 Handoff — Generic OCR Evaluation

## Frontend connection

The Streamlit app's **Generic OCR Evaluation** panel accepts a user-supplied genuine UTF-8 JSONL file. It runs the evaluator locally against the selected ten-sample manifest and exposes JSON and Markdown downloads. It does not call Gemma, and it does not convert Gemma's structured label fields into raw OCR transcripts or polygons.

Set `LABELGUARD_OCR_DATASET_ROOT` to a local directory containing the manifest-relative `Food Packaging OCR Dataset/` paths, or enter that directory in the panel. The dataset is intentionally not committed because its source licence and redistribution terms remain unknown.

## Command-line use

```bash
python evaluate_real_ocr.py path/to/genuine_predictions.jsonl --output-dir evaluation_reports --dataset-root /path/to/dataset-root
```

The evaluator requires the record schema in `sample_data/REAL_OCR_EVALUATION_PROTOCOL.md`. It validates sample IDs, task types, required fields, duplicates, missing samples, and task-specific prediction shapes. Detection uses polygon IoU ≥ 0.50 and one-to-one maximum-total-IoU assignment. Recognition uses protocol normalization and reports exact match, CER, and WER. The metrics remain separate.

## Current evidence and limitations

No genuine prediction file or OCR score is included. The linked `gemma_service.inspect_label` returns structured food-label fields, not a raw recognition transcript or original-image pixel polygons. Thus this frontend integration is evaluation-only until a compatible genuine prediction producer is supplied. These generic OCR annotations do not measure ingredient, nutrition, expiry, allergen, or regulatory extraction.

Test command: `python -m pytest -q`. Results are recorded in `sample_data/PHASE7F_INFERENCE_DECISION.md`; tests use synthetic fixtures and do not represent Gemma inference.
