# Phase 7F — Genuine Inference & OCR Task Capability Decision

## Decision

No genuine inference was run and no prediction JSONL was created. At audit time, no API credential was configured, and the existing `inspect_label` output contract does not match either task in the generic OCR protocol. Mock mode was not used for benchmark predictions.

## Model implementation inspected

- Entry point: `gemma_service.inspect_label(image_bytes: bytes, mime_type: str) -> dict`.
- Model: the service default is `gemma-4-26b-a4b-it`; `GEMMA_MODEL_NAME` can override it. The environment override was unset, so this is the configured default, not a model actually used for inference.
- Inference path: hosted Google Generative Language `generateContent` API via `requests.post`, with a 75 second request timeout. The image bytes are base64-encoded and sent with the MIME type. The code does not resize, crop, or transform image coordinates.
- Generation config in code: JSON response MIME type and temperature `0.1`.
- Dependencies: `dotenv`, `pydantic`, `requests`, and Pillow are importable in the active environment. The workspace has `requirements-member1.txt`; it does not have `requirements.txt`.
- Credentials: neither `GEMMA_API_KEY` nor `GOOGLE_API_KEY` is configured; `FORCE_MOCK_GEMMA` is `0`. Per the no-fake-results requirement, no images were submitted to the mock fallback or API.
- Hardware: the implementation calls a hosted API; no local model checkpoint or GPU inference path is present in this service. Network access and valid API credentials are required for live calls.

## Task capability decision

| Protocol task | Evidence from implementation | Decision |
| --- | --- | --- |
| General text recognition for the two crops | The prompt requests structured food-label fields, and the schema returns field strings such as `product_name`, `ingredients_list`, and `nutritional_info`. It does not return the complete raw transcription of a crop as the protocol requires. | Unsupported by the current output contract; no sample was run. |
| Text-region detection for eight package images | The schema and prompt contain no text-region list, boxes, or polygons. | Unsupported by the current implementation. |
| Original-image pixel-coordinate polygons | No coordinate output or geometric postprocessing exists. | Unsupported; no polygons can be claimed or evaluated. |

The model may generate text values for specific semantic fields, but those values cannot be substituted for a raw recognition transcript. No locations can be inferred from the returned fields. This decision is based on the implementation inspected, not on a live capability test.

## Selected samples and results

- Manifest: `labelguard-real-ocr-test-v1`.
- Selected images present: 10 total, comprising 8 detection and 2 recognition samples.
- Samples attempted: 0.
- Samples completed: 0.
- Samples failed: 0 (no inference runs were attempted).
- Genuine prediction files: none found or created.
- Metrics: unavailable. No OCR evaluator run against genuine predictions.
- Baseline and targeted experiment: unavailable because no valid baseline exists. No model changes or targeted improvements were attempted.
- Food-label error examples: unavailable because no genuine model outputs exist. Generic detection annotations do not provide semantic food-field ground truth.

## Reproduction and validation

Capability/environment audit command:

```powershell
python -c "import os,importlib.util; print('model_env',os.getenv('GEMMA_MODEL_NAME') or '(unset; service default applies)'); print('api_key_configured',bool(os.getenv('GEMMA_API_KEY') or os.getenv('GOOGLE_API_KEY'))); print('dependencies',{n:bool(importlib.util.find_spec(n)) for n in ['dotenv','pydantic','requests','PIL']})"
```

Test command and result:

```text
python -m pytest -q
43 passed
```

Focused evaluator command:

```text
python -m pytest -q tests/test_real_ocr.py
18 passed
```

These are unit and synthetic-fixture tests, not model inference.

## Blockers and next action

1. Member 1 must provide a compatible genuine output path for raw text-recognition transcripts if the two crop samples are to be scored.
2. Detection remains unsupported unless the model implementation can return text-region polygons in original-image pixel coordinates. Text-only output must not be converted into polygons.
3. A live run of the existing hosted Gemma field extractor additionally requires a valid `GEMMA_API_KEY` or `GOOGLE_API_KEY`; no credential was present during this audit.

The evaluator protocol, manifest, and annotations remain unchanged. No model or dataset was downloaded, and no mock output was presented as model output.
