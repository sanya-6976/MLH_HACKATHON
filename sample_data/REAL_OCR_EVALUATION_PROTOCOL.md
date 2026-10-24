# Real OCR Evaluation Protocol

## Scope and provenance

This protocol applies only to the ten real test-split samples in `real_evaluation_manifest.json`: eight `det/test` package photos and two `rec/test` text crops. It is separate from the eight synthetic examples in `evaluation_manifest.json`. Do not merge the records or copy the source images.

The source URL, publisher/authors, dataset version, licence, attribution terms, and redistribution permission remain **UNKNOWN**. Resolve those from the original source before external demonstration or redistribution. The dataset provides generic OCR annotations only; it does not provide ground truth for ingredients, nutrition values, expiry dates, allergens, product-specific fields, or regulatory compliance. No predictions or model accuracy are present in this protocol, and Gemma has not been evaluated.

## Current split inventory

Rechecked in the current workspace on 2026-10-24. Each split has an `images/` directory, `labels.txt`, and the same number of JPEG images and label rows:

| Task/split | JPEG images | Label rows | First label row (abbreviated) |
| --- | ---: | ---: | --- |
| `det/train` | 8,736 | 8,736 | `0001.jpg` with transcription and polygon JSON |
| `det/valid` | 1,092 | 1,092 | `0001.jpg` with transcription and polygon JSON |
| `det/test` | 1,092 | 1,092 | `0001.jpg` with transcription and polygon JSON |
| `rec/train` | 8,736 | 8,736 | `0001.jpg<TAB>kazidomi` |
| `rec/valid` | 1,092 | 1,092 | `0001.jpg<TAB>ALFREZI` |
| `rec/test` | 1,092 | 1,092 | `0001.jpg<TAB>REDBUSH` |

This differs from the extraction status recorded in `real_dataset_audit.md`: that audit said `rec/train` was partial at 4,267 JPEGs and `rec/valid` was missing. The current filesystem now contains the complete README-described counts for both recognition splits. The audit's earlier count is retained as a historical snapshot, not the current state. No other split-count discrepancy was found in this recheck.

The README describes detection rows as tab-separated image filename plus a JSON list of `{ "transcription", "points" }` regions, and recognition rows as tab-separated filename plus transcription. It does not specify point coordinate conventions beyond polygons; the protocol treats the provided values as pixel coordinates `(x, y)` in the corresponding original image and records this as an assumption to verify if upstream documentation becomes available.

## Text detection procedure

Score only localization for the `text_detection` task. Do not include predicted text transcription correctness in detection TP/FP/FN; any later text-spotting score must be separately named and reported. Each ground-truth region is one polygon from the referenced `det/<split>/labels.txt` row. Each predicted region must provide a polygon in the original image's pixel coordinate frame.

1. Parse the exact ground-truth row named by the manifest's `annotation_path` and image filename. Validate that each polygon has at least three distinct points and finite coordinates. Treat malformed ground truth as an evaluation blocker for that image; do not silently count it as a miss.
2. For each valid predicted/ground-truth polygon pair in an image, calculate polygon intersection-over-union (IoU): `area(intersection) / area(union)`. Use polygon geometry, not axis-aligned bounding-box IoU. Use a robust polygon intersection implementation and polygon area (shoelace formula); do not resize or otherwise transform one side independently.
3. A pair is eligible when IoU is **at least 0.50**. Perform a one-to-one maximum-total-IoU bipartite assignment among eligible pairs for that image. Each prediction and each ground-truth polygon can be matched at most once. Resolve exact assignment ties deterministically by original prediction order, then ground-truth annotation order.
4. Count matched pairs as true positives (TP), unmatched predictions as false positives (FP), and unmatched ground-truth regions as false negatives (FN). An invalid predicted polygon is an FP and must also be recorded in `errors`; a prediction with no regions is a valid empty result, not a missing run.
5. Aggregate counts across eligible images and report detection precision `TP / (TP + FP)`, recall `TP / (TP + FN)`, and F1. Report TP, FP, FN, number of scored images, and the IoU threshold alongside the metrics. If a denominator is zero, report the metric as `null`, not zero. Do not tune the 0.50 threshold on this test subset.

The dataset's detection labels also include text strings, but this detection procedure does not assess them. The separate recognition task below uses only the recognition crop annotations; do not combine the task metrics.

## Text recognition procedure

Score only the `text_recognition` entries in `rec/test`. Each selected image is a text crop and its exact expected transcription is the value in its referenced `rec/<split>/labels.txt` row.

Preserve raw prediction and reference strings. For the primary comparison, normalize both with Unicode NFC, trim leading/trailing whitespace, and collapse each run of whitespace to one ASCII space. Do not case-fold, remove punctuation, strip accents/diacritics, transliterate, or otherwise rewrite characters. Report the normalization exactly as stated here.

Report exact-match accuracy (normalized prediction equals normalized reference) and character error rate (CER) separately from detection metrics. CER is total Levenshtein character edits divided by total reference characters across scored recognition crops, computed on the normalized Unicode code-point sequences. A missing/empty prediction is scored as an empty string (all reference characters are deletions); a failed or absent run is not scored. Report the number of scored crops and raw plus normalized strings so results can be audited. With only two selected crops, characterize any result as a smoke check, not a representative dataset accuracy estimate.

## Prediction handoff format

Member 1 should provide UTF-8 JSON Lines (`.jsonl`), one record per selected sample and task actually run. Keep detection and recognition as separate records. Use the manifest `image_id` as `sample_id`; do not use a product-field schema. Detection coordinates must refer to the original image frame. Recognition prediction must be the raw predicted transcription.

Required record fields:

- `sample_id`: `image_id` from `real_evaluation_manifest.json`.
- `task_type`: `text_detection` or `text_recognition`.
- `status`: `not_run`, `complete`, or `failed`. Only complete records are scored.
- `prediction`: `null` for `not_run`/`failed`; for a complete detection record, `{ "regions": [{ "polygon": [[x, y], ...], "text": "optional raw text", "confidence": 0.0 }] }`; for a complete recognition record, `{ "text": "raw predicted transcription" }`. Omit optional text/confidence when unavailable. Confidence, if supplied, must be in `[0, 1]`.
- `annotation_reference`: `path` copied from the manifest and `record_id` equal to the image filename (for example, `0001.jpg`).
- `uncertainty`: `{ "level": "none|low|medium|high|unknown", "confidence": null or a number from 0 to 1, "notes": "..." }`.
- `errors`: array of `{ "code": "...", "message": "..." }`. Use an empty array when the run completed without recorded errors; record malformed output, decode failures, or inference errors explicitly.

The following Draft 2020-12 JSON Schema validates record shape. The run-state rule (`not_run`/`failed` must have `prediction: null`; `complete` must have the task-appropriate prediction object) is also a required semantic check for the consumer.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "required": ["sample_id", "task_type", "status", "prediction", "annotation_reference", "uncertainty", "errors"],
  "properties": {
    "sample_id": { "type": "string", "minLength": 1 },
    "task_type": { "enum": ["text_detection", "text_recognition"] },
    "status": { "enum": ["not_run", "complete", "failed"] },
    "prediction": { "oneOf": [{ "type": "null" }, { "$ref": "#/$defs/detection_prediction" }, { "$ref": "#/$defs/recognition_prediction" }] },
    "annotation_reference": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path", "record_id"],
      "properties": { "path": { "type": "string" }, "record_id": { "type": "string" } }
    },
    "uncertainty": {
      "type": "object",
      "additionalProperties": false,
      "required": ["level", "confidence", "notes"],
      "properties": {
        "level": { "enum": ["none", "low", "medium", "high", "unknown"] },
        "confidence": { "type": ["number", "null"], "minimum": 0, "maximum": 1 },
        "notes": { "type": "string" }
      }
    },
    "errors": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["code", "message"],
        "properties": { "code": { "type": "string" }, "message": { "type": "string" } }
      }
    }
  },
  "$defs": {
    "point": {
      "type": "array",
      "prefixItems": [{ "type": "number" }, { "type": "number" }],
      "minItems": 2,
      "maxItems": 2
    },
    "polygon": { "type": "array", "items": { "$ref": "#/$defs/point" }, "minItems": 3 },
    "detection_prediction": {
      "type": "object",
      "additionalProperties": false,
      "required": ["regions"],
      "properties": {
        "regions": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["polygon"],
            "properties": {
              "polygon": { "$ref": "#/$defs/polygon" },
              "text": { "type": "string" },
              "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
            }
          }
        }
      }
    },
    "recognition_prediction": {
      "type": "object",
      "additionalProperties": false,
      "required": ["text"],
      "properties": { "text": { "type": "string" } }
    }
  }
}
```

Template records below demonstrate `not_run` only. They are not predictions, must not be scored, and contain no model output:

```json
{"sample_id":"det-test-0001","task_type":"text_detection","status":"not_run","prediction":null,"annotation_reference":{"path":"Food Packaging OCR Dataset/Food Packaging OCR Dataset/det/test/labels.txt","record_id":"0001.jpg"},"uncertainty":{"level":"unknown","confidence":null,"notes":"No prediction supplied; template only."},"errors":[]}
{"sample_id":"rec-test-0001","task_type":"text_recognition","status":"not_run","prediction":null,"annotation_reference":{"path":"Food Packaging OCR Dataset/Food Packaging OCR Dataset/rec/test/labels.txt","record_id":"0001.jpg"},"uncertainty":{"level":"unknown","confidence":null,"notes":"No prediction supplied; template only."},"errors":[]}
```

## Connecting Member 1's extractor

The evaluator needs Member 1 to provide the JSONL records above for the ten manifest samples that the extractor can run. At minimum, return `sample_id`, `task_type`, `status`, task-specific raw `prediction`, `annotation_reference`, `uncertainty`, and `errors`. Detection must return all predicted polygons in original-image pixel coordinates; recognition must return the raw transcription for each crop. If the extractor cannot produce detection polygons, detection cannot be scored; do not substitute text-only OCR output. If only one task is supported, emit records for that task and leave the other task unrun. No Gemma inference or accuracy calculation has been performed as part of this handoff.