"""Evaluate genuine OCR predictions against the selected real OCR annotations."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "sample_data" / "real_evaluation_manifest.json"
DETECTION_ANNOTATION = "text_detection_polygons_with_transcriptions"
RECOGNITION_ANNOTATION = "single_text_transcription"
TASK_DETECTION = "text_detection"
TASK_RECOGNITION = "text_recognition"
IOU_THRESHOLD = 0.50
POINT_EPSILON = 1e-12

RECORD_FIELDS = {
    "sample_id",
    "task_type",
    "status",
    "prediction",
    "annotation_reference",
    "uncertainty",
    "errors",
}


class PolygonError(ValueError):
    """Raised when a polygon cannot be evaluated safely."""


def normalize_text(value: str) -> str:
    """Apply only the normalization documented in the evaluation protocol."""
    normalized = unicodedata.normalize("NFC", value)
    return re.sub(r"\s+", " ", normalized.strip())


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    """Return Levenshtein distance using linear memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _cross(first: tuple[float, float], second: tuple[float, float], third: tuple[float, float]) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _signed_area(polygon: list[tuple[float, float]]) -> float:
    return sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    ) / 2.0


def _clean_polygon(points: Any) -> list[tuple[float, float]]:
    if not isinstance(points, (list, tuple)):
        raise PolygonError("polygon must be an array of [x, y] points")

    polygon: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise PolygonError("each polygon point must contain exactly two coordinates")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point):
            raise PolygonError("polygon coordinates must be numbers")
        x_value, y_value = float(point[0]), float(point[1])
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise PolygonError("polygon coordinates must be finite")
        normalized_point = (x_value, y_value)
        if not polygon or normalized_point != polygon[-1]:
            polygon.append(normalized_point)
    if len(polygon) > 1 and polygon[0] == polygon[-1]:
        polygon.pop()

    if len(set(polygon)) < 3:
        raise PolygonError("polygon must have at least three distinct points")
    if len(polygon) < 3 or _signed_area(polygon) == 0:
        raise PolygonError("polygon must have non-zero area")
    if _has_self_intersection(polygon):
        raise PolygonError("polygon must be a simple, non-self-intersecting polygon")
    _triangulate(polygon)
    return polygon


def _orientation(first: tuple[float, float], second: tuple[float, float], third: tuple[float, float]) -> int:
    value = _cross(first, second, third)
    if abs(value) <= POINT_EPSILON:
        return 0
    return 1 if value > 0 else -1


def _on_segment(
    first: tuple[float, float], point: tuple[float, float], second: tuple[float, float]
) -> bool:
    return (
        min(first[0], second[0]) - POINT_EPSILON <= point[0] <= max(first[0], second[0]) + POINT_EPSILON
        and min(first[1], second[1]) - POINT_EPSILON <= point[1] <= max(first[1], second[1]) + POINT_EPSILON
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    if orientations[0] * orientations[1] < 0 and orientations[2] * orientations[3] < 0:
        return True
    return (
        (orientations[0] == 0 and _on_segment(first_start, second_start, first_end))
        or (orientations[1] == 0 and _on_segment(first_start, second_end, first_end))
        or (orientations[2] == 0 and _on_segment(second_start, first_start, second_end))
        or (orientations[3] == 0 and _on_segment(second_start, first_end, second_end))
    )


def _has_self_intersection(polygon: list[tuple[float, float]]) -> bool:
    count = len(polygon)
    for first_index in range(count):
        first_next = (first_index + 1) % count
        for second_index in range(first_index + 1, count):
            second_next = (second_index + 1) % count
            if first_index == second_index or first_next == second_index or second_next == first_index:
                continue
            if _segments_intersect(
                polygon[first_index],
                polygon[first_next],
                polygon[second_index],
                polygon[second_next],
            ):
                return True
    return False


def _point_in_triangle(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
    winding: float,
) -> bool:
    return all(
        _cross(edge_start, edge_end, point) * winding >= -POINT_EPSILON
        for edge_start, edge_end in ((first, second), (second, third), (third, first))
    )


def _triangulate(polygon: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Triangulate a simple polygon with ear clipping for general polygon IoU."""
    winding = 1.0 if _signed_area(polygon) > 0 else -1.0
    remaining = list(range(len(polygon)))
    triangles: list[list[tuple[float, float]]] = []

    while len(remaining) > 3:
        ear_found = False
        for position, current_index in enumerate(remaining):
            previous_index = remaining[position - 1]
            next_index = remaining[(position + 1) % len(remaining)]
            previous_point = polygon[previous_index]
            current_point = polygon[current_index]
            next_point = polygon[next_index]
            if _cross(previous_point, current_point, next_point) * winding <= POINT_EPSILON:
                continue
            if any(
                _point_in_triangle(
                    polygon[other_index], previous_point, current_point, next_point, winding
                )
                for other_index in remaining
                if other_index not in {previous_index, current_index, next_index}
            ):
                continue
            triangles.append([previous_point, current_point, next_point])
            del remaining[position]
            ear_found = True
            break
        if not ear_found:
            raise PolygonError("polygon could not be triangulated safely")

    final_triangle = [polygon[index] for index in remaining]
    if abs(_signed_area(final_triangle)) <= POINT_EPSILON:
        raise PolygonError("polygon triangulation produced a degenerate triangle")
    triangles.append(final_triangle)
    return triangles


def _clip_convex_polygon(
    subject: list[tuple[float, float]], clipper: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    output = subject
    winding = 1.0 if _signed_area(clipper) > 0 else -1.0
    for edge_index, edge_start in enumerate(clipper):
        edge_end = clipper[(edge_index + 1) % len(clipper)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_distance = _cross(edge_start, edge_end, previous) * winding
        for current in input_points:
            current_distance = _cross(edge_start, edge_end, current) * winding
            previous_inside = previous_distance >= -POINT_EPSILON
            current_inside = current_distance >= -POINT_EPSILON
            if previous_inside != current_inside:
                denominator = previous_distance - current_distance
                if denominator != 0:
                    fraction = previous_distance / denominator
                    output.append(
                        (
                            previous[0] + fraction * (current[0] - previous[0]),
                            previous[1] + fraction * (current[1] - previous[1]),
                        )
                    )
            if current_inside:
                output.append(current)
            previous = current
            previous_distance = current_distance
    return output


def polygon_iou(first_points: Any, second_points: Any) -> float:
    """Calculate intersection-over-union for simple, possibly concave polygons."""
    first = _clean_polygon(first_points)
    second = _clean_polygon(second_points)
    first_area = abs(_signed_area(first))
    second_area = abs(_signed_area(second))
    first_triangles = _triangulate(first)
    second_triangles = _triangulate(second)
    intersection_area = 0.0
    for first_triangle in first_triangles:
        for second_triangle in second_triangles:
            clipped = _clip_convex_polygon(first_triangle, second_triangle)
            if len(clipped) >= 3:
                intersection_area += abs(_signed_area(clipped))
    union_area = first_area + second_area - intersection_area
    if union_area <= 0 or not math.isfinite(union_area):
        raise PolygonError("polygon union area is invalid")
    return min(1.0, max(0.0, intersection_area / union_area))


def _pair_add(first: tuple[float, int], second: tuple[float, int]) -> tuple[float, int]:
    return first[0] + second[0], first[1] + second[1]


def _pair_subtract(first: tuple[float, int], second: tuple[float, int]) -> tuple[float, int]:
    return first[0] - second[0], first[1] - second[1]


def maximum_iou_assignment(
    iou_matrix: list[list[float]], threshold: float = IOU_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Return maximum-total-IoU eligible pairs with deterministic lexicographic ties."""
    prediction_count = len(iou_matrix)
    if prediction_count == 0:
        return []
    ground_truth_count = len(iou_matrix[0])
    if any(len(row) != ground_truth_count for row in iou_matrix):
        raise ValueError("IoU matrix must be rectangular")
    if ground_truth_count == 0:
        return []

    column_count = ground_truth_count + prediction_count
    tie_base = ground_truth_count + 1
    costs: list[list[tuple[float, int]]] = []
    for prediction_index, row in enumerate(iou_matrix):
        cost_row = []
        tie_place = tie_base ** (prediction_count - prediction_index - 1)
        for ground_truth_index, overlap in enumerate(row):
            if not math.isfinite(overlap) or overlap < threshold:
                cost_row.append((1.0, 0))
            else:
                tie_preference = (ground_truth_count - ground_truth_index) * tie_place
                cost_row.append((-overlap, -tie_preference))
        cost_row.extend([(0.0, 0)] * prediction_count)
        costs.append(cost_row)

    zero = (0.0, 0)
    row_potential = [zero for _ in range(prediction_count + 1)]
    column_potential = [zero for _ in range(column_count + 1)]
    column_row = [0] * (column_count + 1)
    predecessor_column = [0] * (column_count + 1)

    for row_number in range(1, prediction_count + 1):
        column_row[0] = row_number
        current_column = 0
        minimum_cost: list[tuple[float, int] | None] = [None] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = column_row[current_column]
            delta: tuple[float, int] | None = None
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = _pair_subtract(
                    _pair_subtract(costs[current_row - 1][candidate_column - 1], row_potential[current_row]),
                    column_potential[candidate_column],
                )
                if minimum_cost[candidate_column] is None or reduced_cost < minimum_cost[candidate_column]:
                    minimum_cost[candidate_column] = reduced_cost
                    predecessor_column[candidate_column] = current_column
                if delta is None or minimum_cost[candidate_column] < delta:
                    delta = minimum_cost[candidate_column]
                    next_column = candidate_column
            if delta is None:
                raise ValueError("unable to complete polygon assignment")
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    assigned_row = column_row[candidate_column]
                    row_potential[assigned_row] = _pair_add(row_potential[assigned_row], delta)
                    column_potential[candidate_column] = _pair_subtract(
                        column_potential[candidate_column], delta
                    )
                elif minimum_cost[candidate_column] is not None:
                    minimum_cost[candidate_column] = _pair_subtract(
                        minimum_cost[candidate_column], delta
                    )
            current_column = next_column
            if column_row[current_column] == 0:
                break
        while True:
            previous_column = predecessor_column[current_column]
            column_row[current_column] = column_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    row_column = [0] * prediction_count
    for column_index in range(1, column_count + 1):
        if column_row[column_index] != 0:
            row_column[column_row[column_index] - 1] = column_index

    matches = []
    for prediction_index, column_index in enumerate(row_column):
        ground_truth_index = column_index - 1
        if (
            0 <= ground_truth_index < ground_truth_count
            and iou_matrix[prediction_index][ground_truth_index] >= threshold
        ):
            matches.append(
                (prediction_index, ground_truth_index, iou_matrix[prediction_index][ground_truth_index])
            )
    return matches


def score_detection(
    predicted_regions: list[dict[str, Any]],
    ground_truth_regions: list[dict[str, Any]],
    threshold: float = IOU_THRESHOLD,
) -> dict[str, Any]:
    ground_truth_polygons = []
    for region_index, region in enumerate(ground_truth_regions):
        try:
            ground_truth_polygons.append(_clean_polygon(region["points"]))
        except (KeyError, TypeError, PolygonError) as error:
            raise PolygonError(f"ground-truth region {region_index}: {error}") from error

    valid_predictions: list[tuple[int, list[tuple[float, float]]]] = []
    prediction_errors: list[dict[str, Any]] = []
    invalid_prediction_count = 0
    for prediction_index, region in enumerate(predicted_regions):
        try:
            polygon = _clean_polygon(region.get("polygon"))
            valid_predictions.append((prediction_index, polygon))
        except (AttributeError, PolygonError) as error:
            invalid_prediction_count += 1
            prediction_errors.append(
                {
                    "code": "invalid_prediction_polygon",
                    "message": f"Prediction region {prediction_index}: {error}",
                }
            )

    iou_matrix = [
        [polygon_iou(prediction, ground_truth) for ground_truth in ground_truth_polygons]
        for _, prediction in valid_predictions
    ]
    matched = maximum_iou_assignment(iou_matrix, threshold)
    true_positives = len(matched)
    false_positives = invalid_prediction_count + len(predicted_regions) - invalid_prediction_count - true_positives
    false_negatives = len(ground_truth_polygons) - true_positives
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "matches": [
            {
                "prediction_index": valid_predictions[prediction_index][0],
                "ground_truth_index": ground_truth_index,
                "iou": overlap,
            }
            for prediction_index, ground_truth_index, overlap in matched
        ],
        "errors": prediction_errors,
    }


def _expected_task(entry: dict[str, Any]) -> str:
    if entry.get("annotation_type") == DETECTION_ANNOTATION:
        return TASK_DETECTION
    if entry.get("annotation_type") == RECOGNITION_ANNOTATION:
        return TASK_RECOGNITION
    raise ValueError(f"unsupported manifest annotation type: {entry.get('annotation_type')}")


def _read_ground_truth(entry: dict[str, Any], root: Path) -> Any:
    annotation_path = root / entry["annotation_path"]
    filename = Path(entry["original_relative_path"]).name
    try:
        matching_rows = [
            line
            for line in annotation_path.read_text(encoding="utf-8").splitlines()
            if line.startswith(filename + "\t")
        ]
    except OSError as error:
        raise ValueError(f"cannot read annotation file {annotation_path}: {error}") from error
    if len(matching_rows) != 1:
        raise ValueError(f"expected one annotation row for {filename}; found {len(matching_rows)}")
    payload = matching_rows[0].split("\t", 1)[1]
    if entry["annotation_type"] == DETECTION_ANNOTATION:
        try:
            regions = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed detection annotation for {filename}: {error}") from error
        if not isinstance(regions, list):
            raise ValueError(f"detection annotation for {filename} is not a list")
        return regions
    return payload


def _validate_record(record: Any, manifest_entry: dict[str, Any]) -> list[str]:
    if not isinstance(record, dict):
        return ["record must be a JSON object"]
    issues = []
    missing_fields = RECORD_FIELDS - record.keys()
    extra_fields = record.keys() - RECORD_FIELDS
    if missing_fields:
        issues.append(f"missing required fields: {', '.join(sorted(missing_fields))}")
    if extra_fields:
        issues.append(f"unexpected fields: {', '.join(sorted(extra_fields))}")
    if "task_type" in record:
        expected_task = _expected_task(manifest_entry)
        if record["task_type"] != expected_task:
            issues.append(f"task_type must be {expected_task!r}")
    if "status" in record and (
        not isinstance(record["status"], str)
        or record["status"] not in {"not_run", "complete", "failed"}
    ):
        issues.append("status must be 'not_run', 'complete', or 'failed'")
    if issues:
        return issues

    expected_task = _expected_task(manifest_entry)
    if record["status"] != "complete":
        if record["prediction"] is not None:
            issues.append("prediction must be null unless status is 'complete'")
    elif record["task_type"] == TASK_DETECTION:
        prediction = record["prediction"]
        if not isinstance(prediction, dict) or set(prediction) != {"regions"}:
            issues.append("detection prediction must be an object containing only 'regions'")
        elif not isinstance(prediction["regions"], list):
            issues.append("detection prediction 'regions' must be an array")
        else:
            for region_index, region in enumerate(prediction["regions"]):
                if not isinstance(region, dict) or "polygon" not in region:
                    issues.append(f"detection region {region_index} must contain 'polygon'")
                    continue
                if set(region) - {"polygon", "text", "confidence"}:
                    issues.append(f"detection region {region_index} contains unexpected fields")
                if "text" in region and not isinstance(region["text"], str):
                    issues.append(f"detection region {region_index} text must be a string")
                if "confidence" in region and not _valid_confidence(region["confidence"]):
                    issues.append(f"detection region {region_index} confidence must be between 0 and 1")
    elif record["task_type"] == TASK_RECOGNITION:
        if not isinstance(record["prediction"], dict) or set(record["prediction"]) != {"text"}:
            issues.append("recognition prediction must be an object containing only 'text'")
        elif not isinstance(record["prediction"]["text"], str):
            issues.append("recognition prediction text must be a string")

    expected_reference = {
        "path": manifest_entry["annotation_path"],
        "record_id": Path(manifest_entry["original_relative_path"]).name,
    }
    if record["annotation_reference"] != expected_reference:
        issues.append(f"annotation_reference must equal {expected_reference}")

    uncertainty = record["uncertainty"]
    if (
        not isinstance(uncertainty, dict)
        or set(uncertainty) != {"level", "confidence", "notes"}
        or not isinstance(uncertainty.get("level"), str)
        or uncertainty.get("level") not in {"none", "low", "medium", "high", "unknown"}
        or not _valid_confidence(uncertainty.get("confidence"), allow_none=True)
        or not isinstance(uncertainty.get("notes"), str)
    ):
        issues.append("uncertainty must contain valid level, confidence, and notes fields")

    errors = record["errors"]
    if not isinstance(errors, list) or any(
        not isinstance(item, dict)
        or set(item) != {"code", "message"}
        or not isinstance(item.get("code"), str)
        or not isinstance(item.get("message"), str)
        for item in errors
    ):
        issues.append("errors must be an array of {code, message} objects")
    return issues


def _valid_confidence(value: Any, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _empty_task_counts(expected_samples: int) -> dict[str, Any]:
    return {
        "expected_samples": expected_samples,
        "scored_samples": 0,
        "failed_samples": 0,
        "not_run_samples": 0,
        "missing_samples": 0,
        "invalid_samples": 0,
        "ground_truth_errors": [],
    }


def evaluate_records(
    manifest: dict[str, Any],
    prediction_records: list[tuple[int, Any]],
    dataset_root: Path = ROOT,
    parse_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate parsed JSONL records; record tuples contain input line and JSON value."""
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("manifest must contain an entries array")
    manifest_by_id = {entry["image_id"]: entry for entry in entries}
    if len(manifest_by_id) != len(entries):
        raise ValueError("manifest contains duplicate image_id values")

    grouped: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    invalid_records: list[dict[str, Any]] = []
    for line_number, record in prediction_records:
        if not isinstance(record, dict):
            invalid_records.append({"line": line_number, "sample_id": None, "issues": ["record must be a JSON object"]})
            continue
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in manifest_by_id:
            invalid_records.append(
                {
                    "line": line_number,
                    "sample_id": sample_id if isinstance(sample_id, str) else None,
                    "issues": ["sample_id is missing or is not present in the real evaluation manifest"],
                }
            )
            continue
        grouped[sample_id].append((line_number, record))

    duplicate_ids = {sample_id for sample_id, records in grouped.items() if len(records) > 1}
    invalid_sample_ids: set[str] = set(duplicate_ids)
    for sample_id in sorted(duplicate_ids):
        lines = [line_number for line_number, _ in grouped[sample_id]]
        invalid_records.append(
            {"line": lines[0], "sample_id": sample_id, "issues": [f"duplicate predictions on lines {lines}"]}
        )

    valid_records: dict[str, tuple[int, dict[str, Any]]] = {}
    for sample_id, records in grouped.items():
        if sample_id in duplicate_ids:
            continue
        line_number, record = records[0]
        issues = _validate_record(record, manifest_by_id[sample_id])
        if issues:
            invalid_records.append({"line": line_number, "sample_id": sample_id, "issues": issues})
            invalid_sample_ids.add(sample_id)
        else:
            valid_records[sample_id] = (line_number, record)

    detection_entries = [entry for entry in entries if entry["annotation_type"] == DETECTION_ANNOTATION]
    recognition_entries = [entry for entry in entries if entry["annotation_type"] == RECOGNITION_ANNOTATION]
    detection_report = _empty_task_counts(len(detection_entries))
    detection_report.update(
        {
            "iou_threshold": IOU_THRESHOLD,
            "matching": "maximum-total-IoU one-to-one assignment among pairs with IoU >= threshold; exact ties prefer earlier prediction then earlier ground-truth order",
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "samples": [],
        }
    )
    recognition_report = _empty_task_counts(len(recognition_entries))
    recognition_report.update(
        {
            "normalization": "Unicode NFC; trim leading/trailing whitespace; collapse each whitespace run to one ASCII space; preserve case, punctuation, digits, and diacritics",
            "exact_matches": 0,
            "exact_match_accuracy": None,
            "character_edits": 0,
            "reference_characters": 0,
            "cer": None,
            "word_edits": 0,
            "reference_words": 0,
            "wer": None,
            "samples": [],
        }
    )

    missing_ids = set(manifest_by_id) - set(grouped)
    task_reports = {TASK_DETECTION: detection_report, TASK_RECOGNITION: recognition_report}
    for entry in entries:
        sample_id = entry["image_id"]
        task_type = _expected_task(entry)
        task_report = task_reports[task_type]
        if sample_id in missing_ids:
            task_report["missing_samples"] += 1
            task_report["samples"].append({"sample_id": sample_id, "status": "missing_prediction"})
            continue
        if sample_id in invalid_sample_ids:
            task_report["invalid_samples"] += 1
            task_report["samples"].append({"sample_id": sample_id, "status": "invalid_prediction_record"})
            continue
        record = valid_records[sample_id][1]
        if record["status"] == "not_run":
            task_report["not_run_samples"] += 1
            task_report["samples"].append({"sample_id": sample_id, "status": "not_run"})
            continue
        if record["status"] == "failed":
            task_report["failed_samples"] += 1
            task_report["samples"].append(
                {"sample_id": sample_id, "status": "failed", "errors": record["errors"]}
            )
            continue

        try:
            ground_truth = _read_ground_truth(entry, dataset_root)
            image_path = dataset_root / entry["original_relative_path"]
            if not image_path.is_file():
                raise ValueError(f"image does not exist: {image_path}")
        except (KeyError, OSError, ValueError) as error:
            task_report["ground_truth_errors"].append({"sample_id": sample_id, "error": str(error)})
            task_report["invalid_samples"] += 1
            task_report["samples"].append(
                {"sample_id": sample_id, "status": "ground_truth_error", "error": str(error)}
            )
            continue

        if task_type == TASK_DETECTION:
            prediction_regions = record["prediction"]["regions"]
            try:
                scored = score_detection(prediction_regions, ground_truth, IOU_THRESHOLD)
            except PolygonError as error:
                task_report["ground_truth_errors"].append({"sample_id": sample_id, "error": str(error)})
                task_report["invalid_samples"] += 1
                task_report["samples"].append(
                    {"sample_id": sample_id, "status": "ground_truth_error", "error": str(error)}
                )
                continue
            task_report["scored_samples"] += 1
            task_report["tp"] += scored["tp"]
            task_report["fp"] += scored["fp"]
            task_report["fn"] += scored["fn"]
            task_report["samples"].append(
                {
                    "sample_id": sample_id,
                    "status": "scored",
                    "tp": scored["tp"],
                    "fp": scored["fp"],
                    "fn": scored["fn"],
                    "matches": scored["matches"],
                    "prediction_errors": scored["errors"],
                    "source_errors": record["errors"],
                }
            )
        else:
            raw_reference = ground_truth
            raw_prediction = record["prediction"]["text"]
            normalized_reference = normalize_text(raw_reference)
            normalized_prediction = normalize_text(raw_prediction)
            character_edits = edit_distance(normalized_reference, normalized_prediction)
            reference_words = normalized_reference.split() if normalized_reference else []
            prediction_words = normalized_prediction.split() if normalized_prediction else []
            word_edits = edit_distance(reference_words, prediction_words)
            exact_match = normalized_reference == normalized_prediction
            task_report["scored_samples"] += 1
            task_report["exact_matches"] += int(exact_match)
            task_report["character_edits"] += character_edits
            task_report["reference_characters"] += len(normalized_reference)
            task_report["word_edits"] += word_edits
            task_report["reference_words"] += len(reference_words)
            task_report["samples"].append(
                {
                    "sample_id": sample_id,
                    "status": "scored",
                    "raw_reference": raw_reference,
                    "raw_prediction": raw_prediction,
                    "normalized_reference": normalized_reference,
                    "normalized_prediction": normalized_prediction,
                    "exact_match": exact_match,
                    "character_edits": character_edits,
                    "reference_characters": len(normalized_reference),
                    "cer": _safe_ratio(character_edits, len(normalized_reference)),
                    "word_edits": word_edits,
                    "reference_words": len(reference_words),
                    "wer": _safe_ratio(word_edits, len(reference_words)),
                    "source_errors": record["errors"],
                }
            )

    detection_report["precision"] = _safe_ratio(
        detection_report["tp"], detection_report["tp"] + detection_report["fp"]
    )
    detection_report["recall"] = _safe_ratio(
        detection_report["tp"], detection_report["tp"] + detection_report["fn"]
    )
    if detection_report["precision"] is not None and detection_report["recall"] is not None:
        detection_report["f1"] = _safe_ratio(
            2 * detection_report["precision"] * detection_report["recall"],
            detection_report["precision"] + detection_report["recall"],
        )
    recognition_report["exact_match_accuracy"] = _safe_ratio(
        recognition_report["exact_matches"], recognition_report["scored_samples"]
    )
    recognition_report["cer"] = _safe_ratio(
        recognition_report["character_edits"], recognition_report["reference_characters"]
    )
    recognition_report["wer"] = _safe_ratio(
        recognition_report["word_edits"], recognition_report["reference_words"]
    )

    return {
        "report_version": 1,
        "manifest_id": manifest.get("manifest_id"),
        "dataset_name": manifest.get("dataset", {}).get("name"),
        "provenance_and_license": {
            "publisher_or_authors": manifest.get("dataset", {}).get("publisher_or_authors", "UNKNOWN"),
            "version": manifest.get("dataset", {}).get("version", "UNKNOWN"),
            "license": manifest.get("dataset", {}).get("license", "UNKNOWN"),
        },
        "scope": "Generic text detection and text recognition only; not product-field extraction or regulatory compliance.",
        "input_errors": parse_errors or [],
        "invalid_records": invalid_records,
        "missing_sample_ids": sorted(missing_ids),
        "counts": {
            "manifest_samples": len(entries),
            "prediction_records": len(prediction_records),
            "invalid_records": len(invalid_records),
            "missing_predictions": len(missing_ids),
            "duplicate_sample_ids": sorted(duplicate_ids),
        },
        "metrics": {
            TASK_DETECTION: detection_report,
            TASK_RECOGNITION: recognition_report,
        },
        "limitations": manifest.get("limitations", []),
    }


def parse_jsonl(path: Path) -> tuple[list[tuple[int, Any]], list[dict[str, Any]]]:
    records: list[tuple[int, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as predictions_file:
        for line_number, line in enumerate(predictions_file, start=1):
            if not line.strip():
                continue
            try:
                records.append((line_number, json.loads(line)))
            except json.JSONDecodeError as error:
                errors.append(
                    {
                        "line": line_number,
                        "error": f"malformed JSON: {error.msg} (column {error.colno})",
                    }
                )
    return records, errors


def _markdown_report(report: dict[str, Any]) -> str:
    detection = report["metrics"][TASK_DETECTION]
    recognition = report["metrics"][TASK_RECOGNITION]

    def display(value: Any) -> str:
        return "n/a" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = [
        "# Real OCR Evaluation Report",
        "",
        f"Dataset: {report['dataset_name']} (`{report['manifest_id']}`)",
        "",
        "This report covers generic OCR only. It is not an evaluation of ingredients, nutrition, expiry dates, product fields, or regulatory compliance. It does not establish Gemma results.",
        "",
        "## Text Detection",
        "",
        f"- IoU threshold: {detection['iou_threshold']:.2f}",
        f"- Matching: {detection['matching']}",
        f"- Scored images: {detection['scored_samples']} / {detection['expected_samples']}; missing: {detection['missing_samples']}; failed: {detection['failed_samples']}; not run: {detection['not_run_samples']}; invalid: {detection['invalid_samples']}",
        f"- TP / FP / FN: {detection['tp']} / {detection['fp']} / {detection['fn']}",
        f"- Precision / recall / F1: {display(detection['precision'])} / {display(detection['recall'])} / {display(detection['f1'])}",
        "",
        "## Text Recognition",
        "",
        f"- Normalization: {recognition['normalization']}.",
        f"- Scored crops: {recognition['scored_samples']} / {recognition['expected_samples']}; missing: {recognition['missing_samples']}; failed: {recognition['failed_samples']}; not run: {recognition['not_run_samples']}; invalid: {recognition['invalid_samples']}",
        f"- Exact-match accuracy: {display(recognition['exact_match_accuracy'])}",
        f"- CER: {display(recognition['cer'])} ({recognition['character_edits']} edits / {recognition['reference_characters']} reference characters)",
        f"- WER: {display(recognition['wer'])} ({recognition['word_edits']} edits / {recognition['reference_words']} reference words)",
        "",
        "## Invalid and Skipped Cases",
        "",
        f"- Invalid records: {report['counts']['invalid_records']}; malformed JSONL lines: {len(report['input_errors'])}.",
        f"- Missing prediction IDs: {', '.join(report['missing_sample_ids']) if report['missing_sample_ids'] else 'none'}.",
        f"- Detection ground-truth blockers: {len(detection['ground_truth_errors'])}; recognition ground-truth blockers: {len(recognition['ground_truth_errors'])}.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend(
        [
            "- Detection and recognition results are separate. The small, manually selected subset is a smoke check, not a representative accuracy estimate.",
            "- Dataset publisher, version, and licence remain UNKNOWN; external demonstration or redistribution permission is unconfirmed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate supplied OCR predictions against the real test subset.",
        epilog=(
            "Example JSONL record (not a prediction): "
            '{"sample_id":"rec-test-0001","task_type":"text_recognition","status":"complete",'
            '"prediction":{"text":"raw output"},"annotation_reference":{"path":".../rec/test/labels.txt",'
            '"record_id":"0001.jpg"},"uncertainty":{"level":"unknown","confidence":null,"notes":""},"errors":[]} '
            "Use genuine model output only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("predictions", nargs="?", help="Member 1 UTF-8 JSONL predictions file")
    parser.add_argument("--output-dir", help="Caller-selected directory for JSON and Markdown reports")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"real evaluation manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT,
        help="Local root containing the manifest-relative dataset paths (default: project root)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.predictions:
        print(parser.format_help(), file=sys.stderr, end="")
        parser.exit(2, "error: a genuine predictions JSONL file is required; see the example schema above\n")
    if not args.output_dir:
        parser.error("--output-dir is required")
    prediction_path = Path(args.predictions)
    if not prediction_path.is_file():
        parser.error(f"predictions file not found: {prediction_path}")
    if not args.manifest.is_file():
        parser.error(f"manifest file not found: {args.manifest}")

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        prediction_records, parse_errors = parse_jsonl(prediction_path)
        report = evaluate_records(manifest, prediction_records, args.dataset_root.expanduser(), parse_errors)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        print(f"Evaluation could not start: {error}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_report_path = output_dir / "real_ocr_report.json"
    markdown_report_path = output_dir / "real_ocr_report.md"
    json_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_report_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"JSON report: {json_report_path}")
    print(f"Markdown report: {markdown_report_path}")
    print(
        "Scored samples: "
        f"detection={report['metrics'][TASK_DETECTION]['scored_samples']}, "
        f"recognition={report['metrics'][TASK_RECOGNITION]['scored_samples']}"
    )
    incomplete = bool(
        report["input_errors"]
        or report["invalid_records"]
        or report["missing_sample_ids"]
        or report["metrics"][TASK_DETECTION]["failed_samples"]
        or report["metrics"][TASK_DETECTION]["not_run_samples"]
        or report["metrics"][TASK_RECOGNITION]["failed_samples"]
        or report["metrics"][TASK_RECOGNITION]["not_run_samples"]
    )
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
