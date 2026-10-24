"""Synthetic test fixtures only; these are not real images or model results."""

import json
from pathlib import Path

import pytest

from evaluate_real_ocr import (
    DETECTION_ANNOTATION,
    RECOGNITION_ANNOTATION,
    edit_distance,
    evaluate_records,
    maximum_iou_assignment,
    main,
    normalize_text,
    parse_jsonl,
    polygon_iou,
    score_detection,
)


RECTANGLE = [[0, 0], [10, 0], [10, 10], [0, 10]]
OUTSIDE_RECTANGLE = [[20, 20], [30, 20], [30, 30], [20, 30]]


def _entry(tmp_path, sample_id, task_type, annotation):
    task_folder = "det" if task_type == "text_detection" else "rec"
    filename = f"{sample_id}.jpg"
    image_path = Path("fixture") / task_folder / "test" / "images" / filename
    labels_path = Path("fixture") / task_folder / "test" / "labels.txt"
    image_file = tmp_path / image_path
    labels_file = tmp_path / labels_path
    image_file.parent.mkdir(parents=True, exist_ok=True)
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_bytes(b"synthetic test fixture; not an image result")
    labels_file.write_text(f"{filename}\t{annotation}\n", encoding="utf-8")
    return {
        "image_id": sample_id,
        "source_dataset": "synthetic test fixture",
        "original_relative_path": image_path.as_posix(),
        "split": "test",
        "annotation_path": labels_path.as_posix(),
        "annotation_type": DETECTION_ANNOTATION if task_type == "text_detection" else RECOGNITION_ANNOTATION,
        "expected_ground_truth": {},
        "selection_reason": "test fixture only",
        "uncertainty_notes": "synthetic test fixture; not a real model result",
        "source_provenance": "test fixture",
        "source_version": "test fixture",
        "license_status": "test fixture",
    }


def _record(entry, task_type, prediction, status="complete"):
    return {
        "sample_id": entry["image_id"],
        "task_type": task_type,
        "status": status,
        "prediction": prediction,
        "annotation_reference": {
            "path": entry["annotation_path"],
            "record_id": Path(entry["original_relative_path"]).name,
        },
        "uncertainty": {"level": "unknown", "confidence": None, "notes": "test fixture"},
        "errors": [],
    }


def _manifest(entries):
    return {
        "manifest_id": "synthetic-fixture-only",
        "dataset": {"name": "synthetic test fixture"},
        "limitations": ["Synthetic unit-test fixtures; not real results."],
        "entries": entries,
    }


def test_protocol_normalization_preserves_punctuation_digits_and_unicode():
    assert normalize_text("  Cafe\u0301\t100%.  ") == "Café 100%."
    assert normalize_text("A-1, B!") == "A-1, B!"


def test_recognition_exact_match_keeps_raw_and_normalized_text(tmp_path):
    reference = "Cafe\u0301 100%."
    entry = _entry(tmp_path, "rec-exact-fixture", "text_recognition", reference)
    record = _record(entry, "text_recognition", {"text": "Café\t100%."})

    report = evaluate_records(_manifest([entry]), [(1, record)], tmp_path)
    metrics = report["metrics"]["text_recognition"]
    sample = metrics["samples"][0]

    assert metrics["exact_matches"] == 1
    assert metrics["exact_match_accuracy"] == 1
    assert sample["raw_reference"] == reference
    assert sample["raw_prediction"] == "Café\t100%."
    assert sample["normalized_reference"] == sample["normalized_prediction"] == "Café 100%."


@pytest.mark.parametrize(
    ("reference", "prediction", "expected_distance"),
    [("cat", "cut", 1), ("cat", "cart", 1), ("cart", "cat", 1)],
    ids=["substitution", "insertion", "deletion"],
)
def test_edit_distance_substitution_insertion_deletion(reference, prediction, expected_distance):
    assert edit_distance(reference, prediction) == expected_distance


def test_recognition_reports_cer_and_wer_with_reference_denominators(tmp_path):
    entry = _entry(tmp_path, "rec-wer-fixture", "text_recognition", "one two")
    record = _record(entry, "text_recognition", {"text": "one too"})

    metrics = evaluate_records(_manifest([entry]), [(1, record)], tmp_path)["metrics"]["text_recognition"]

    assert metrics["character_edits"] == 1
    assert metrics["reference_characters"] == len("one two")
    assert metrics["cer"] == pytest.approx(1 / 7)
    assert metrics["word_edits"] == 1
    assert metrics["reference_words"] == 2
    assert metrics["wer"] == pytest.approx(0.5)


def test_polygon_iou_and_one_to_one_matching():
    assert polygon_iou(RECTANGLE, RECTANGLE) == pytest.approx(1.0)
    result = score_detection(
        [{"polygon": RECTANGLE}, {"polygon": RECTANGLE}],
        [{"points": RECTANGLE}],
    )

    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["matches"] == [{"prediction_index": 0, "ground_truth_index": 0, "iou": 1.0}]


def test_assignment_uses_maximum_total_iou_and_deterministic_ties():
    assert maximum_iou_assignment([[0.9, 0.8], [0.85, 0.1]]) == [
        (0, 1, 0.8),
        (1, 0, 0.85),
    ]
    assert maximum_iou_assignment([[0.75], [0.75]]) == [(0, 0, 0.75)]


def test_concave_polygon_is_supported():
    concave = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]]
    assert polygon_iou(concave, concave) == pytest.approx(1.0)


def test_unmatched_and_empty_detection_regions_are_safe():
    unmatched = score_detection([{"polygon": OUTSIDE_RECTANGLE}], [{"points": RECTANGLE}])
    empty_predictions = score_detection([], [{"points": RECTANGLE}])
    empty_ground_truth = score_detection([{"polygon": RECTANGLE}], [])
    both_empty = score_detection([], [])

    assert (unmatched["tp"], unmatched["fp"], unmatched["fn"]) == (0, 1, 1)
    assert (empty_predictions["tp"], empty_predictions["fp"], empty_predictions["fn"]) == (0, 0, 1)
    assert (empty_ground_truth["tp"], empty_ground_truth["fp"], empty_ground_truth["fn"]) == (0, 1, 0)
    assert (both_empty["tp"], both_empty["fp"], both_empty["fn"]) == (0, 0, 0)


def test_invalid_prediction_polygon_is_false_positive_and_error():
    result = score_detection([{"polygon": [[0, 0], [1, 1]]}], [])

    assert result["fp"] == 1
    assert result["errors"][0]["code"] == "invalid_prediction_polygon"


def test_invalid_ground_truth_polygon_blocks_image():
    with pytest.raises(ValueError, match="ground-truth region 0"):
        score_detection([], [{"points": [[0, 0], [1, 1]]}])


def test_duplicate_ids_and_missing_predictions_are_reported(tmp_path):
    recognition_entry = _entry(tmp_path, "rec-duplicate-fixture", "text_recognition", "word")
    detection_entry = _entry(
        tmp_path,
        "det-missing-fixture",
        "text_detection",
        json.dumps([{"transcription": "word", "points": RECTANGLE}]),
    )
    duplicate = _record(recognition_entry, "text_recognition", {"text": "word"})

    report = evaluate_records(
        _manifest([recognition_entry, detection_entry]),
        [(1, duplicate), (2, duplicate)],
        tmp_path,
    )

    assert report["counts"]["duplicate_sample_ids"] == ["rec-duplicate-fixture"]
    assert report["missing_sample_ids"] == ["det-missing-fixture"]
    assert report["metrics"]["text_recognition"]["invalid_samples"] == 1
    assert report["metrics"]["text_detection"]["missing_samples"] == 1


def test_malformed_jsonl_is_reported_and_valid_records_continue(tmp_path):
    entry = _entry(tmp_path, "rec-jsonl-fixture", "text_recognition", "word")
    record = _record(entry, "text_recognition", {"text": "word"})
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("{bad json}\n" + json.dumps(record) + "\n", encoding="utf-8")

    records, errors = parse_jsonl(predictions_path)
    report = evaluate_records(_manifest([entry]), records, tmp_path, errors)

    assert report["input_errors"][0]["line"] == 1
    assert report["metrics"]["text_recognition"]["scored_samples"] == 1


def test_mismatched_task_and_missing_fields_are_invalid(tmp_path):
    entry = _entry(tmp_path, "rec-invalid-fixture", "text_recognition", "word")
    record = _record(entry, "text_detection", {"regions": []})
    record.pop("errors")

    report = evaluate_records(_manifest([entry]), [(1, record)], tmp_path)

    assert report["metrics"]["text_recognition"]["invalid_samples"] == 1
    issues = report["invalid_records"][0]["issues"]
    assert any("missing required fields" in issue for issue in issues)
    assert any("task_type must be 'text_recognition'" in issue for issue in issues)


def test_not_run_and_failed_predictions_are_not_scored(tmp_path):
    not_run_entry = _entry(tmp_path, "rec-not-run-fixture", "text_recognition", "word")
    failed_entry = _entry(tmp_path, "rec-failed-fixture", "text_recognition", "word")
    records = [
        (1, _record(not_run_entry, "text_recognition", None, status="not_run")),
        (2, _record(failed_entry, "text_recognition", None, status="failed")),
    ]

    metrics = evaluate_records(_manifest([not_run_entry, failed_entry]), records, tmp_path)["metrics"]["text_recognition"]

    assert metrics["scored_samples"] == 0
    assert metrics["not_run_samples"] == 1
    assert metrics["failed_samples"] == 1
    assert metrics["exact_match_accuracy"] is None


def test_cli_without_predictions_prints_usage_example(capsys):
    with pytest.raises(SystemExit) as error:
        main([])

    output = capsys.readouterr().err
    assert error.value.code == 2
    assert "genuine predictions JSONL file is required" in output
    assert '"task_type":"text_recognition"' in output


def test_cli_help_includes_dataset_root_for_local_dataset(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    output = capsys.readouterr().out
    assert error.value.code == 0
    assert "--dataset-root" in output
