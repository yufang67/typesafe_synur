import json
from copy import deepcopy

import pytest

from synur.observations import SchemaRegistry
from synur.reporting import (
    build_short_transcript_report,
    build_transcript_report,
    save_transcript_report,
)


@pytest.fixture
def registry():
    return SchemaRegistry.from_entries([
        {"id": "1", "name": "Nausea", "value_type": "SINGLE_SELECT", "value_enum": ["Yes", "No"]},
        {"id": "2", "name": "Symptoms", "value_type": "MULTI_SELECT", "value_enum": ["A", "B"]},
        {"id": "3", "name": "Urine output", "value_type": "NUMERIC"},
        {"id": "4", "name": "Therapy", "value_type": "SINGLE_SELECT", "value_enum": ["Yes", "No"]},
    ])


def observation(registry, concept_id, value):
    concept = registry.by_id[concept_id]
    return {"id": concept.id, "name": concept.name, "value_type": concept.value_type, "value": value}


def source(row_id, labels, **kwargs):
    return {"id": row_id, "transcript": f"Synthetic transcript {row_id}.",
            "observations": labels, **kwargs}


def prediction(registry, row_id, labels, **kwargs):
    emitted = {item.get("id") for item in labels if isinstance(item, dict)}
    return {
        "id": row_id, "status": "complete", "observations": labels,
        "audit": [{"id": concept.id, "status": "emitted" if concept.id in emitted else "absent"}
                  for concept in registry.concepts],
        "failures": [],
        **kwargs,
    }


def test_report_tags_and_metrics_are_per_transcript(registry):
    gold = [observation(registry, "1", "No"), observation(registry, "2", ["A"]),
            observation(registry, "3", 150)]
    emitted = [observation(registry, "1", "No"), observation(registry, "2", ["B"]),
               observation(registry, "4", "Yes")]
    rows = [source("mixed", gold, split="dev"),
            source("exact", [gold[2]], split="dev")]
    predictions = [
        prediction(registry, "mixed", emitted),
        prediction(registry, "exact", [observation(registry, "3", 150.0)]),
    ]
    original = deepcopy((rows, predictions))
    report = build_transcript_report(rows, predictions, registry)
    assert report["enabled_value_types"] == ["SINGLE_SELECT", "MULTI_SELECT", "NUMERIC"]
    assert report["reference_view"] == "normalized"
    mixed, exact = report["transcripts"]
    assert all("raw_expected_observations" not in entry for entry in report["transcripts"])
    assert mixed["transcript"] == rows[0]["transcript"]
    assert mixed["split"] == "dev"
    assert mixed["available"]
    assert mixed["error_counts"] == {"COR": 1, "DEL": 1, "INS": 1, "SUB": 1}
    assert mixed["precision"] == mixed["recall"] == mixed["f1"] == 1 / 3
    expected = {item["observation"]["id"]: item for item in mixed["expected_observations"]}
    predicted = {item["observation"]["id"]: item for item in mixed["predicted_observations"]}
    assert {key: item["error_type"] for key, item in expected.items()} == {
        "1": "COR", "2": "SUB", "3": "DEL",
    }
    assert {key: item["error_type"] for key, item in predicted.items()} == {
        "1": "COR", "2": "SUB", "4": "INS",
    }
    for entries, field in ((expected.values(), "expected_observation"),
                           (predicted.values(), "predicted_observation")):
        for item in entries:
            comparison = mixed["comparisons"][item["comparison_index"]]
            assert comparison["error_type"] == item["error_type"]
            assert comparison[field] == item["observation"]
    assert mixed["comparisons"][predicted["4"]["comparison_index"]]["expected_observation"] is None
    assert mixed["comparisons"][expected["3"]["comparison_index"]]["predicted_observation"] is None
    assert expected["2"]["comparison_index"] == predicted["2"]["comparison_index"]
    assert exact["precision"] == exact["recall"] == exact["f1"] == 1
    assert exact["error_counts"] == {"COR": 1, "DEL": 0, "INS": 0, "SUB": 0}
    assert report["micro_metrics"] == {
        "available": True, "transcript_count": 2, "evaluated_transcript_count": 2,
        "skipped_expected_count": 0, "tp": 2, "fp": 2, "fn": 2,
        "precision": 0.5, "recall": 0.5, "f1": 0.5,
    }
    assert list(report)[-1] == "micro_metrics"
    assert (rows, predictions) == original
    assert json.loads(json.dumps(report)) == report
    mixed["expected_observations"][0]["observation"]["value"] = "changed"
    assert (rows, predictions) == original


@pytest.mark.parametrize("reference_view", ["raw", "normalized"])
def test_report_retains_selected_reference_view_and_normalizations(registry, reference_view):
    canonical = observation(registry, "3", 150)
    raw = {**canonical, "id": "003"}
    report = build_transcript_report(
        [source("x", [raw])], [prediction(registry, "x", [canonical])], registry,
        reference_view=reference_view,
    )
    result = report["transcripts"][0]
    assert "raw_expected_observations" not in result
    assert len(result["reference_changes"]) == 1
    if reference_view == "normalized":
        assert result["expected_observations"][0]["observation"] == canonical
        assert result["error_counts"] == {"COR": 1, "DEL": 0, "INS": 0, "SUB": 0}
    else:
        assert result["expected_observations"][0]["observation"] == raw
        assert result["error_counts"] == {"COR": 0, "DEL": 1, "INS": 1, "SUB": 0}


@pytest.mark.parametrize("status", ["missing", "failed"])
def test_unavailable_rows_never_fabricate_errors_or_scores(registry, status):
    label = observation(registry, "1", "No")
    records = [] if status == "missing" else [
        prediction(registry, "x", [label], status="failed", failures=[{"error": "timeout"}])
    ]
    # A different successful row must not make this row appear evaluated.
    records.append(prediction(registry, "ok", [label]))
    result = build_transcript_report(
        [source("x", [label]), source("ok", [label])], records, registry
    )["transcripts"][0]
    assert not result["available"]
    assert result["prediction_status"] == status
    assert result["unavailable_reason"]
    assert result["error_counts"] is None
    assert result["precision"] is result["recall"] is result["f1"] is None
    assert result["comparisons"] == []
    assert result["expected_observations"] == [
        {"observation": label, "error_type": None, "comparison_index": None}
    ]
    if status == "failed":
        assert result["predicted_observations"][0]["error_type"] is None
        assert result["failures"] == [{"error": "timeout"}]
    else:
        assert result["predicted_observations"] == []


def test_successful_empty_predictions_are_deletions(registry):
    label = observation(registry, "1", "No")
    result = build_transcript_report(
        [source("x", [label])], [prediction(registry, "x", [])], registry
    )["transcripts"][0]
    assert result["available"]
    assert result["expected_observations"][0]["error_type"] == "DEL"
    assert result["predicted_observations"] == []
    assert result["precision"] is None
    assert result["recall"] == result["f1"] == 0


def test_partial_predictions_have_scores_and_failures(registry):
    first = observation(registry, "1", "No")
    second = observation(registry, "3", 150)
    result = build_transcript_report(
        [source("x", [first, second])],
        [prediction(registry, "x", [first], status="partial", failures=[{"error": "timeout"}])],
        registry,
    )["transcripts"][0]
    assert result["prediction_status"] == "partial"
    assert result["available"]
    assert result["error_counts"] == {"COR": 1, "DEL": 1, "INS": 0, "SUB": 0}
    assert result["failures"]


@pytest.mark.parametrize("reference_view", ["raw", "normalized"])
@pytest.mark.parametrize("encoded", [False, True])
def test_string_references_are_retained_as_skip_without_affecting_scores(
    registry, reference_view, encoded
):
    label = observation(registry, "1", "No")
    skipped = {"id": "5", "name": "Note", "value_type": "STRING", "value": "Exact original text"}
    labels = [skipped, label, deepcopy(skipped)]
    raw = json.dumps(labels) if encoded else labels
    rows = [source("x", raw)]
    before = deepcopy(rows)
    report = build_transcript_report(
        rows, [prediction(registry, "x", [label])], registry, reference_view=reference_view
    )
    entry = report["transcripts"][0]
    skips = [item for item in entry["expected_observations"] if item["error_type"] == "SKIP"]
    assert len(skips) == entry["skipped_expected_count"] == 2
    assert all(item["observation"] == skipped for item in skips)
    assert all(entry["comparisons"][item["comparison_index"]]["error_type"] == "SKIP"
               for item in skips)
    assert "raw_expected_observations" not in entry
    assert entry["error_counts"] == {"COR": 1, "DEL": 0, "INS": 0, "SUB": 0}
    assert entry["precision"] == entry["recall"] == entry["f1"] == 1
    assert not entry["reference_issues"]
    assert report["excluded_value_types"] == ["STRING"]
    assert report["micro_metrics"]["skipped_expected_count"] == 2
    assert report["micro_metrics"]["tp"] == 1
    assert report["micro_metrics"]["fn"] == 0
    assert report["micro_metrics"]["f1"] == 1
    assert rows == before


def test_skip_labels_are_identified_even_without_predictions(registry):
    skipped = {"id": "5", "name": "Note", "value_type": "STRING", "value": "Text"}
    label = observation(registry, "1", "No")
    report = build_transcript_report([source("x", [label, skipped])], [], registry)
    entry = report["transcripts"][0]
    assert [item["error_type"] for item in entry["expected_observations"]] == [None, "SKIP"]
    assert entry["error_counts"] is None
    assert report["micro_metrics"]["available"] is False
    assert report["micro_metrics"]["skipped_expected_count"] == 1
    assert all(report["micro_metrics"][name] is None
               for name in ("tp", "fp", "fn", "precision", "recall", "f1"))


def test_only_skipped_references_do_not_create_deletions(registry):
    skipped = {"id": "5", "name": "Note", "value_type": "STRING", "value": "Text"}
    report = build_transcript_report(
        [source("x", [skipped])], [prediction(registry, "x", [])], registry
    )
    assert report["transcripts"][0]["error_counts"] == {
        "COR": 0, "DEL": 0, "INS": 0, "SUB": 0
    }
    assert report["micro_metrics"]["available"]
    assert report["micro_metrics"]["tp"] == report["micro_metrics"]["fn"] == 0
    assert report["micro_metrics"]["f1"] is None


def test_malformed_reference_container_does_not_disappear_as_skip(registry):
    raw = {"id": "5", "name": "Note", "value_type": "STRING", "value": "Not a label list"}
    report = build_transcript_report(
        [source("x", raw)], [prediction(registry, "x", [])], registry
    )
    assert report["transcripts"][0]["skipped_expected_count"] == 0
    assert report["micro_metrics"]["fn"] == 1
    assert any(issue["code"] == "label_count_unknown"
               for issue in report["transcripts"][0]["reference_issues"])


def test_micro_metrics_include_missing_and_failed_row_misses_but_not_skips(registry):
    label = observation(registry, "1", "No")
    skipped = {"id": "5", "name": "Note", "value_type": "STRING", "value": "Text"}
    rows = [source(row_id, [label, skipped]) for row_id in ("ok", "missing", "failed")]
    report = build_transcript_report(
        rows,
        [prediction(registry, "ok", [label]), prediction(registry, "failed", [], status="failed")],
        registry,
    )
    assert report["micro_metrics"] == {
        "available": True, "transcript_count": 3, "evaluated_transcript_count": 1,
        "skipped_expected_count": 3, "tp": 1, "fp": 0, "fn": 2,
        "precision": 1, "recall": 1 / 3, "f1": 0.5,
    }
    assert report["transcripts"][1]["recall"] is None
    assert report["transcripts"][2]["recall"] is None


def test_empty_report_micro_metrics_are_unavailable(registry):
    report = build_transcript_report([], [], registry)
    assert report["micro_metrics"]["available"] is False
    assert report["micro_metrics"]["transcript_count"] == 0
    assert report["micro_metrics"]["f1"] is None


def test_predicted_provenance_retains_recorded_spans_reasons_decisions_and_requests(registry):
    transcript = "Urine output 150 cc."
    label = observation(registry, "3", 150)
    start = transcript.index("150")
    evidence = {"start": start, "end": start + 3, "text": "150", "value": 150, "kind": "number"}
    audit = {
        "id": "3", "status": "emitted", "reason": "Recorded fixture reason.",
        "evidence": [evidence],
        "decisions": [{"question_id": "obs_3_number_0", "choice": "candidate_0",
                       "confidence": 0.9, "probabilities": {"candidate_0": 0.9, "ambiguous": 0.1}}],
    }
    request = {"question_ids": ["obs_3_number_0"], "request_id": "fixture-request",
               "model": "fixture-not-jev", "status": "returned"}
    record = prediction(
        registry, "x", [label], audit=[audit],
        requests=[request, {"question_ids": ["obs_4_value"], "request_id": "unrelated"}],
        metadata={"actual_models": ["fixture-not-jev"], "state_hash": "fixture-hash"},
    )
    original = deepcopy(record)
    result = build_transcript_report(
        [source("x", [label], transcript=transcript)], [record], registry
    )["transcripts"][0]["predicted_observations"][0]
    provenance = result["provenance"]
    assert provenance["audit_status"] == "recorded"
    assert provenance["audit"] == [audit]
    assert provenance["requests"] == [request]
    assert provenance["metadata"] == record["metadata"]
    span = provenance["audit"][0]["evidence"][0]
    assert transcript[span["start"]:span["end"]] == span["text"]
    provenance["audit"][0]["evidence"][0]["text"] = "modified"
    provenance["requests"][0]["request_id"] = "modified"
    assert record == original


def test_no_audit_does_not_invent_a_span_from_transcript(registry):
    label = observation(registry, "3", 150)
    report = build_transcript_report(
        [source("x", [label], transcript="Urine output 150 cc.")],
        [prediction(registry, "x", [label], audit=[])], registry,
    )
    provenance = report["transcripts"][0]["predicted_observations"][0]["provenance"]
    assert provenance == {"audit_status": "missing", "audit": [], "requests": [], "metadata": {}}


def test_duplicate_audits_are_preserved_as_ambiguous_provenance(registry):
    label = observation(registry, "1", "No")
    audits = [{"id": "1", "status": "emitted", "reason": reason} for reason in ("one", "two")]
    report = build_transcript_report(
        [source("x", [label])],
        [prediction(registry, "x", [label], audit=audits)], registry,
    )
    provenance = report["transcripts"][0]["predicted_observations"][0]["provenance"]
    assert provenance["audit_status"] == "ambiguous"
    assert provenance["audit"] == audits


@pytest.mark.parametrize("requests", [None, [None], [{"question_ids": "not a list"}]])
def test_malformed_request_provenance_is_an_explicit_error(registry, requests):
    label = observation(registry, "1", "No")
    with pytest.raises(ValueError, match="requests|question_ids"):
        build_transcript_report(
            [source("x", [label])],
            [prediction(registry, "x", [label], requests=requests)], registry,
        )


def test_split_id_validation_happens_before_per_row_matching(registry):
    label = observation(registry, "1", "No")
    rows = [source("same", [label], split="dev"), source("same", [], split="test")]
    with pytest.raises(ValueError, match="split to disambiguate"):
        build_transcript_report(rows, [prediction(registry, "same", [label])], registry)
    report = build_transcript_report(
        rows, [prediction(registry, "same", [label], split="dev")], registry
    )
    assert report["transcripts"][0]["error_counts"]["COR"] == 1
    assert report["transcripts"][1]["prediction_status"] == "missing"
    with pytest.raises(ValueError, match="Duplicate source"):
        build_transcript_report([rows[0], rows[0]], [], registry)
    record = prediction(registry, "same", [label], split="dev")
    with pytest.raises(ValueError, match="Duplicate prediction"):
        build_transcript_report(rows, [record, record], registry)


def test_invalid_duplicate_predictions_and_unrequested_rows_are_visible(registry):
    label = observation(registry, "1", "No")
    report = build_transcript_report(
        [source("x", [label])],
        [prediction(registry, "x", [label, deepcopy(label)]),
         prediction(registry, "outside", [label])],
        registry,
    )
    result = report["transcripts"][0]
    assert result["error_counts"] == {"COR": 0, "DEL": 0, "INS": 1, "SUB": 1}
    assert result["prediction_issues"]
    assert any(issue["code"] == "unrequested_row" for issue in report["prediction_issues"])


@pytest.mark.parametrize("reference_view", ["raw", "normalized"])
def test_short_report_contains_only_requested_fields_and_preserves_metrics(registry, reference_view):
    label = observation(registry, "3", 150)
    skipped = {"id": "5", "name": "Note", "value_type": "STRING", "value": "Text"}
    rows = [
        source(row_id, [{**label, "id": "003"}, skipped])
        for row_id in ("complete", "partial", "empty", "missing", "failed")
    ]
    predictions = [
        prediction(registry, "complete", [label]),
        prediction(registry, "partial", [observation(registry, "3", 100)], status="partial"),
        prediction(registry, "empty", []),
        prediction(registry, "failed", [], status="failed", failures=[{"error": "timeout"}]),
    ]
    report = build_transcript_report(rows, predictions, registry, reference_view=reference_view)
    before = deepcopy(report)
    short = build_short_transcript_report(report)
    assert list(short) == ["transcripts", "micro_metrics"]
    assert short["micro_metrics"] == report["micro_metrics"]
    assert len(short["transcripts"]) == len(rows)
    for entry, full in zip(short["transcripts"], report["transcripts"], strict=True):
        assert list(entry) == ["id", "transcript", "comparisons", "metrics"]
        assert entry["id"] == full["id"]
        assert entry["transcript"] == full["transcript"]
        assert entry["comparisons"] == full["comparisons"]
        assert entry["metrics"] == {
            name: full[name]
            for name in (
                "available", "error_counts", "skipped_expected_count", "precision", "recall", "f1"
            )
        }
    for entry in short["transcripts"][-2:]:
        assert entry["metrics"]["available"] is False
        assert entry["metrics"]["error_counts"] is None
        assert entry["metrics"]["f1"] is None
    short["transcripts"][0]["comparisons"][0]["expected_observation"]["value"] = "changed"
    short["transcripts"][0]["metrics"]["error_counts"]["COR"] = 99
    short["micro_metrics"]["tp"] = 99
    assert report == before


def test_empty_short_report_retains_unavailable_micro_metrics(registry):
    report = build_transcript_report([], [], registry)
    short = build_short_transcript_report(report)
    assert short == {"transcripts": [], "micro_metrics": report["micro_metrics"]}
    assert short["micro_metrics"]["available"] is False
    assert short["micro_metrics"]["f1"] is None


def test_report_saves_valid_json_without_overwriting(registry, tmp_path):
    label = observation(registry, "1", "No")
    directory = tmp_path / "results" / "first"
    rows = [source("x", [label])]
    path = save_transcript_report(
        directory, rows, [prediction(registry, "x", [label])], registry,
        metadata={"requested_model": "fixture-not-jev"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "transcript_report.json"
    assert payload["transcripts"][0]["error_counts"]["COR"] == 1
    assert payload["metadata"] == {"requested_model": "fixture-not-jev"}
    assert payload["created_at"]
    assert payload["format_version"] == 3
    assert list(payload)[-1] == "micro_metrics"
    assert payload["micro_metrics"]["precision"] == payload["micro_metrics"]["f1"] == 1
    assert "raw_expected_observations" not in payload["transcripts"][0]
    short_path = path.with_name("transcript_report_short.json")
    short = json.loads(short_path.read_text(encoding="utf-8"))
    assert short == build_short_transcript_report(payload)
    before = path.read_bytes()
    short_before = short_path.read_bytes()
    with pytest.raises(FileExistsError):
        save_transcript_report(directory, rows, [], registry)
    assert path.read_bytes() == before
    assert short_path.read_bytes() == short_before
    assert set(directory.iterdir()) == {path, short_path}


def test_invalid_report_inputs_fail_before_creating_files(registry, tmp_path):
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="transcript string"):
        save_transcript_report(directory, [{"id": "x", "observations": []}], [], registry)
    with pytest.raises(ValueError, match="reference_view"):
        build_transcript_report([], [], registry, reference_view="other")
    with pytest.raises(ValueError):
        save_transcript_report(directory, [], [], registry, metadata={"invalid": float("nan")})
    assert not directory.exists()
