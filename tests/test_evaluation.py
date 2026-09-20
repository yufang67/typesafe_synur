import json
from copy import deepcopy

import pytest

from synur.evaluation import evaluate
from synur.observations import SchemaRegistry


@pytest.fixture
def registry():
    return SchemaRegistry.from_entries(
        [
            {
                "id": "1",
                "name": "Nausea",
                "value_type": "SINGLE_SELECT",
                "value_enum": ["Yes", "No"],
            },
            {"id": "2", "name": "Symptoms", "value_type": "MULTI_SELECT", "value_enum": ["A", "B"]},
            {"id": "3", "name": "Note", "value_type": "STRING"},
            {"id": "4", "name": "Temperature", "value_type": "NUMERIC"},
        ]
    )


def obs(registry, concept_id, value):
    concept = registry.by_id[concept_id]
    return {
        "id": concept_id,
        "name": concept.name,
        "value_type": concept.value_type,
        "value": value,
    }


def source(row_id, labels, **kwargs):
    return {"id": row_id, "transcript": "Synthetic example", "observations": labels, **kwargs}


def prediction(registry, row_id, labels, status="complete", **kwargs):
    emitted = {label["id"] for label in labels if isinstance(label, dict) and "id" in label}
    record = {
        "id": row_id,
        "status": status,
        "observations": labels,
        "audit": [
            {"id": concept.id, "status": "emitted" if concept.id in emitted else "absent"}
            for concept in registry.concepts
        ],
        "failures": [],
    }
    record.update(kwargs)
    return record


def test_exact_multiset_scores_order_numeric_and_text(registry):
    gold = [
        obs(registry, "1", "No"),
        obs(registry, "2", ["B", "A"]),
        obs(registry, "3", " Exact text "),
        obs(registry, "4", 1),
    ]
    labels = [
        obs(registry, "4", 1.0),
        obs(registry, "3", " Exact text "),
        obs(registry, "2", ["A", "B"]),
        obs(registry, "1", "No"),
    ]
    report = evaluate([source("x", gold)], [prediction(registry, "x", labels)], registry)
    assert report["available"]
    assert report["normalized"]["observation"] == {
        "tp": 4,
        "fp": 0,
        "fn": 0,
        "precision": 1,
        "recall": 1,
        "f1": 1,
    }
    assert report["raw"]["observation"] == report["normalized"]["observation"]
    assert report["coverage"]["fully_audited_complete_rows"] == 1
    assert report["coverage"]["audit_rate"] == 1
    assert report["schema_validity"] == {"submitted": 4, "valid": 4, "invalid": 0, "rate": 1}
    assert not report["prediction_issues"]
    assert json.loads(json.dumps(report)) == report


def test_known_counts_per_type_concept_and_error_analysis(registry):
    gold = [obs(registry, "1", "Yes"), obs(registry, "3", "A"), obs(registry, "4", 2)]
    labels = [obs(registry, "1", "Yes"), obs(registry, "3", "a"), obs(registry, "2", ["A"])]
    report = evaluate([source("x", gold)], [prediction(registry, "x", labels)], registry)
    score = report["normalized"]
    assert score["observation"] == {
        "tp": 1,
        "fp": 2,
        "fn": 2,
        "precision": 1 / 3,
        "recall": 1 / 3,
        "f1": 1 / 3,
    }
    assert score["concept"]["tp"] == 2
    assert score["per_type"]["NUMERIC"]["fn"] == 1
    assert score["per_type"]["STRING"]["fp"] == score["per_type"]["STRING"]["fn"] == 1
    assert score["edit_counts"] == {
        "correct": 1, "insertions": 1, "deletions": 1, "substitutions": 1
    }
    assert score["alignment"]["correct"][0]["reference"] == gold[0]
    assert score["alignment"]["insertions"][0]["prediction"] == labels[2]
    assert score["alignment"]["deletions"][0]["reference"] == gold[2]
    assert score["alignment"]["substitutions"][0]["reference"] == gold[1]
    assert score["alignment"]["substitutions"][0]["prediction"] == labels[1]
    assert len(score["errors"]["false_positives"]) == 2
    assert len(score["errors"]["missing_observations"]) == 2
    assert len(score["errors"]["value_mismatches"]) == 1


@pytest.mark.parametrize("value", ["text ", "Text", " text", "te\u0301xt"])
def test_text_comparison_is_conservative(registry, value):
    result = evaluate(
        [source("x", [obs(registry, "3", "text")])],
        [prediction(registry, "x", [obs(registry, "3", value)])],
        registry,
    )
    assert result["normalized"]["observation"]["tp"] == 0


def test_numeric_comparison_has_no_tolerance_and_preserves_large_integers(registry):
    rows = [
        source("a", [obs(registry, "4", 1)]),
        source("b", [obs(registry, "4", 9007199254740993)]),
    ]
    records = [
        prediction(registry, "a", [obs(registry, "4", 1.0000000000000002)]),
        prediction(registry, "b", [obs(registry, "4", 9007199254740992)]),
    ]
    assert evaluate(rows, records, registry)["raw"]["observation"]["tp"] == 0


def test_raw_and_normalized_views_preserve_input_and_expose_repairs(registry):
    label = obs(registry, "3", 123)
    label["id"] = "003"
    rows = [source("x", json.dumps([label]), split="dev")]
    records = [prediction(registry, "x", [obs(registry, "3", "123")])]
    before = deepcopy((rows, records))
    report = evaluate(rows, records, registry)
    assert report["raw"]["observation"]["fn"] == 1
    assert report["normalized"]["observation"]["tp"] == 1
    assert report["counts"]["normalization_changes"] == 2
    assert report["raw"]["edit_counts"] == {
        "correct": 0, "insertions": 1, "deletions": 1, "substitutions": 0
    }
    assert report["normalized"]["edit_counts"] == {
        "correct": 1, "insertions": 0, "deletions": 0, "substitutions": 0
    }
    assert report["reference_changes"][0]["split"] == "dev"
    assert (rows, records) == before


def test_duplicate_and_conflicting_gold_remain_individual_misses(registry):
    label = obs(registry, "1", "Yes")
    report = evaluate(
        [source("x", [label, deepcopy(label), obs(registry, "1", "No")])],
        [prediction(registry, "x", [label])],
        registry,
    )
    assert report["normalized"]["observation"]["tp"] == 1
    assert report["normalized"]["observation"]["fn"] == 2
    assert report["normalized"]["concept"]["fn"] == 2
    assert len(report["reference_issues"]) == 2


def test_edit_alignment_matches_exact_values_before_substitutions(registry):
    gold = [obs(registry, "1", "Yes"), obs(registry, "1", "No"), obs(registry, "2", ["B", "A"])]
    predicted = [obs(registry, "2", ["A", "B"]), obs(registry, "1", "No")]
    report = evaluate([source("x", gold)], [prediction(registry, "x", predicted)], registry)
    assert report["normalized"]["edit_counts"] == {
        "correct": 2, "insertions": 0, "deletions": 1, "substitutions": 0
    }
    assert report["normalized"]["alignment"]["deletions"][0]["reference"] == gold[0]


def test_multi_select_value_difference_is_one_substitution(registry):
    report = evaluate(
        [source("x", [obs(registry, "2", ["A", "B"])])],
        [prediction(registry, "x", [obs(registry, "2", ["A"])])],
        registry,
    )
    assert report["normalized"]["edit_counts"] == {
        "correct": 0, "insertions": 0, "deletions": 0, "substitutions": 1
    }
    assert report["normalized"]["observation"]["precision"] == 0
    assert report["normalized"]["observation"]["recall"] == 0
    assert report["normalized"]["observation"]["f1"] == 0


@pytest.mark.parametrize("labels", [[None], [{"id": "unknown"}], [None, {"id": "unknown"}]])
def test_malformed_unpaired_labels_remain_in_edit_counts(registry, labels):
    report = evaluate(
        [source("gold-only", labels), source("prediction-only", [])],
        [prediction(registry, "gold-only", []),
         prediction(registry, "prediction-only", labels)],
        registry,
    )
    for view in ("raw", "normalized"):
        assert report[view]["edit_counts"] == {
            "correct": 0, "insertions": len(labels), "deletions": len(labels), "substitutions": 0
        }


def test_malformed_golds_never_disappear(registry):
    labels = [None, {"id": "unknown"}, obs(registry, "4", True), obs(registry, "1", "Maybe")]
    report = evaluate(
        [source("x", labels), source("y", "bad JSON")], [prediction(registry, "x", [])], registry
    )
    assert report["normalized"]["observation"]["fn"] == 5
    assert report["raw"]["observation"]["fn"] == 5
    assert report["normalized"]["per_type"]["INVALID"]["fn"] == 3
    assert report["reference_issues"]
    assert any(issue["code"] == "label_count_unknown" for issue in report["reference_issues"])


def test_missing_failed_and_review_rows_penalize_end_to_end_recall(registry):
    label = obs(registry, "1", "No")
    rows = [source(row_id, [label]) for row_id in ("ok", "missing", "failed", "review")]
    records = [
        prediction(registry, "ok", [label]),
        prediction(
            registry,
            "failed",
            [],
            "failed",
            audit=[{"id": "1", "status": "failed"}],
            failures=[{"kind": "transport", "message": "timeout"}],
        ),
        prediction(registry, "review", [], "partial", audit=[{"id": "1", "status": "review"}]),
    ]
    report = evaluate(rows, records, registry)
    assert report["raw"]["observation"]["recall"] == 0.25
    assert report["raw"]["observation"]["fn"] == 3
    assert report["coverage"]["missing_rows"] == 1
    assert report["coverage"]["failed_rows"] == 1
    assert report["coverage"]["partial_rows"] == 1
    assert report["counts"]["review"] == report["counts"]["failed"] == 1
    assert report["counts"]["review_rate"] == 1 / 16
    assert report["counts"]["failures_by_kind"] == {"transport": 1}
    assert report["failures"][0]["row_id"] == "failed"


def test_failed_row_emissions_cannot_claim_true_positives(registry):
    label = obs(registry, "1", "Yes")
    report = evaluate(
        [source("ok", []), source("bad", [label])],
        [prediction(registry, "ok", []), prediction(registry, "bad", [label], "failed")],
        registry,
    )
    assert report["raw"]["observation"]["tp"] == 0
    assert report["raw"]["observation"]["fn"] == 1
    assert any(issue["code"] == "failed_row_emissions" for issue in report["prediction_issues"])


@pytest.mark.parametrize("mode", ["none", "failed", "outside"])
def test_no_real_extraction_metrics_are_unavailable(registry, mode):
    label = obs(registry, "1", "No")
    records = []
    if mode == "failed":
        records = [prediction(registry, "x", [], "failed")]
    elif mode == "outside":
        records = [prediction(registry, "other", [label])]
    report = evaluate([source("x", [label])], records, registry)
    assert not report["available"]
    for view in ("raw", "normalized"):
        for metric in ("observation", "concept"):
            assert report[view][metric]["fn"] == 1
            assert all(report[view][metric][key] is None for key in ("precision", "recall", "f1"))


@pytest.mark.parametrize(
    "gold,predicted,expected",
    [
        (False, False, (None, None, None)),
        (True, False, (None, 0, 0)),
        (False, True, (0, None, 0)),
    ],
)
def test_empty_denominator_conventions(registry, gold, predicted, expected):
    label = obs(registry, "1", "No")
    report = evaluate(
        [source("x", [label] if gold else [])],
        [prediction(registry, "x", [label] if predicted else [])],
        registry,
    )
    assert report["available"]
    score = report["raw"]["observation"]
    assert tuple(score[key] for key in ("precision", "recall", "f1")) == expected


def test_empty_requested_sample_is_unavailable(registry):
    report = evaluate([], [], registry)
    assert not report["available"]
    assert report["coverage"]["row_rate"] is None
    assert report["coverage"]["audit_rate"] is None
    assert report["schema_validity"]["rate"] is None


def test_invalid_predictions_and_duplicates_are_false_positives(registry):
    good = obs(registry, "1", "Yes")
    malformed = obs(registry, "4", True)
    report = evaluate(
        [source("x", [good, obs(registry, "4", 1)])],
        [prediction(registry, "x", [good, deepcopy(good), malformed])],
        registry,
    )
    assert report["raw"]["observation"]["tp"] == 0
    assert report["raw"]["observation"]["fp"] == 3
    assert report["raw"]["observation"]["fn"] == 2
    assert report["schema_validity"]["invalid"] == 3
    assert report["schema_validity"]["rate"] == 0
    assert report["raw"]["edit_counts"] == {
        "correct": 0, "insertions": 1, "deletions": 0, "substitutions": 2
    }


def test_concept_matching_does_not_cross_row_boundaries(registry):
    label = obs(registry, "1", "Yes")
    report = evaluate(
        [source("a", [label]), source("b", [])],
        [prediction(registry, "a", []), prediction(registry, "b", [label])],
        registry,
    )
    assert report["raw"]["observation"]["tp"] == 0
    assert report["raw"]["concept"]["tp"] == 0
    assert report["raw"]["edit_counts"] == {
        "correct": 0, "insertions": 1, "deletions": 1, "substitutions": 0
    }


def test_split_qualified_identity_and_ambiguous_predictions(registry):
    rows = [source("x", [], split="dev"), source("x", [], split="test")]
    with pytest.raises(ValueError, match="split to disambiguate"):
        evaluate(rows, [prediction(registry, "x", [])], registry)
    report = evaluate(rows, [prediction(registry, "x", [], split="test")], registry)
    assert report["coverage"]["missing_rows"] == 1
    with pytest.raises(ValueError, match="split to disambiguate"):
        evaluate(
            [source("x", []), source("x", [], split="dev")],
            [prediction(registry, "x", [])],
            registry,
        )


def test_reject_ambiguous_duplicate_records_and_broken_outer_contract(registry):
    row = source("x", [])
    record = prediction(registry, "x", [])
    with pytest.raises(ValueError, match="Duplicate source"):
        evaluate([row, row], [], registry)
    with pytest.raises(ValueError, match="Duplicate prediction"):
        evaluate([row], [record, record], registry)
    with pytest.raises(ValueError, match="status"):
        evaluate([row], [{**record, "status": "skipped"}], registry)
    with pytest.raises(ValueError, match="audit must be a list"):
        evaluate([row], [{**record, "audit": None}], registry)
    with pytest.raises(ValueError, match="missing observations"):
        evaluate([{"id": "x"}], [], registry)


def test_full_schema_audit_is_verified_not_assumed(registry):
    audit = [
        {"id": "1", "status": "absent"},
        {"id": "1", "status": "absent"},
        {"id": "2", "status": "review"},
        {"id": "unknown", "status": "absent"},
    ]
    report = evaluate([source("x", [])], [prediction(registry, "x", [], audit=audit)], registry)
    assert report["coverage"]["fully_audited_complete_rows"] == 0
    assert report["coverage"]["audited_concepts"] == 1
    codes = {issue["code"] for issue in report["prediction_issues"]}
    assert {"duplicate_audit", "invalid_audit", "incomplete_schema_audit"} <= codes


def test_type_restricted_scores_keep_unexpected_labels_visible(registry):
    selected = SchemaRegistry(tuple(registry.concepts[:2]))
    label = obs(registry, "3", "unexpected")
    report = evaluate(
        [source("x", [label])],
        [prediction(selected, "x", [label])],
        selected,
    )
    assert report["value_types"] == ["SINGLE_SELECT", "MULTI_SELECT"]
    assert report["normalized"]["observation"]["fp"] == 1
    assert report["normalized"]["observation"]["fn"] == 1
    assert report["normalized"]["per_type"]["STRING"]["fp"] == 1
    assert "NUMERIC" not in report["normalized"]["per_type"]
    assert report["reference_issues"]
    assert report["prediction_issues"]


def test_candidate_coverage_is_posthoc_separate_from_selection_quality(registry):
    rows = [
        source("x", [obs(registry, "3", "source text"), obs(registry, "4", 2)]),
        source("missing", [obs(registry, "3", "unavailable")]),
    ]
    records = [
        prediction(
            registry,
            "x",
            [],
            diagnostics={
                "candidate_values": {"3": ["source text", "other"], "4": [1.0, 3]},
                "hierarchy": ["group_0"],
            },
            metadata={"model": "offline-test-fixture"},
        )
    ]
    before = deepcopy((rows, records))
    report = evaluate(rows, records, registry)
    assert report["diagnostics"]["candidate_coverage"] == {
        "expected": 3,
        "evaluated": 2,
        "covered": 1,
        "unavailable": 1,
        "uncovered": 1,
        "rate": 0.5,
    }
    assert report["normalized"]["observation"]["recall"] == 0
    assert report["diagnostics"]["rows"][0]["metadata"]["model"] == "offline-test-fixture"
    assert (rows, records) == before


def test_candidate_numeric_canonicalization_and_empty_candidates(registry):
    report = evaluate(
        [source("x", [obs(registry, "3", "text"), obs(registry, "4", 2)])],
        [prediction(registry, "x", [], diagnostics={"candidate_values": {"3": [], "4": [2.0]}})],
        registry,
    )
    assert report["diagnostics"]["candidate_coverage"]["rate"] == 0.5
    assert report["diagnostics"]["candidate_coverage"]["uncovered"] == 1


def test_shared_candidate_pools_and_concept_override(registry):
    report = evaluate(
        [source("x", [obs(registry, "3", "text"), obs(registry, "4", 2)])],
        [
            prediction(
                registry,
                "x",
                [],
                diagnostics={
                    "candidate_pools": {"STRING": ["text"], "NUMERIC": [2.0]},
                    "candidate_values": {"3": []},
                },
            )
        ],
        registry,
    )
    assert report["diagnostics"]["candidate_coverage"]["evaluated"] == 2
    assert report["diagnostics"]["candidate_coverage"]["covered"] == 1
    assert report["diagnostics"]["candidate_coverage"]["unavailable"] == 0


@pytest.mark.parametrize("pools", [None, [], {"STRING": 3}, {"UNKNOWN": []}])
def test_invalid_shared_candidate_pools_are_reported(registry, pools):
    report = evaluate(
        [source("x", [obs(registry, "3", "text")])],
        [prediction(registry, "x", [], diagnostics={"candidate_pools": pools})],
        registry,
    )
    assert any(issue["code"] == "invalid_candidate_pools" for issue in report["prediction_issues"])
    assert report["diagnostics"]["candidate_coverage"]["unavailable"] == 1


def test_invalid_candidate_diagnostics_are_visible(registry):
    report = evaluate(
        [source("x", [obs(registry, "3", "text"), obs(registry, "4", 2)])],
        [
            prediction(
                registry,
                "x",
                [],
                diagnostics={"candidate_values": {"3": "not a list", "4": [True], "unknown": []}},
            )
        ],
        registry,
    )
    codes = [issue["code"] for issue in report["prediction_issues"]]
    assert codes.count("invalid_candidate_values") == 2
    assert codes.count("invalid_candidate_value") == 1
    assert report["diagnostics"]["candidate_coverage"]["unavailable"] == 1
