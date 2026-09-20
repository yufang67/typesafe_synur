from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from synur.observations import Concept, SchemaRegistry, normalize_references, validate_observations


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
            {
                "id": "2",
                "name": "Symptoms",
                "value_type": "MULTI_SELECT",
                "value_enum": ["Pain", "Fatigue", "Caf\u00e9"],
            },
            {"id": "3", "name": "Note", "value_type": "STRING"},
            {"id": "4", "name": "Temperature", "value_type": "NUMERIC"},
            {
                "id": "5",
                "name": "Grade",
                "value_type": "SINGLE_SELECT",
                "value_enum": ["1", "2", "3.0"],
            },
            {
                "id": "6",
                "name": "Description",
                "value_type": "SINGLE_SELECT",
                "value_enum": ["Caf\u00e9", "It\u2019s present"],
            },
        ]
    )


def observation(registry, concept_id, value):
    concept = registry.by_id[concept_id]
    return {
        "id": concept.id,
        "name": concept.name,
        "value_type": concept.value_type,
        "value": value,
    }


def test_registry_round_trip_and_frozen_concept(registry):
    assert len(registry.concepts) == 6
    assert registry.by_id["3"] == Concept("3", "Note", "STRING")
    assert SchemaRegistry.from_entries(registry.to_entries()) == registry
    with pytest.raises(FrozenInstanceError):
        registry.concepts[0].id = "changed"
    entries = registry.to_entries()
    entries[0]["value_enum"].append("unknown")
    assert "unknown" not in registry.concepts[0].value_enum
    ids = registry.by_id
    ids.clear()
    assert len(registry.by_id) == 6


@pytest.mark.parametrize(
    "entries,match",
    [
        ({}, "list"),
        ([None], "object"),
        ([{"id": 1, "name": "X", "value_type": "STRING"}], "id"),
        ([{"id": "1", "name": "", "value_type": "STRING"}], "name"),
        ([{"id": "1", "name": "X", "value_type": "OTHER"}], "value_type"),
        ([{"id": "1", "name": "X", "value_type": "SINGLE_SELECT"}], "must not be empty"),
        ([{"id": "1", "name": "X", "value_type": "STRING", "value_enum": ["x"]}], "must be empty"),
        ([{"id": "1", "name": "X", "value_type": "MULTI_SELECT", "value_enum": [1]}], "strings"),
        (
            [{"id": "1", "name": "X", "value_type": "MULTI_SELECT", "value_enum": ["x", "x"]}],
            "duplicate",
        ),
        ([{"id": "1", "name": "X", "value_type": "STRING"}] * 2, "duplicate concept"),
    ],
)
def test_registry_rejects_malformed_entries(entries, match):
    with pytest.raises(ValueError, match=match):
        SchemaRegistry.from_entries(entries)


def test_all_four_types_and_returned_deep_copy(registry):
    raw = [
        observation(registry, "1", "No"),
        observation(registry, "2", ["Pain", "Fatigue"]),
        observation(registry, "3", "a note"),
        observation(registry, "4", -12.5),
    ]
    result = validate_observations(raw, registry)
    assert result == raw
    result[1]["value"].append("Caf\u00e9")
    assert raw[1]["value"] == ["Pain", "Fatigue"]
    assert validate_observations([], registry) == []


@pytest.mark.parametrize(
    "concept_id,value",
    [
        ("1", "yes"),
        ("1", 1),
        ("1", ["Yes"]),
        ("1", None),
        ("2", "Pain"),
        ("2", []),
        ("2", ["Pain", "Pain"]),
        ("2", ["pain"]),
        ("2", [False]),
        ("2", ["Pain", None]),
        ("3", 123),
        ("3", ""),
        ("3", "   "),
        ("3", ["note"]),
        ("4", True),
        ("4", False),
        ("4", "1"),
        ("4", float("nan")),
        ("4", float("inf")),
        ("4", -float("inf")),
        ("4", None),
    ],
)
def test_validation_rejects_invalid_values(registry, concept_id, value):
    with pytest.raises(ValueError, match="Observation 0.*value"):
        validate_observations([observation(registry, concept_id, value)], registry)


@pytest.mark.parametrize(
    "patch,match",
    [
        ({"id": "01"}, "unknown concept ID"),
        ({"id": 1}, "unknown concept ID"),
        ({"id": ["1"]}, "unknown concept ID"),
        ({"name": "Wrong"}, "schema name"),
        ({"value_type": "STRING"}, "schema type"),
        ({"confidence": 0.9}, "unexpected fields"),
    ],
)
def test_validation_rejects_metadata_and_shape_mismatches(registry, patch, match):
    raw = observation(registry, "1", "Yes")
    raw.update(patch)
    with pytest.raises(ValueError, match=match):
        validate_observations([raw], registry)
    with pytest.raises(ValueError, match="missing required fields"):
        validate_observations([{"id": "1"}], registry)


@pytest.mark.parametrize("raw", [{}, None, "[]", (), [None]])
def test_prediction_container_is_strict(registry, raw):
    with pytest.raises(ValueError):
        validate_observations(raw, registry)


@pytest.mark.parametrize("second", ["Yes", "No"])
def test_repeated_and_conflicting_predictions_rejected(registry, second):
    with pytest.raises(ValueError, match="duplicate concept ID"):
        validate_observations(
            [observation(registry, "1", "Yes"), observation(registry, "1", second)], registry
        )


def test_reference_repairs_preserve_raw_and_log_context(registry):
    raw = [
        observation(registry, "1", "No"),
        observation(registry, "3", 123),
        observation(registry, "5", 3.0),
    ]
    raw[0]["id"] = "001"
    original = deepcopy(raw)
    result = normalize_references(raw, registry, split="dev", row_id="row-1")
    assert raw == original == result.raw
    assert result.observations[0]["id"] == "1"
    assert result.observations[1]["value"] == "123"
    assert result.observations[2]["value"] == "3.0"
    assert result.issues == []
    assert [entry["reason"] for entry in result.changes] == [
        "unambiguous_zero_padded_id",
        "numeric_to_exact_string",
        "numeric_to_exact_string",
    ]
    assert all(
        change["split"] == "dev" and change["row_id"] == "row-1" for change in result.changes
    )
    assert result.changes[0]["original"] == "001"
    result.observations[0]["value"] = "Yes"
    assert result.raw[0]["value"] == "No"


@pytest.mark.parametrize(
    "patch", [{"name": "Wrong"}, {"value_type": "STRING"}, {"id": "1.0"}, {"id": 1}, {"id": "+01"}]
)
def test_zero_padding_requires_exact_metadata_and_digit_shape(registry, patch):
    label = {**observation(registry, "1", "Yes"), "id": "001", **patch}
    result = normalize_references([label], registry)
    assert result.observations == [label]
    assert result.changes == []
    assert result.issues


def test_ambiguous_numeric_ids_not_repaired(registry):
    entries = registry.to_entries()
    entries.append({"id": "01", "name": "Other", "value_type": "STRING"})
    ambiguous = SchemaRegistry.from_entries(entries)
    raw = {**observation(registry, "1", "Yes"), "id": "0001"}
    result = normalize_references([raw], ambiguous)
    assert result.observations == [raw]
    assert result.changes == []
    assert result.issues


@pytest.mark.parametrize(
    "concept_id,value",
    [("5", 1.0), ("5", 4), ("3", True), ("3", float("inf")), ("4", "12"), ("2", [1])],
)
def test_no_broad_numeric_or_text_coercion(registry, concept_id, value):
    raw = observation(registry, concept_id, value)
    result = normalize_references([raw], registry)
    assert result.changes == []
    assert result.observations == [raw]
    assert result.issues


def test_narrow_enum_encoding_repairs_single_and_multi(registry):
    cafe = "Caf\u00e9"
    apostrophe = "It\u2019s present"
    raw = [
        observation(registry, "6", apostrophe.encode("utf-8").decode("cp1252")),
        observation(registry, "2", ["Pain", cafe.encode("utf-8").decode("latin-1")]),
    ]
    result = normalize_references(raw, registry)
    assert result.observations[0]["value"] == apostrophe
    assert result.observations[1]["value"] == ["Pain", cafe]
    assert [change["field"] for change in result.changes] == ["value", "value[1]"]
    assert not result.issues
    assert raw != result.observations


@pytest.mark.parametrize(
    "concept_id,value",
    [
        ("6", "Cafe"),
        ("6", " Caf\u00e9"),
        ("6", "Caf\ufffd"),
        ("6", "Caf\u00e9".encode("utf-8").decode("latin-1").encode("utf-8").decode("latin-1")),
        ("3", "Caf\u00e9".encode("utf-8").decode("latin-1")),
    ],
)
def test_unknown_text_or_multiple_encoding_steps_not_repaired(registry, concept_id, value):
    raw = observation(registry, concept_id, value)
    result = normalize_references([raw], registry)
    assert result.observations == [raw]
    assert not result.changes


def test_references_preserve_repeated_and_conflicting_labels(registry):
    yes = observation(registry, "1", "Yes")
    raw = [yes, deepcopy(yes), observation(registry, "1", "No")]
    result = normalize_references(raw, registry)
    assert result.observations == raw
    assert [issue["code"] for issue in result.issues] == ["duplicate_id", "conflicting_id"]
    assert not result.changes


def test_reference_json_and_malformed_labels_remain_visible(registry):
    assert normalize_references("[]", registry).observations == []
    malformed = normalize_references([None, {"id": "unknown"}, 3], registry)
    assert len(malformed.observations) == 3
    assert malformed.observations[0] == {"_raw": None}
    assert malformed.issues
    for raw in ("not json", {"id": "1"}, None):
        result = normalize_references(raw, registry)
        assert len(result.observations) == 1
        assert result.raw == raw
        assert result.issues[0]["code"] == "label_count_unknown"
    missing = observation(registry, "1", "Yes")
    missing.pop("value")
    result = normalize_references([missing], registry)
    assert result.observations == [missing]
    assert not result.changes
