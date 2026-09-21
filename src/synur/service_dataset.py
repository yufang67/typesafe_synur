"""Read local extraction-service SYNUR exports without changing source files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from synur.dataset import LocalDataset, _json, _read_file
from synur.observations import SchemaRegistry

TYPE_IDS = {
    "SingleSelect": "SINGLE_SELECT",
    "MultiSelect": "MULTI_SELECT",
    "Numeric": "NUMERIC",
    "String": "STRING",
}


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string.")
    return value


def _objects(value: object, context: str) -> list[dict]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{context} must be a list of objects.")
    return value


def _read_export(path: Path) -> tuple[dict, dict]:
    path = Path(path).absolute()
    raw = _read_file(path)
    try:
        value = _json(raw.decode("utf-8-sig"), str(path))
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 in {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value, {
        "path": str(path),
        "filename": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _transcript(row: dict) -> str:
    context = f"Test {row['id']!r}"
    envelope = row.get("transcript")
    if not isinstance(envelope, dict) or not isinstance(envelope.get("transcript"), dict):
        raise ValueError(f"{context} requires a structured transcript.")
    turns = _objects(envelope["transcript"].get("turns"), f"{context} turns")
    if not turns or [turn.get("index") for turn in turns] != list(range(len(turns))):
        raise ValueError(f"{context} requires nonempty, consecutively indexed turns.")
    recordings = _objects(envelope.get("recordings"), f"{context} recordings")
    selected: set[int] = set()
    for recording in recordings:
        first, last = recording.get("first_turn"), recording.get("last_turn")
        if (
            type(recording.get("do_extraction")) is not bool
            or type(first) is not int
            or type(last) is not int
            or not 0 <= first <= last < len(turns)
        ):
            raise ValueError(f"{context} has an invalid recording range.")
        if recording["do_extraction"]:
            selected.update(range(first, last + 1))
    if selected != set(range(len(turns))):
        raise ValueError(f"{context} must select all turns for extraction.")
    return "\n\n".join(
        f"[{_string(turn.get('speaker'), f'{context} speaker')}] "
        f"{_string(turn.get('text'), f'{context} turn text')}"
        for turn in turns
    )


def load_service_dataset(dataset_path: Path, schema_path: Path) -> LocalDataset:
    """Convert service exports, using only the explicitly supplied schema.

    Date concepts are unsupported and explicitly inventoried in the manifest.
    Any Date references are retained separately as excluded_observations.
    All other reference values remain unchanged, including label anomalies.
    """
    data, data_record = _read_export(dataset_path)
    schema, schema_record = _read_export(schema_path)
    source_entries = _objects(schema.get("observations"), "Schema observations")
    lists = schema.get("listValueSets")
    if not source_entries or not isinstance(lists, dict):
        raise ValueError("Schema requires observations and a listValueSets object.")
    entries = []
    excluded = []
    by_id = {}
    for source in source_entries:
        concept_id = _string(source.get("observationId"), "Schema observationId")
        name = _string(source.get("displayName"), f"Concept {concept_id} displayName")
        kind = _string(source.get("typeId"), f"Concept {concept_id} typeId")
        if concept_id in by_id:
            raise ValueError(f"Duplicate schema observationId {concept_id!r}.")
        by_id[concept_id] = source
        if kind == "Date":
            excluded.append(source)
            continue
        if kind not in TYPE_IDS:
            raise ValueError(f"Unsupported schema typeId {kind!r} for {concept_id!r}.")
        entry: dict = {"id": concept_id, "name": name, "value_type": TYPE_IDS[kind]}
        if kind in ("SingleSelect", "MultiSelect"):
            list_id = source.get("listValueSetId")
            values = lists.get(list_id) if isinstance(list_id, str) else None
            if not isinstance(values, dict):
                raise ValueError(f"Missing listValueSet for concept {concept_id!r}.")
            entry["value_enum"] = [
                _string(item.get("displayName"), f"Concept {concept_id} enum value")
                for item in _objects(values.get("values"), f"Concept {concept_id} values")
            ]
        entries.append(entry)
    SchemaRegistry.from_entries(entries)
    tests = _objects(data.get("tests"), "Dataset tests")
    if not tests:
        raise ValueError("Dataset tests must not be empty.")
    rows = []
    seen = set()
    for test in tests:
        row_id = _string(test.get("id"), "Test id")
        if row_id in seen:
            raise ValueError(f"Duplicate test id {row_id!r}.")
        seen.add(row_id)
        expected = test.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"Test {row_id!r} requires expected observations.")
        labels, skipped = [], []
        for label in _objects(expected.get("observations"), f"Test {row_id} observations"):
            concept_id = _string(label.get("observationId"), "Reference observationId")
            if concept_id not in by_id or "value" not in label:
                raise ValueError(f"Test {row_id!r}: unknown concept or missing reference value.")
            source = by_id[concept_id]
            if source["typeId"] == "Date":
                skipped.append(label)
                continue
            labels.append({
                "id": concept_id,
                "name": _string(label.get("displayName"), "Reference displayName"),
                "value_type": TYPE_IDS[source["typeId"]],
                "value": label["value"],
            })
        rows.append({
            "id": row_id,
            "transcript": _transcript(test),
            "observations": labels,
            "excluded_observations": skipped,
        })
    data_record["row_count"] = len(rows)
    schema_record["row_count"] = len(source_entries)
    return LocalDataset(entries, {"local": rows}, {
        "format_version": 1,
        "source": "local-extraction-service-export",
        "test_set_id": data.get("testSetId"),
        "dataset_version": data.get("version"),
        "files": [data_record, schema_record],
        "source_concept_count": len(source_entries),
        "excluded_schema_entries": excluded,
        "excluded_reference_count": sum(len(row["excluded_observations"]) for row in rows),
        "schema_policy": "explicit-schema-file-overrides-embedded-schema",
        "transcript_policy": "all-turns-in-source-order-with-speaker-labels",
    })
