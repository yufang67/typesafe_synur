import hashlib
import json

import pytest

from synur.service_dataset import load_service_dataset


@pytest.fixture
def exports(tmp_path):
    schema = {
        "observations": [
            {"observationId": "1", "displayName": "Nausea",
             "typeId": "SingleSelect", "listValueSetId": "yes-no"},
            {"observationId": "2", "displayName": "Symptoms",
             "typeId": "MultiSelect", "listValueSetId": "yes-no"},
            {"observationId": "3", "displayName": "Temperature", "typeId": "Numeric"},
            {"observationId": "4", "displayName": "Note", "typeId": "String"},
            {"observationId": "5", "displayName": "Removal Date", "typeId": "Date"},
        ],
        "listValueSets": {"yes-no": {"values": [{"displayName": "Yes"}, {"displayName": "No"}]}},
    }
    data = {
        "testSetId": "local-fixture",
        "version": "5",
        "schemaData": {"observations": "must not be used"},
        "tests": [{
            "id": "2-001",
            "transcript": {
                "recordings": [{"do_extraction": True, "first_turn": 0, "last_turn": 1}],
                "transcript": {"turns": [
                    {"index": 0, "speaker": "Clinician", "text": "No nausea.\nTemperature 37."},
                    {"index": 1, "speaker": "Patient", "text": "A note."},
                ]},
            },
            "expected": {"observations": [
                {"observationId": entry["observationId"], "displayName": entry["displayName"],
                 "value": value, "provenance": "source-only metadata"}
                for entry, value in zip(
                    schema["observations"], ["No", ["Yes", "No"], 37, "A note.", "2026-09-21"],
                    strict=True,
                )
            ]},
        }],
    }
    dataset_path, schema_path = tmp_path / "dataset.json", tmp_path / "schema.json"

    def write():
        dataset_path.write_text(json.dumps(data), encoding="utf-8")
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        return dataset_path, schema_path

    write()
    return data, schema, write


def test_service_exports_preserve_values_and_record_exact_sources(exports, monkeypatch):
    data, schema, write = exports
    paths = write()
    original = [path.read_bytes() for path in paths]

    def forbidden(*args, **kwargs):
        pytest.fail("Loader attempted network access")

    monkeypatch.setattr("socket.create_connection", forbidden)
    loaded = load_service_dataset(*paths)
    assert list(loaded.splits) == ["local"]
    row = loaded.splits["local"][0]
    assert row["id"] == "2-001"
    assert row["transcript"] == "[Clinician] No nausea.\nTemperature 37.\n\n[Patient] A note."
    assert [item["value"] for item in row["observations"]] == ["No", ["Yes", "No"], 37, "A note."]
    assert [entry["value_type"] for entry in loaded.schema_entries] == [
        "SINGLE_SELECT", "MULTI_SELECT", "NUMERIC", "STRING",
    ]
    assert loaded.schema_entries[0]["value_enum"] == ["Yes", "No"]
    assert row["excluded_observations"] == data["tests"][0]["expected"]["observations"][-1:]
    assert loaded.manifest["source_concept_count"] == 5
    assert loaded.manifest["excluded_reference_count"] == 1
    assert loaded.manifest["excluded_schema_entries"] == schema["observations"][-1:]
    for path, raw, record in zip(paths, original, loaded.manifest["files"], strict=True):
        assert path.read_bytes() == raw
        assert record["path"] == str(path)
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["byte_count"] == len(raw)


@pytest.mark.parametrize("mutation", [
    lambda d, s: d["tests"].append(d["tests"][0]),
    lambda d, s: d["tests"][0].update(id=1),
    lambda d, s: d["tests"][0].update(expected=None),
    lambda d, s: d["tests"][0]["expected"]["observations"][0].update(observationId="unknown"),
    lambda d, s: d["tests"][0]["expected"]["observations"][0].pop("value"),
    lambda d, s: d["tests"][0]["transcript"]["recordings"][0].update(do_extraction=False),
    lambda d, s: d["tests"][0]["transcript"]["recordings"][0].update(last_turn=0),
    lambda d, s: d["tests"][0]["transcript"]["transcript"]["turns"][0].update(index=1),
    lambda d, s: d["tests"][0]["transcript"]["transcript"]["turns"][0].update(text=None),
    lambda d, s: d.update(tests=[]),
    lambda d, s: s["observations"].append(s["observations"][0]),
    lambda d, s: s["observations"][0].update(typeId="Unknown"),
    lambda d, s: s["observations"][0].update(listValueSetId="missing"),
    lambda d, s: s["listValueSets"]["yes-no"].update(values=[]),
    lambda d, s: s["listValueSets"]["yes-no"]["values"].append({"displayName": "No"}),
])
def test_service_export_invalid_structure_fails_explicitly(exports, mutation):
    data, schema, write = exports
    mutation(data, schema)
    with pytest.raises(ValueError):
        load_service_dataset(*write())


@pytest.mark.parametrize("raw", [b"\xff", b'{"tests":[],"tests":[]}', b"[]", b"{"])
def test_service_export_invalid_json(exports, raw):
    _, _, write = exports
    dataset_path, schema_path = write()
    dataset_path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_service_dataset(dataset_path, schema_path)


def test_service_export_keeps_label_anomalies_for_evaluation(exports):
    data, _, write = exports
    data["tests"][0]["expected"]["observations"][0]["value"] = 42
    data["tests"][0]["expected"]["observations"][0]["displayName"] = "Incorrect name"
    loaded = load_service_dataset(*write())
    assert loaded.splits["local"][0]["observations"][0]["value"] == 42
    assert loaded.splits["local"][0]["observations"][0]["name"] == "Incorrect name"
