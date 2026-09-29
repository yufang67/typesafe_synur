import ast
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from synur.dataset import LocalDataset
from synur.evaluation import evaluate
from synur.observations import VALUE_TYPES, SchemaRegistry
from synur.reporting import save_transcript_report

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "synur_observation_extraction.ipynb"


@pytest.mark.parametrize("provider", ["typesafe", "pi-scorer"])
def test_notebook_defaults_to_bundled_exports(monkeypatch, provider):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("SYNUR_MODEL_PROVIDER", provider)
    for name in ("TYPESAFE_MODEL", "TYPESAFE_DEFAULT_MODEL", "TYPESAFE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SYNUR_DATASET_PATH", raising=False)
    monkeypatch.delenv("SYNUR_SCHEMA_PATH", raising=False)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    namespace = {}
    exec(compile(cells["configuration"].source, str(NOTEBOOK), "exec"), namespace)
    namespace["display"] = lambda value: None
    exec(compile(cells["load-data"].source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["DATASET_PATH"] == ROOT / "data" / "synur" / "synur_dataset.v5.json"
    assert namespace["SCHEMA_PATH"] == ROOT / "data" / "synur" / "synur_schema.v4.json"
    assert namespace["dataset"].manifest["source_concept_count"] == 198
    assert namespace["SPLIT"] == "local"
    assert len(namespace["rows"]) == 422
    assert len(namespace["registry"].concepts) == 166
    assert namespace["PROVIDER"] == provider
    assert namespace["MODEL"] == ("pi-scorer" if provider == "pi-scorer" else "jev-1.13.0")


def test_notebook_credential_cell_has_no_saved_input_or_outputs():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    setup = next(cell for cell in notebook.cells if cell.id == "api-key-setup")
    assert setup.execution_count is None
    assert setup.outputs == []
    assert "configure_api_key(enabled=LIVE_CALLS, provider=PROVIDER)" in setup.source
    ids = [cell.id for cell in notebook.cells]
    assert ids.index("configuration") < ids.index("api-key-setup") < ids.index("run-extraction")
    assert ids.index("quality-report") < ids.index("run-extraction") < ids.index("evaluate-export")


@pytest.mark.parametrize("provider", ["typesafe", "pi-scorer"])
@pytest.mark.parametrize("dataset_kind", ["service", "snapshot"])
def test_all_notebook_cells_offline_with_synthetic_data(
    provider, dataset_kind, monkeypatch, capsys
):
    import getpass
    import socket

    import httpx2
    import typesafe_sdk

    def forbidden(*args, **kwargs):
        pytest.fail("Offline notebook attempted credentials, a model client, or network access")

    for name in ("TYPESAFE_API_KEY", "TYPESAFE_MODEL", "TYPESAFE_DEFAULT_MODEL", "TYPESAFE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SYNUR_MODEL_PROVIDER", provider)
    monkeypatch.setattr(getpass, "getpass", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx2.Client, "send", forbidden)
    monkeypatch.setattr(typesafe_sdk.TypeSafeClient, "system_one", forbidden)
    monkeypatch.setattr("synur.jev.TypeSafeClient", forbidden)
    schema = [
        {"id": "1", "name": "Synthetic category", "value_type": "SINGLE_SELECT",
         "value_enum": ["A", "B"]},
        {"id": "2", "name": "Synthetic members", "value_type": "MULTI_SELECT",
         "value_enum": ["X", "Y"]},
        {"id": "3", "name": "Synthetic number", "value_type": "NUMERIC"},
        {"id": "4", "name": "Excluded note", "value_type": "STRING"},
    ]
    split = "local" if dataset_kind == "service" else "mediqa_synur_dev"
    dataset = LocalDataset(schema, {split: [{
        "id": "synthetic", "transcript": "Synthetic category A, member X, number 42.",
        "observations": [
            {key: entry[key] for key in ("id", "name", "value_type")} | {"value": value}
            for entry, value in zip(schema, ("A", ["X"], 42, "note"), strict=True)
        ],
    }]}, {})
    monkeypatch.setattr("synur.dataset.load_dataset", lambda _: deepcopy(dataset))
    monkeypatch.setattr("synur.service_dataset.load_service_dataset", lambda *_: deepcopy(dataset))
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    configuration = next(cell for cell in notebook.cells if cell.id == "configuration")
    tree = ast.parse(configuration.source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"LIVE_CALLS", "SAVE_RESULTS", "SAVE_REPORT"}
            for target in node.targets
        ):
            node.value = ast.Constant(value=False)
        elif dataset_kind == "snapshot" and isinstance(node, ast.Assign):
            replacements = {"DATASET_PATH": None, "SCHEMA_PATH": None, "SPLIT": split}
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.Constant(value=replacements[target.id])
    configuration.source = ast.unparse(tree)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["PROVIDER"] == provider
    assert namespace["MODEL"] == ("pi-scorer" if provider == "pi-scorer" else "jev-1.13.0")
    assert namespace["predictions"] == []
    assert namespace["request_preview"]["live_calls_made"] is False
    assert namespace["request_preview"]["concept_count"] == 3
    assert namespace["request_preview"]["numeric_candidates"][0]["value"] == 42
    assert namespace["metrics"]["available"] is False
    assert "report_path" not in namespace and "run_dir" not in namespace
    assert "Live calls disabled; no API key requested." in capsys.readouterr().out


@pytest.mark.parametrize("row_id", [None, "mixed", "disabled-only"])
def test_notebook_filters_labels_and_evaluation_without_changing_source(row_id):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    configuration = next(cell for cell in notebook.cells if cell.id == "configuration")
    enabled = next(
        ast.literal_eval(node.value)
        for node in ast.parse(configuration.source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ENABLED_VALUE_TYPES"
                for target in node.targets)
    )
    assert enabled == ("SINGLE_SELECT", "MULTI_SELECT", "NUMERIC")
    schema = [
        {"id": "1", "name": "Nausea", "value_type": "SINGLE_SELECT", "value_enum": ["No"]},
        {"id": "2", "name": "Symptoms", "value_type": "MULTI_SELECT", "value_enum": ["A"]},
        {"id": "3", "name": "Note", "value_type": "STRING"},
        {"id": "4", "name": "Temperature", "value_type": "NUMERIC"},
    ]
    labels = [
        {key: entry[key] for key in ("id", "name", "value_type")} | {"value": value}
        for entry, value in zip(schema, ("No", ["A"], "text", 37), strict=True)
    ]
    source_rows = [
        {"id": "mixed", "transcript": "Synthetic example", "observations": labels},
        {"id": "disabled-only", "transcript": "A note.", "observations": labels[2:3]},
    ]
    dataset = LocalDataset(schema, {"dev": source_rows, "test": deepcopy(source_rows)}, {})
    original = deepcopy(dataset)
    namespace = {
        "load_dataset": lambda _: dataset,
        "DATA_DIR": ROOT,
        "DATASET_PATH": None,
        "SCHEMA_PATH": None,
        "SchemaRegistry": SchemaRegistry,
        "VALUE_TYPES": VALUE_TYPES,
        "ENABLED_VALUE_TYPES": enabled,
        "SAMPLE_LIMIT": 2,
        "ROW_ID": row_id,
        "SPLIT": "dev",
        "Counter": Counter,
        "JSON": lambda value: value,
        "display": lambda value: None,
    }
    load = next(cell for cell in notebook.cells if cell.id == "load-data")
    exec(compile(load.source, str(NOTEBOOK), "exec"), namespace)
    selected_ids = [row_id] if row_id is not None else ["mixed", "disabled-only"]
    assert [row["id"] for row in namespace["rows"]] == selected_ids
    registry = namespace["registry"]
    assert [concept.id for concept in registry.concepts] == ["1", "2", "4"]
    assert dataset == original
    for rows in namespace["scoped_splits"].values():
        assert rows[0]["observations"] == [labels[0], labels[1], labels[3]]
        assert rows[1]["observations"] == []
    records = [
        {
            "id": row["id"],
            "status": "complete",
            "observations": row["observations"],
            "audit": [
                {"id": concept.id, "status": "emitted" if row["observations"] else "absent"}
                for concept in registry.concepts
            ],
            "failures": [],
        }
        for row in namespace["rows"]
    ]
    report = evaluate(namespace["rows"], records, registry)
    correct = 3 if "mixed" in selected_ids else 0
    for view in ("raw", "normalized"):
        assert report[view]["observation"]["tp"] == correct
        assert report[view]["observation"]["fp"] == report[view]["observation"]["fn"] == 0
        assert report[view]["observation"]["f1"] == (1 if correct else None)
        assert report[view]["edit_counts"] == {
            "correct": correct, "insertions": 0, "deletions": 0, "substitutions": 0
        }
        assert set(report[view]["per_type"]) == {*enabled, "INVALID"}
    assert report["coverage"]["expected_concepts"] == 3 * len(selected_ids)
    assert report["coverage"]["audit_rate"] == 1
    assert report["schema_validity"]["submitted"] == correct
    assert not report["reference_issues"]
    assert report["diagnostics"]["candidate_coverage"]["expected"] == int("mixed" in selected_ids)
    if "mixed" in selected_ids:
        wrong = deepcopy(records)
        wrong[0]["observations"][-1]["value"] = 38
        scored = evaluate(namespace["rows"], wrong, registry)["normalized"]
        assert scored["edit_counts"] == {
            "correct": 2, "insertions": 0, "deletions": 0, "substitutions": 1
        }
        assert scored["observation"]["precision"] == scored["observation"]["recall"] == 2 / 3
        assert scored["observation"]["f1"] == 2 / 3
        missing = deepcopy(records)
        missing[0]["observations"].pop()
        missing[0]["audit"][-1]["status"] = "absent"
        scored = evaluate(namespace["rows"], missing, registry)["normalized"]
        assert scored["edit_counts"] == {
            "correct": 2, "insertions": 0, "deletions": 1, "substitutions": 0
        }
        assert scored["observation"]["precision"] == 1
        assert scored["observation"]["recall"] == 2 / 3
        assert scored["observation"]["f1"] == 0.8
    assert dataset == original
    for invalid_id in ("not-found", "", 152):
        namespace["ROW_ID"] = invalid_id
        with pytest.raises(ValueError, match="ROW_ID|exactly one row"):
            exec(compile(load.source, str(NOTEBOOK), "exec"), namespace)


@pytest.mark.parametrize("provider", ["typesafe", "pi-scorer"])
def test_notebook_displays_transcript_labels_predictions_and_scores(capsys, tmp_path, provider):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    registry = SchemaRegistry.from_entries(
        [{"id": "1", "name": "Nausea", "value_type": "SINGLE_SELECT", "value_enum": ["No"]}]
    )
    label = {"id": "1", "name": "Nausea", "value_type": "SINGLE_SELECT", "value": "No"}
    rows = [{"id": "152", "transcript": "Patient denies nausea.", "observations": [label]}]
    skipped = {"id": "5", "name": "Note", "value_type": "STRING", "value": "Source note"}
    source_rows = [{**rows[0], "observations": [label, skipped]}]
    record = {
        "id": "152", "status": "complete", "observations": [label],
        "audit": [{"id": "1", "status": "emitted"}], "failures": [],
    }
    displayed = []
    configured = []

    class FixtureAdapter:
        def __init__(self, **kwargs):
            configured.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    from synur.observations import normalize_references

    namespace = {
        "rows": rows, "registry": registry, "SPLIT": "dev",
        "JSON": lambda value: value, "display": displayed.append,
        "normalize_references": normalize_references,
        "JevAdapter": FixtureAdapter, "LIVE_CALLS": True,
        "MODEL": "offline-fixture-not-jev", "SETTINGS": None,
        "PROVIDER": provider, "BASE_URL": None,
        "extract": lambda *args, **kwargs: deepcopy(record),
        "evaluate": evaluate, "SAVE_RESULTS": False, "SAVE_REPORT": True,
        "ROOT": tmp_path, "RUN_ID": f"{provider}_2026-09-29_fixture",
        "save_transcript_report": save_transcript_report,
        "dataset": LocalDataset([], {"dev": source_rows}, {}),
    }
    for cell_id in ("quality-report", "run-extraction", "evaluate-export"):
        exec(compile(cells[cell_id].source, str(NOTEBOOK), "exec"), namespace)
    assert configured == [{
        "enabled": True, "provider": provider, "model": "offline-fixture-not-jev", "base_url": None,
    }]
    assert displayed[0] == rows[0]["observations"]
    assert displayed[1] == record["observations"]
    assert displayed[2]["normalized"] == {
        "correct": 1, "insertions": 0, "deletions": 0, "substitutions": 0,
        "precision": 1, "recall": 1, "f1": 1,
    }
    assert displayed[3] == namespace["metrics"]["raw"]["alignment"]
    assert displayed[4] == namespace["metrics"]["normalized"]["alignment"]
    output = capsys.readouterr().out
    assert output.index("Patient denies nausea.") < output.index("Reference labels")
    assert output.index("Reference labels") < output.index("Model observations")
    assert output.index("Model observations") < output.index("Observation-level errors and scores")
    exported = json.loads(namespace["report_path"].read_text(encoding="utf-8"))
    assert exported["metadata"]["provider"] == provider
    assert namespace["report_path"].parent.name == f"report_{namespace['RUN_ID']}"
    assert exported["metadata"]["run_id"] == namespace["RUN_ID"]
    assert exported["transcripts"][0]["transcript"] == rows[0]["transcript"]
    assert exported["transcripts"][0]["split"] == "dev"
    assert exported["transcripts"][0]["expected_observations"][0]["error_type"] == "COR"
    assert exported["transcripts"][0]["predicted_observations"][0]["observation"] == label
    assert exported["transcripts"][0]["precision"] == exported["transcripts"][0]["f1"] == 1
    assert exported["transcripts"][0]["expected_observations"][-1]["observation"] == skipped
    assert exported["transcripts"][0]["expected_observations"][-1]["error_type"] == "SKIP"
    assert exported["transcripts"][0]["predicted_observations"][0]["provenance"]["audit"] == record["audit"]
    assert exported["micro_metrics"]["precision"] == exported["micro_metrics"]["recall"] == 1
    assert exported["micro_metrics"]["skipped_expected_count"] == 1
    assert list(exported)[-1] == "micro_metrics"
    assert "raw_expected_observations" not in exported["transcripts"][0]
    short_path = namespace["report_path"].with_name("transcript_report_short.json")
    short = json.loads(short_path.read_text(encoding="utf-8"))
    assert str(short_path) in output
    assert list(short["transcripts"][0]) == ["id", "transcript", "comparisons", "metrics"]
    assert short["transcripts"][0]["comparisons"] == exported["transcripts"][0]["comparisons"]
    assert short["transcripts"][0]["metrics"]["f1"] == 1
    assert short["micro_metrics"] == exported["micro_metrics"]
    namespace.update(SAVE_RESULTS=True, json=json)
    exec(compile(cells["run-extraction"].source, str(NOTEBOOK), "exec"), namespace)
    checkpoint = namespace["checkpoint_path"]
    assert checkpoint.is_file()
    assert checkpoint.name == f"checkpoint_{namespace['RUN_ID']}.jsonl"
    assert [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()] == [
        record,
    ]


@pytest.mark.skipif(
    not (ROOT / "data" / "synur" / "manifest.json").is_file(),
    reason="Run the separate dataset download before the local notebook integration test.",
)
def test_notebook_executes_offline_without_model_credentials(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("SYNUR_DATA_DIR", raising=False)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    configuration = next(cell for cell in notebook.cells if cell.id == "configuration")
    tree = ast.parse(configuration.source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in ("LIVE_CALLS", "SAVE_REPORT", "SAVE_RESULTS")
            for target in node.targets
        ):
            node.value = ast.Constant(value=False)
        elif isinstance(node, ast.Assign):
            replacements = {
                "DATASET_PATH": None, "SCHEMA_PATH": None,
                "SPLIT": "mediqa_synur_dev", "ROW_ID": "152",
            }
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.Constant(value=replacements[target.id])
    configuration.source = ast.unparse(tree)
    guard = nbformat.v4.new_code_cell("""
import getpass
import socket
import urllib.request
import httpx2
import typesafe_sdk

def forbidden(*args, **kwargs):
    raise AssertionError("Offline notebook attempted a network/model call")

urllib.request.urlopen = forbidden
getpass.getpass = forbidden
httpx2.Client.send = forbidden
typesafe_sdk.TypeSafeClient.system_one = forbidden
original_connect = socket.socket.connect
def local_only(sock, address):
    if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
        forbidden()
    return original_connect(sock, address)
socket.socket.connect = local_only
""")
    notebook.cells.insert(0, guard)
    notebook.cells.append(
        nbformat.v4.new_code_cell("""
assert len(full_registry.concepts) == 193
assert SPLIT == "mediqa_synur_dev"
assert ROW_ID == "152"
assert [row["id"] for row in rows] == ["152"]
assert len(registry.concepts) == 162
assert request_preview["concept_count"] == len(registry.concepts)
assert {concept.value_type for concept in registry.concepts} == set(ENABLED_VALUE_TYPES)
assert set(request_preview["value_question_examples"]) == set(ENABLED_VALUE_TYPES)
assert any(candidate["value"] == 150 for candidate in request_preview["numeric_candidates"])
assert request_preview["text_candidates"] == []
assert excluded_value_types == ("STRING",)
assert len(rows[0]["observations"]) == 12
assert any(label["value_type"] == "NUMERIC" and label["value"] == 150
           for label in rows[0]["observations"])
assert all(label.get("value_type") not in excluded_value_types
           for split_rows in scoped_splits.values()
           for row in split_rows for label in row["observations"])
assert metrics["value_types"] == list(ENABLED_VALUE_TYPES)
assert metrics["coverage"]["expected_concepts"] == len(rows) * len(registry.concepts)
assert metrics["raw"]["observation"]["fn"] == sum(len(row["observations"]) for row in rows)
assert set(metrics["normalized"]["per_type"]) == {*ENABLED_VALUE_TYPES, "INVALID"}
assert metrics["diagnostics"]["candidate_coverage"]["expected"] == 1
assert predictions == []
assert "summary" not in globals()
assert LIVE_CALLS is False
assert SAVE_REPORT is False
assert "report_path" not in globals()
print("OFFLINE_NOTEBOOK_OK")
""")
    )
    executed = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    assert any(
        "OFFLINE_NOTEBOOK_OK" in output.get("text", "") for output in executed.cells[-1].outputs
    )
