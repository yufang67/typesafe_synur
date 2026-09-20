"""Offline fixtures exercise both standalone setup and strictly local loading."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import URLError

import pytest

from synur.dataset import (
    DATASET_REVISION,
    EXPECTED_COUNTS,
    MANIFEST_FILENAME,
    SCHEMA_COUNT,
    SCHEMA_FILENAME,
    SOURCE_FILES,
    SPLIT_FILENAMES,
    DatasetError,
    load_dataset,
    source_url,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_synur.py"
spec = importlib.util.spec_from_file_location("download_synur", SCRIPT)
assert spec is not None and spec.loader is not None
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


@pytest.fixture
def sources() -> dict[str, bytes]:
    anomalous = [
        {"id": "001", "name": "Text", "value_type": "STRING", "value": 42},
        {"id": "1", "name": "Text", "value_type": "SINGLE_SELECT", "value": 7},
        {"id": "1", "name": "Text", "value_type": "STRING", "value": "unrecognized"},
        {"id": "999", "name": "Unknown", "value_type": "NUMERIC", "value": "ambiguous"},
        {"id": "2", "name": "List", "value_type": "MULTI_SELECT", "value": ["odd", "odd"]},
    ]
    result = {"README.md": b"# Local test fixture\n"}
    result[SCHEMA_FILENAME] = _json_bytes(
        [
            {"id": str(index), "name": f"Concept {index}", "value_type": "STRING"}
            for index in range(SCHEMA_COUNT)
        ]
    )
    for filename, split in SPLIT_FILENAMES.items():
        result[filename] = b"".join(
            _json_bytes(
                {
                    "id": str(index),
                    "transcript": "Original transcript\r\nwith punctuation.",
                    "observations": json.dumps(anomalous if index == 0 else []),
                    "extra": {"unchanged": True},
                }
            )
            for index in range(EXPECTED_COUNTS[split])
        )
    return result


@pytest.fixture
def fake_network(monkeypatch, sources):
    calls: list[str] = []
    by_url = {source_url(filename): content for filename, content in sources.items()}

    def fetch(url: str) -> bytes:
        calls.append(url)
        return by_url[url]

    monkeypatch.setattr(downloader, "_fetch", fetch)
    return calls


def _rewrite_manifest(root: Path, transform) -> None:
    path = root / MANIFEST_FILENAME
    manifest = json.loads(path.read_bytes())
    transform(manifest)
    path.write_bytes(_json_bytes(manifest))


def _replace_source(root: Path, filename: str, raw: bytes) -> None:
    (root / filename).write_bytes(raw)

    def update(manifest):
        record = next(item for item in manifest["files"] if item["filename"] == filename)
        record.update(sha256=hashlib.sha256(raw).hexdigest(), byte_count=len(raw))

    _rewrite_manifest(root, update)


def test_fresh_download_manifest_and_preservation(tmp_path, sources, fake_network):
    manifest = downloader.download_dataset(tmp_path)
    dataset = load_dataset(tmp_path)
    assert dataset.manifest == manifest
    assert manifest["source"] == "microsoft/SYNUR"
    assert manifest["revision"] == DATASET_REVISION
    assert len(dataset.schema_entries) == 193
    assert {split: len(rows) for split, rows in dataset.splits.items()} == EXPECTED_COUNTS
    assert len(fake_network) == len(SOURCE_FILES)
    for record in manifest["files"]:
        raw = sources[record["filename"]]
        assert (tmp_path / record["filename"]).read_bytes() == raw
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["byte_count"] == len(raw)
        assert record["source_url"] == source_url(record["filename"])
    raw_row = json.loads(sources[next(iter(SPLIT_FILENAMES))].splitlines()[0])
    row = dataset.splits["mediqa_synur_train"][0]
    assert row == {**raw_row, "observations": json.loads(raw_row["observations"])}
    assert row["observations"][0]["id"] == "001"
    assert row["observations"][0]["value"] == 42
    assert row["observations"][1]["value"] == 7
    assert len(row["observations"]) == 5
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_and_loader_never_network(tmp_path, fake_network, monkeypatch):
    first = downloader.download_dataset(tmp_path)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail("Network access was attempted")

    monkeypatch.setattr(downloader, "_fetch", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    assert downloader.download_dataset(tmp_path) == first
    assert load_dataset(tmp_path).manifest == first
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()
    }


def test_unicode_separators_and_encoding_anomalies_preserved(tmp_path, sources, fake_network):
    downloader.download_dataset(tmp_path)
    filename = next(iter(SPLIT_FILENAMES))
    rows = [json.loads(line) for line in sources[filename].splitlines()]
    rows[0]["transcript"] = "Text with \u2028 and \u0085 inside a JSON string."
    observations = [{"id": "001", "value": "\u00c3\u00a9", "extra": "\u2029"}]
    rows[0]["observations"] = json.dumps(observations, ensure_ascii=False)
    raw = b"".join(_json_bytes(row) for row in rows)
    _replace_source(tmp_path, filename, raw)
    loaded = load_dataset(tmp_path).splits["mediqa_synur_train"][0]
    assert loaded["transcript"] == rows[0]["transcript"]
    assert loaded["observations"] == observations
    assert (tmp_path / filename).read_bytes() == raw


@pytest.mark.parametrize("filename", [MANIFEST_FILENAME, SCHEMA_FILENAME, *SPLIT_FILENAMES, "README.md"])
def test_missing_files_actionable(tmp_path, fake_network, filename):
    downloader.download_dataset(tmp_path)
    (tmp_path / filename).unlink()
    with pytest.raises(DatasetError, match=r"download_synur\.py"):
        load_dataset(tmp_path)


def test_missing_directory_actionable(tmp_path):
    with pytest.raises(DatasetError, match=r"download_synur\.py"):
        load_dataset(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_missing_cached_source_does_not_trigger_network(tmp_path, fake_network):
    downloader.download_dataset(tmp_path)
    (tmp_path / SCHEMA_FILENAME).unlink()
    fake_network.clear()
    with pytest.raises(DatasetError, match="Cannot read"):
        downloader.download_dataset(tmp_path)
    assert not fake_network


@pytest.mark.parametrize("same_size", [True, False])
def test_corrupt_cache_is_not_reused_or_refetched(tmp_path, fake_network, same_size):
    downloader.download_dataset(tmp_path)
    filename = next(iter(SPLIT_FILENAMES))
    path = tmp_path / filename
    raw = path.read_bytes()
    path.write_bytes(b"!" + raw[1:] if same_size else raw + b" ")
    fake_network.clear()
    for operation in (load_dataset, downloader.download_dataset):
        with pytest.raises(DatasetError, match="checksum mismatch|Byte count mismatch"):
            operation(tmp_path)
    assert fake_network == []


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(revision="other"),
        lambda m: m.update(source="other/SYNUR"),
        lambda m: m.update(format_version=True),
        lambda m: m["files"].pop(),
        lambda m: m["files"].append(m["files"][0]),
        lambda m: m["files"].__setitem__(1, m["files"][0]),
        lambda m: m["files"][0].update(filename="../outside.jsonl"),
        lambda m: m["files"][0].update(filename="C:\\outside.jsonl"),
        lambda m: m["files"][0].update(source_url="https://example.invalid/data"),
        lambda m: m["files"][0].update(sha256="not-a-hash"),
        lambda m: m["files"][0].update(byte_count=True),
        lambda m: m["files"][0].update(row_count=0),
    ],
)
def test_untrusted_manifest_rejected_before_network(tmp_path, fake_network, change):
    downloader.download_dataset(tmp_path)
    _rewrite_manifest(tmp_path, change)
    fake_network.clear()
    with pytest.raises(DatasetError):
        downloader.download_dataset(tmp_path)
    assert fake_network == []


@pytest.mark.parametrize("raw", [b"{", b"[]", b"\xff", b'{"files":[],"files":[]}'])
def test_malformed_manifest(tmp_path, fake_network, raw):
    downloader.download_dataset(tmp_path)
    (tmp_path / MANIFEST_FILENAME).write_bytes(raw)
    with pytest.raises(DatasetError):
        load_dataset(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: rows[0].update(id=3),
        lambda rows: rows[0].update(transcript=None),
        lambda rows: rows[0].update(observations=[]),
        lambda rows: rows[0].update(observations="{"),
        lambda rows: rows[0].update(observations="{}"),
        lambda rows: rows[0].update(observations="[null]"),
        lambda rows: rows[0].update(observations='[{"value":NaN}]'),
        lambda rows: rows[1].update(id=rows[0]["id"]),
        lambda rows: rows.pop(),
    ],
)
def test_malformed_rows_even_with_matching_checksum(tmp_path, sources, fake_network, change):
    downloader.download_dataset(tmp_path)
    filename = next(iter(SPLIT_FILENAMES))
    rows = [json.loads(line) for line in sources[filename].splitlines()]
    change(rows)
    _replace_source(tmp_path, filename, b"".join(_json_bytes(row) for row in rows))
    with pytest.raises(DatasetError):
        load_dataset(tmp_path)


@pytest.mark.parametrize("raw", [b"", b"\xff", b"not json\n", b"\n", b'{"id":"1","id":"2"}\n'])
def test_malformed_source_bytes(tmp_path, fake_network, raw):
    downloader.download_dataset(tmp_path)
    _replace_source(tmp_path, next(iter(SPLIT_FILENAMES)), raw)
    with pytest.raises(DatasetError):
        load_dataset(tmp_path)


@pytest.mark.parametrize("kind", ["count", "entry", "duplicate", "not-list"])
def test_malformed_schema(tmp_path, sources, fake_network, kind):
    downloader.download_dataset(tmp_path)
    schema = json.loads(sources[SCHEMA_FILENAME])
    if kind == "count":
        schema.pop()
    elif kind == "entry":
        schema[0]["id"] = 0
    elif kind == "duplicate":
        schema[1]["id"] = schema[0]["id"]
    else:
        schema = {}
    _replace_source(tmp_path, SCHEMA_FILENAME, _json_bytes(schema))
    with pytest.raises(DatasetError):
        load_dataset(tmp_path)


def test_failed_download_has_no_trusted_manifest_and_can_resume(
    tmp_path, sources, monkeypatch
):
    calls = 0

    def interrupted(url):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DatasetError("Simulated download failure")
        return sources[next(iter(SPLIT_FILENAMES))]

    monkeypatch.setattr(downloader, "_fetch", interrupted)
    with pytest.raises(DatasetError, match="Simulated"):
        downloader.download_dataset(tmp_path)
    assert not (tmp_path / MANIFEST_FILENAME).exists()
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(DatasetError):
        load_dataset(tmp_path)
    urls: list[str] = []
    by_url = {source_url(filename): raw for filename, raw in sources.items()}

    def resume(url):
        urls.append(url)
        return by_url[url]

    monkeypatch.setattr(downloader, "_fetch", resume)
    downloader.download_dataset(tmp_path)
    assert len(urls) == len(SOURCE_FILES)
    assert len(load_dataset(tmp_path).schema_entries) == 193


def test_malformed_download_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "_fetch", lambda url: b"not JSON\n")
    with pytest.raises(DatasetError):
        downloader.download_dataset(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_atomic_replace_failure_cleans_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / "README.md"
    destination.write_bytes(b"original")

    def fail_replace(*args):
        raise OSError("Simulated replacement failure")

    monkeypatch.setattr(downloader.os, "replace", fail_replace)
    with pytest.raises(OSError, match="Simulated"):
        downloader._atomic_write(tmp_path, "README.md", b"replacement")
    assert destination.read_bytes() == b"original"
    assert [path.name for path in tmp_path.iterdir()] == ["README.md"]


def test_manifest_publication_failure_leaves_no_trust_marker(tmp_path, fake_network, monkeypatch):
    replace = downloader.os.replace

    def fail_manifest(source, destination):
        if Path(destination).name == MANIFEST_FILENAME:
            raise OSError("Simulated manifest publication failure")
        replace(source, destination)

    monkeypatch.setattr(downloader.os, "replace", fail_manifest)
    with pytest.raises(DatasetError, match="manifest publication"):
        downloader.download_dataset(tmp_path)
    assert not (tmp_path / MANIFEST_FILENAME).exists()
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(DatasetError):
        load_dataset(tmp_path)


def test_symlink_file_cannot_escape_output(tmp_path, fake_network):
    outside = tmp_path / "outside"
    outside.write_bytes(b"do not modify")
    root = tmp_path / "dataset"
    root.mkdir()
    try:
        (root / "README.md").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    with pytest.raises(DatasetError, match="link"):
        downloader.download_dataset(root)
    assert outside.read_bytes() == b"do not modify"
    assert not fake_network


@pytest.mark.parametrize("error", [URLError("offline fixture"), IncompleteRead(b"partial", 20)])
def test_network_failure_is_explicit(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(downloader, "urlopen", fail)
    with pytest.raises(DatasetError, match="Download failed"):
        downloader._fetch(source_url("README.md"))


@pytest.mark.parametrize(
    ("status", "length", "content", "message"),
    [
        (503, None, b"error", "HTTP 503"),
        (200, "100", b"short", "Incomplete download"),
        (200, "nonnumeric", b"content", "Invalid Content-Length"),
        (200, None, b"x" * 21, "exceeds"),
    ],
)
def test_http_response_validation(monkeypatch, status, length, content, message):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return content[:limit]

    response = Response()
    response.status = status
    response.headers = {} if length is None else {"Content-Length": length}
    monkeypatch.setattr(downloader, "urlopen", lambda *args, **kwargs: response)
    monkeypatch.setattr(downloader, "MAX_FILE_BYTES", 20)
    with pytest.raises(DatasetError, match=message):
        downloader._fetch(source_url("README.md"))


def test_cli_reports_failure(monkeypatch, capsys, tmp_path):
    def fail(output):
        raise DatasetError("Simulated failure")

    monkeypatch.setattr(downloader, "download_dataset", fail)
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--output", str(tmp_path)])
    assert downloader.main() == 1
    assert "ERROR: Simulated failure" in capsys.readouterr().err
