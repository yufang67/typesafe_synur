"""Read verified local SYNUR files without fetching or normalizing any labels.

Create the local snapshot separately with ``scripts\\download_synur.py``.
The manifest is published only after all six source files have been validated.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_SOURCE = "microsoft/SYNUR"
DATASET_REVISION = "c7f79af4dcc8e5fb175c40cef0592d85a76bf11c"
EXPECTED_COUNTS = {
    "mediqa_synur_train": 122,
    "mediqa_synur_dev": 101,
    "mediqa_synur_test": 199,
    "original": 223,
}
SCHEMA_COUNT = 193
SCHEMA_FILENAME = "synur_schema.json"
MANIFEST_FILENAME = "manifest.json"
SPLIT_FILENAMES = {
    f"{split}-00000-of-00001.jsonl": split for split in EXPECTED_COUNTS
}
SOURCE_FILES = {
    **{filename: f"data/{filename}" for filename in SPLIT_FILENAMES},
    SCHEMA_FILENAME: SCHEMA_FILENAME,
    "README.md": "README.md",
}


class DatasetError(ValueError):
    """Missing, malformed, or untrusted local dataset data."""

    def __init__(self, message: str) -> None:
        super().__init__(
            f"{message}\n"
            "Run the separate setup command: "
            "python scripts\\download_synur.py --output <data_dir> "
            "(normally data\\synur). For corrupt files or a corrupt manifest, "
            "move/remove the affected output directory before downloading again."
        )


@dataclass
class LocalDataset:
    schema_entries: list[dict[str, Any]]
    splits: dict[str, list[dict[str, Any]]]
    manifest: dict[str, Any]


def source_url(filename: str) -> str:
    """Return the pinned upstream URL for an allowlisted local filename."""
    if filename not in SOURCE_FILES:
        raise DatasetError(f"Unexpected dataset filename: {filename!r}")
    return (
        f"https://huggingface.co/datasets/{DATASET_SOURCE}/resolve/"
        f"{DATASET_REVISION}/{SOURCE_FILES[filename]}"
    )


def local_directory(data_dir: Path) -> Path:
    """Reject directory links rather than following them outside the chosen root."""
    data_dir = Path(data_dir).absolute()
    # resolve() also detects Windows junctions and linked ancestor directories.
    if data_dir.resolve() != data_dir or data_dir.is_symlink():
        raise DatasetError(f"Dataset directory must not traverse links: {data_dir}")
    if data_dir.exists() and not data_dir.is_dir():
        raise DatasetError(f"Dataset directory is not a directory: {data_dir}")
    return data_dir


def local_path(data_dir: Path, filename: str) -> Path:
    """Resolve only fixed, flat filenames, never paths supplied by a manifest."""
    if filename not in SOURCE_FILES and filename != MANIFEST_FILENAME:
        raise DatasetError(f"Unexpected dataset filename: {filename!r}")
    path = data_dir / filename
    if path.is_symlink() or path.resolve() != path:
        raise DatasetError(f"Dataset file must not be a link: {path}")
    if path.exists() and not path.is_file():
        raise DatasetError(f"Dataset path is not a regular file: {path}")
    return path


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant {value!r}")


def _json(text: str, context: str) -> Any:
    try:
        return json.loads(
            text, object_pairs_hook=_object_pairs, parse_constant=_invalid_constant
        )
    except ValueError as exc:
        raise DatasetError(f"Malformed JSON in {context}: {exc}") from exc


def validate_source_file(
    filename: str, raw: bytes
) -> tuple[int | None, list[dict[str, Any]] | None]:
    """Validate source structure, not clinical labels; retain all observation values."""
    if filename not in SOURCE_FILES:
        raise DatasetError(f"Unexpected dataset filename: {filename!r}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"Invalid UTF-8 in {filename}: {exc}") from exc
    if not text.strip():
        raise DatasetError(f"Empty source file: {filename}")
    if filename == "README.md":
        return None, None
    if filename == SCHEMA_FILENAME:
        schema = _json(text, filename)
        if not isinstance(schema, list) or len(schema) != SCHEMA_COUNT:
            raise DatasetError(f"{filename} must contain {SCHEMA_COUNT} schema entries")
        seen: set[str] = set()
        for index, entry in enumerate(schema):
            if not isinstance(entry, dict) or any(
                not isinstance(entry.get(field), str) or not entry[field]
                for field in ("id", "name", "value_type")
            ):
                raise DatasetError(f"Malformed schema entry {index} in {filename}")
            if entry["id"] in seen:
                raise DatasetError(f"Duplicate schema id {entry['id']!r} in {filename}")
            seen.add(entry["id"])
        return len(schema), schema

    split = SPLIT_FILENAMES[filename]
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    for line_number, line in enumerate(lines, start=1):
        context = f"{filename}, line {line_number}"
        row = _json(line, context)
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or not isinstance(row.get("transcript"), str)
            or not isinstance(row.get("observations"), str)
        ):
            raise DatasetError(
                f"Malformed row in {context}; id, transcript and raw observations "
                "must be strings, with a nonempty id"
            )
        if row["id"] in seen_ids:
            raise DatasetError(f"Duplicate row id {row['id']!r} in {context}")
        seen_ids.add(row["id"])
        observations = _json(row["observations"], f"observations in {context}")
        if not isinstance(observations, list) or any(
            not isinstance(observation, dict) for observation in observations
        ):
            raise DatasetError(f"Observations must be a list of objects in {context}")
        # Do not deduplicate concepts, coerce values/IDs, or repair encoding artifacts.
        rows.append({**row, "observations": observations})
    if len(rows) != EXPECTED_COUNTS[split]:
        raise DatasetError(
            f"{filename} has {len(rows)} rows; expected {EXPECTED_COUNTS[split]}"
        )
    return len(rows), rows


def file_record(filename: str, raw: bytes) -> dict[str, Any]:
    """Validate bytes and describe the exact unmodified upstream content."""
    row_count, _ = validate_source_file(filename, raw)
    return {
        "filename": filename,
        "source_url": source_url(filename),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "row_count": row_count,
    }


def _read_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"Cannot read local dataset file {path}: {exc}") from exc


def _validate_manifest(raw: bytes) -> dict[str, Any]:
    try:
        manifest = _json(raw.decode("utf-8"), MANIFEST_FILENAME)
    except UnicodeDecodeError as exc:
        raise DatasetError(f"Invalid UTF-8 in {MANIFEST_FILENAME}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DatasetError("Manifest must be a JSON object")
    if (
        type(manifest.get("format_version")) is not int
        or manifest["format_version"] != 1
        or manifest.get("source") != DATASET_SOURCE
        or manifest.get("revision") != DATASET_REVISION
    ):
        raise DatasetError("Manifest version, source, or pinned revision does not match")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != len(SOURCE_FILES):
        raise DatasetError("Manifest must describe every expected source file exactly once")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise DatasetError("Manifest file records must be objects")
        filename = record.get("filename")
        if not isinstance(filename, str) or filename not in SOURCE_FILES or filename in seen:
            raise DatasetError(f"Unexpected or duplicate manifest filename: {filename!r}")
        seen.add(filename)
        checksum = record.get("sha256")
        size = record.get("byte_count")
        count = record.get("row_count")
        expected_count = (
            EXPECTED_COUNTS[SPLIT_FILENAMES[filename]]
            if filename in SPLIT_FILENAMES
            else SCHEMA_COUNT if filename == SCHEMA_FILENAME else None
        )
        if (
            record.get("source_url") != source_url(filename)
            or not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            or type(size) is not int
            or size <= 0
            or "row_count" not in record
            or type(count) is not type(expected_count)
            or count != expected_count
        ):
            raise DatasetError(f"Malformed manifest record for {filename}")
    return manifest


def load_dataset(data_dir: Path) -> LocalDataset:
    """Load a complete, verified local snapshot; this function never uses the network.

    Row IDs need only be unique within each split. Observation anomalies remain
    untouched for downstream diagnostics. Raw files are never modified.
    """
    try:
        root = local_directory(data_dir)
        manifest = _validate_manifest(_read_file(local_path(root, MANIFEST_FILENAME)))
        schema_entries: list[dict[str, Any]] = []
        splits: dict[str, list[dict[str, Any]]] = {}
        for record in manifest["files"]:
            filename = record["filename"]
            raw = _read_file(local_path(root, filename))
            if len(raw) != record["byte_count"]:
                raise DatasetError(f"Byte count mismatch for {filename}")
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise DatasetError(f"SHA-256 checksum mismatch for {filename}")
            row_count, entries = validate_source_file(filename, raw)
            if row_count != record["row_count"]:
                raise DatasetError(f"Row count mismatch for {filename}")
            if filename == SCHEMA_FILENAME:
                assert entries is not None
                schema_entries = entries
            elif filename in SPLIT_FILENAMES:
                assert entries is not None
                splits[SPLIT_FILENAMES[filename]] = entries
        return LocalDataset(schema_entries=schema_entries, splits=splits, manifest=manifest)
    except OSError as exc:
        raise DatasetError(f"Cannot access dataset directory {data_dir}: {exc}") from exc
