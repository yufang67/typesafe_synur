"""Download the pinned public SYNUR snapshot separately from local-only loading.

Usage: python scripts\\download_synur.py --output data\\synur

Only the Python standard library is required. Existing snapshots are reused only
after complete manifest validation; corruption is an error, not an automatic repair.
An interrupted first download can be rerun: without a manifest every source file
is downloaded again rather than trusting partially downloaded content.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

# Make the repository's stdlib-only data module available without package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synur.dataset import (  # noqa: E402
    DATASET_REVISION,
    DATASET_SOURCE,
    MANIFEST_FILENAME,
    SOURCE_FILES,
    DatasetError,
    file_record,
    load_dataset,
    local_directory,
    local_path,
    source_url,
)

MAX_FILE_BYTES = 32 * 1024 * 1024


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "synur-local-download/1"})
    try:
        with urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise DatasetError(f"Download returned HTTP {response.status}: {url}")
            content = response.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise DatasetError(f"Download exceeds {MAX_FILE_BYTES} bytes: {url}")
            length = response.headers.get("Content-Length")
            if length is not None:
                try:
                    expected_length = int(length)
                except ValueError as exc:
                    raise DatasetError(f"Invalid Content-Length for {url}: {length!r}") from exc
                if expected_length != len(content):
                    raise DatasetError(f"Incomplete download for {url}")
            return content
    except (URLError, OSError, HTTPException) as exc:
        raise DatasetError(f"Download failed for {url}: {exc}") from exc


def _atomic_write(root: Path, filename: str, content: bytes) -> None:
    destination = local_path(root, filename)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=root, prefix=f".{filename}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        local_path(root, filename)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def download_dataset(output: Path) -> dict[str, Any]:
    """Persist a complete manifest last, or verify and reuse an existing snapshot."""
    try:
        root = local_directory(output)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = local_path(root, MANIFEST_FILENAME)
        # Preflight every destination before network access or replacing any files.
        for filename in SOURCE_FILES:
            local_path(root, filename)
        if manifest_path.exists():
            dataset = load_dataset(root)
            print(f"Verified cached SYNUR snapshot: {root}")
            return dataset.manifest

        print(f"Downloading pinned SYNUR {DATASET_REVISION} to {root}")
        print("No manifest exists; all source files will be fetched, not reused.")
        records = []
        for filename in SOURCE_FILES:
            content = _fetch(source_url(filename))
            record = file_record(filename, content)
            _atomic_write(root, filename, content)
            records.append(record)
            print(
                f"  {filename}: {record['byte_count']} bytes"
                + (f", {record['row_count']} rows" if record["row_count"] is not None else "")
            )
        manifest = {
            "format_version": 1,
            "source": DATASET_SOURCE,
            "revision": DATASET_REVISION,
            "files": records,
        }
        # Recheck persisted bytes before publishing the sole completeness marker.
        for record in records:
            filename = record["filename"]
            if file_record(filename, local_path(root, filename).read_bytes()) != record:
                raise DatasetError(f"File changed during download: {filename}")
        _atomic_write(
            root, MANIFEST_FILENAME, (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        )
        print(f"Saved verified manifest: {manifest_path}")
        return manifest
    except OSError as exc:
        raise DatasetError(f"Cannot write dataset to {output}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data") / "synur")
    args = parser.parse_args()
    try:
        download_dataset(args.output)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
