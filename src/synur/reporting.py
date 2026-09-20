"""Local per-transcript reports using the same alignment and scores as evaluation."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from synur.evaluation import evaluate
from synur.observations import VALUE_TYPES, SchemaRegistry, _reference_items, normalize_references

ERROR_TAGS = {
    "correct": "COR",
    "deletions": "DEL",
    "insertions": "INS",
    "substitutions": "SUB",
}


def _scope_references(
    rows: list[dict], registry: SchemaRegistry
) -> tuple[list[dict], list[list], tuple[str, ...]]:
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("rows must be a list of source row objects.")
    active_types = {concept.value_type for concept in registry.concepts}
    excluded_types = tuple(kind for kind in VALUE_TYPES if kind not in active_types)
    scoped_rows = []
    skipped_rows = []
    for row in rows:
        if "observations" not in row:
            raise ValueError(f"Source row {row.get('id')!r} is missing observations.")
        items, container_error = _reference_items(row["observations"])
        included = []
        skipped = []
        for item in items:
            if (
                container_error is None
                and isinstance(item, dict)
                and item.get("value_type") in excluded_types
            ):
                skipped.append(item)
            else:
                included.append(item)
        scoped_rows.append({
            **row, "observations": row["observations"] if container_error else included,
        })
        skipped_rows.append(skipped)
    return scoped_rows, skipped_rows, excluded_types


def _provenance(observation: object, record: dict) -> dict:
    concept_id = observation.get("id") if isinstance(observation, dict) else None
    audits = [
        entry for entry in record["audit"]
        if isinstance(concept_id, str) and isinstance(entry, dict)
        and entry.get("id") == concept_id
    ]
    question_ids: set[str] = set()
    for audit in audits:
        decisions = audit.get("decisions")
        if isinstance(decisions, list):
            question_ids.update(
                decision["question_id"] for decision in decisions
                if isinstance(decision, dict) and isinstance(decision.get("question_id"), str)
            )
    requests = record.get("requests", [])
    if not isinstance(requests, list) or not all(isinstance(item, dict) for item in requests):
        raise ValueError(f"Prediction {record['id']!r}.requests must be a list of objects.")
    linked_requests = []
    for request in requests:
        ids = request.get("question_ids", [])
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError(f"Prediction {record['id']!r} request question_ids must be strings.")
        if question_ids.intersection(ids):
            linked_requests.append(request)
    return deepcopy({
        "audit_status": "missing" if not audits else "recorded" if len(audits) == 1 else "ambiguous",
        "audit": audits,
        "requests": linked_requests,
        "metadata": record.get("metadata", {}),
    })


def build_transcript_report(
    rows: list[dict],
    predictions: list[dict],
    registry: SchemaRegistry,
    *,
    reference_view: Literal["raw", "normalized"] = "normalized",
) -> dict:
    """Report the selected rows/types without making any model calls.

    Pass original source rows with all labels, plus predictions and the active
    registry. References whose declared type is excluded from that registry
    remain visible as SKIP, without affecting scores. Each transcript is scored
    independently. Missing or failed records have null scored tags, counts, and
    rates, not fabricated deletions. Successful empty predictions are deletions.
    Micro rates pool observation counts across all requested rows; missing and
    failed rows affect recall once any requested extraction is available.

    Per-prediction provenance copies recorded audit entries (including evidence,
    reasons, and decisions), linked requests, and metadata. It never infers a
    span from text or invents a model explanation. Unrequested predictions and
    invalid labels remain visible in diagnostics.
    """
    if reference_view not in ("raw", "normalized"):
        raise ValueError("reference_view must be raw or normalized.")
    scoped_rows, skipped_rows, excluded_types = _scope_references(rows, registry)
    aggregate = evaluate(scoped_rows, predictions, registry)
    transcripts = []
    for row, scoped_row, skipped in zip(rows, scoped_rows, skipped_rows, strict=True):
        if not isinstance(row.get("transcript"), str):
            raise ValueError(f"Row {row['id']!r} requires a transcript string.")
        split = row.get("split", "")
        # Validate identities across the whole sample before matching a single row.
        records = [
            record for record in predictions
            if record["id"] == row["id"] and record.get("split", split) == split
        ]
        record = records[0] if records else None
        evaluation = evaluate([scoped_row], records, registry)
        score = evaluation[reference_view]
        comparisons = []
        expected = []
        predicted = []
        if evaluation["available"]:
            for category, tag in ERROR_TAGS.items():
                for entry in score["alignment"][category]:
                    reference = deepcopy(entry.get("reference"))
                    prediction = deepcopy(entry.get("prediction"))
                    comparison_index = len(comparisons)
                    comparisons.append({
                        "error_type": tag,
                        "expected_observation": reference,
                        "predicted_observation": prediction,
                    })
                    if "reference" in entry:
                        expected.append({
                            "observation": reference,
                            "error_type": tag,
                            "comparison_index": comparison_index,
                        })
                    if "prediction" in entry:
                        predicted.append({
                            "observation": prediction,
                            "error_type": tag,
                            "comparison_index": comparison_index,
                        })
            error_counts = {
                tag: score["edit_counts"][category] for category, tag in ERROR_TAGS.items()
            }
        else:
            references = (
                normalize_references(scoped_row["observations"], registry).observations
                if reference_view == "normalized"
                else _reference_items(scoped_row["observations"])[0]
            )
            expected = [
                {"observation": deepcopy(item), "error_type": None, "comparison_index": None}
                for item in references
            ]
            predicted = [
                {"observation": deepcopy(item), "error_type": None, "comparison_index": None}
                for item in record["observations"]
            ] if record is not None else []
            error_counts = None
        for item in skipped:
            expected.append({
                "observation": deepcopy(item),
                "error_type": "SKIP",
                "comparison_index": len(comparisons),
            })
            comparisons.append({
                "error_type": "SKIP",
                "expected_observation": deepcopy(item),
                "predicted_observation": None,
            })
        if record is not None:
            for item in predicted:
                item["provenance"] = _provenance(item["observation"], record)
        transcripts.append({
            "id": row["id"],
            "split": split,
            "transcript": row["transcript"],
            "prediction_status": record["status"] if record is not None else "missing",
            "available": evaluation["available"],
            "unavailable_reason": (
                None if evaluation["available"]
                else "Prediction failed." if record is not None
                else "No prediction record is available."
            ),
            "expected_observations": expected,
            "predicted_observations": predicted,
            "comparisons": comparisons,
            "error_counts": error_counts,
            "skipped_expected_count": len(skipped),
            **{name: score["observation"][name] for name in ("precision", "recall", "f1")},
            "reference_changes": evaluation["reference_changes"],
            "reference_issues": evaluation["reference_issues"],
            "prediction_issues": evaluation["prediction_issues"],
            "failures": deepcopy(record["failures"]) if record is not None else [],
        })
    return {
        "format_version": 3,
        "reference_view": reference_view,
        "enabled_value_types": aggregate["value_types"],
        "excluded_value_types": list(excluded_types),
        "error_types": {
            "COR": "Exact valid observation match.",
            "DEL": "Expected observation without a matching prediction.",
            "INS": "Extra predicted observation without an expected label.",
            "SUB": "Same concept ID, but a different value or invalid observation.",
            "SKIP": "Reference type is disabled; retained for context but excluded from scoring.",
        },
        "transcripts": transcripts,
        "prediction_issues": aggregate["prediction_issues"],
        "micro_metrics": {
            "available": aggregate["available"],
            "transcript_count": len(rows),
            "evaluated_transcript_count": sum(item["available"] for item in transcripts),
            "skipped_expected_count": sum(len(items) for items in skipped_rows),
            **{
                name: aggregate[reference_view]["observation"][name]
                if aggregate["available"] else None
                for name in ("tp", "fp", "fn", "precision", "recall", "f1")
            },
        },
    }


def build_short_transcript_report(report: dict) -> dict:
    """Keep comparisons and metrics from a full report without recalculating scores."""
    return deepcopy({
        "transcripts": [
            {
                "id": entry["id"],
                "transcript": entry["transcript"],
                "comparisons": entry["comparisons"],
                "metrics": {
                    name: entry[name]
                    for name in (
                        "available", "error_counts", "skipped_expected_count",
                        "precision", "recall", "f1",
                    )
                },
            }
            for entry in report["transcripts"]
        ],
        "micro_metrics": report["micro_metrics"],
    })


def save_transcript_report(
    output_dir: Path,
    rows: list[dict],
    predictions: list[dict],
    registry: SchemaRegistry,
    *,
    reference_view: Literal["raw", "normalized"] = "normalized",
    metadata: dict | None = None,
) -> Path:
    """Write full and short JSON reports to a new directory; return the full report path."""
    report = build_transcript_report(rows, predictions, registry, reference_view=reference_view)
    reports = {
        "transcript_report.json": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata if metadata is not None else {},
            **report,
        },
        "transcript_report_short.json": build_short_transcript_report(report),
    }
    payloads = {
        name: json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False)
        for name, content in reports.items()
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        path = output_dir / name
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(path)
    return output_dir / "transcript_report.json"
