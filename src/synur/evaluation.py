"""Local baseline metrics, not the official SYNUR shared-task scorer."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from synur.observations import (
    VALUE_TYPES,
    SchemaRegistry,
    _observation_errors,
    _reference_items,
    normalize_references,
)


@dataclass
class _Label:
    raw: object
    concept_id: str | None
    value_type: str
    key: tuple | None


def _canonical(value_type: str, value: object) -> object:
    if value_type == "NUMERIC":
        return Decimal(str(value))
    if value_type == "MULTI_SELECT":
        assert isinstance(value, list)
        return tuple(sorted(value))
    return value


def _label(raw: object, registry: SchemaRegistry, *, invalid: bool = False) -> _Label:
    fields = raw if isinstance(raw, dict) else {}
    concept_id = fields.get("id")
    value_type = fields.get("value_type")
    valid = not invalid and not _observation_errors(raw, registry)
    key = None
    if valid:
        assert isinstance(value_type, str)
        key = (concept_id, value_type, _canonical(value_type, fields["value"]))
    return _Label(
        deepcopy(raw),
        concept_id if isinstance(concept_id, str) else None,
        value_type if value_type in VALUE_TYPES else "INVALID",
        key,
    )


def _metric(tp: int, fp: int, fn: int, available: bool) -> dict:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if available and tp + fp else None,
        "recall": tp / (tp + fn) if available and tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if available and 2 * tp + fp + fn else None,
    }


def _counts(
    gold: list[_Label], predicted: list[_Label], key: Callable[[_Label], object]
) -> tuple[int, int, int]:
    gold_keys = Counter(key(label) for label in gold if label.key is not None)
    prediction_keys = Counter(key(label) for label in predicted if label.key is not None)
    tp = sum((gold_keys & prediction_keys).values())
    return tp, len(predicted) - tp, len(gold) - tp


def _unmatched(labels: list[_Label], other: list[_Label]) -> list[_Label]:
    remaining = Counter(label.key for label in other if label.key is not None)
    unmatched: list[_Label] = []
    for label in labels:
        if label.key is not None and remaining[label.key]:
            remaining[label.key] -= 1
        else:
            unmatched.append(label)
    return unmatched


def _score(
    rows: list[tuple[dict, list[_Label], list[_Label]]],
    available: bool,
    value_types: tuple[str, ...],
) -> dict:
    totals = [0, 0, 0]
    concepts = [0, 0, 0]
    observed_types = {label.value_type for _, gold, predicted in rows for label in gold + predicted}
    per_type = {
        value_type: [0, 0, 0]
        for value_type in (*VALUE_TYPES, "INVALID")
        if value_type in value_types or value_type in observed_types or value_type == "INVALID"
    }
    errors: dict[str, list[dict]] = {
        "false_positives": [],
        "missing_observations": [],
        "value_mismatches": [],
    }
    alignment: dict[str, list[dict]] = {
        "correct": [],
        "insertions": [],
        "deletions": [],
        "substitutions": [],
    }
    for context, gold, predicted in rows:
        counts = _counts(gold, predicted, lambda label: label.key)
        concept_counts = _counts(gold, predicted, lambda label: label.concept_id)
        for index in range(3):
            totals[index] += counts[index]
            concepts[index] += concept_counts[index]
        for value_type in per_type:
            type_counts = _counts(
                [label for label in gold if label.value_type == value_type],
                [label for label in predicted if label.value_type == value_type],
                lambda label: label.key,
            )
            for index in range(3):
                per_type[value_type][index] += type_counts[index]
        missing = _unmatched(gold, predicted)
        extra = _unmatched(predicted, gold)
        matches: dict[tuple, list[_Label]] = defaultdict(list)
        for label in predicted:
            if label.key is not None:
                matches[label.key].append(label)
        for label in gold:
            if label.key is not None and matches[label.key]:
                prediction = matches[label.key].pop(0)
                alignment["correct"].append(
                    {**context, "reference": label.raw, "prediction": prediction.raw}
                )
        for category, labels in (("missing_observations", missing), ("false_positives", extra)):
            errors[category].extend(
                {**context, "observation": label.raw, "invalid": label.key is None}
                for label in labels
            )
        missing_by_id: dict[str | None, list[_Label]] = defaultdict(list)
        for label in missing:
            missing_by_id[label.concept_id].append(label)
        for label in extra:
            if label.concept_id is not None and missing_by_id[label.concept_id]:
                reference = missing_by_id[label.concept_id].pop(0)
                substitution = {
                    **context,
                    "id": label.concept_id,
                    "reference": reference.raw,
                    "prediction": label.raw,
                }
                errors["value_mismatches"].append(substitution)
                alignment["substitutions"].append(substitution)
            else:
                alignment["insertions"].append(
                    {**context, "prediction": label.raw, "invalid": label.key is None}
                )
        for remaining in missing_by_id.values():
            alignment["deletions"].extend(
                {**context, "reference": label.raw, "invalid": label.key is None}
                for label in remaining
            )
    return {
        "observation": _metric(totals[0], totals[1], totals[2], available),
        "concept": _metric(concepts[0], concepts[1], concepts[2], available),
        "per_type": {
            name: _metric(counts[0], counts[1], counts[2], available)
            for name, counts in per_type.items()
        },
        "edit_counts": {name: len(items) for name, items in alignment.items()},
        "alignment": alignment,
        "errors": errors,
    }


def _identity(row: dict) -> tuple[str, str | int]:
    row_id = row.get("id")
    split = row.get("split", "")
    if isinstance(row_id, bool) or not isinstance(row_id, (str, int)):
        raise ValueError("Each source/prediction row requires a string or integer id.")
    if not isinstance(split, str):
        raise ValueError(f"Row {row_id!r}: split must be a string when supplied.")
    return split, row_id


def evaluate(rows: list[dict], predictions: list[dict], registry: SchemaRegistry) -> dict:
    """Score all requested rows locally, including missing and failed predictions.

    Observations match as multisets of (ID, type, value) within each row.
    Multi-select order is ignored. Finite numeric values compare as exact
    ``Decimal(str(value))`` numbers (1 equals 1.0, without tolerance or rounding);
    text and enum strings compare exactly, including case and whitespace.
    Concept scores match ID multisets of individually valid labels, ignoring
    value. Invalid labels remain false negatives/positives, never exclusions.
    Repeated golds remain separate labels; every duplicate prediction ID is
    rejected, including its first occurrence. Failed rows emit no predictions.

    Edit alignment first matches exact valid observations as correct (C), then
    pairs remaining labels with the same ID within a row as substitutions (S).
    Unpaired predictions are insertions (I); unpaired references are deletions
    (D). These categories are disjoint: TP=C, FP=I+S, FN=D+S. Each multi-select
    list is one observation, not independently scored members. Invalid labels
    cannot be correct; same-ID malformed pairs are also substitutions. Counts
    remain diagnostic, not model results, when ``available`` is False.

    Precision/recall with zero denominators are None; F1 is None only when
    2*TP+FP+FN is zero, otherwise its usual ratio (including 0). With no requested
    complete/partial prediction records, *all* score rates are None: an all-failed
    or no-call run has no extraction metrics. A successful empty extraction is
    still a prediction. Counts and reference diagnostics remain available.
    Missing/failed rows contribute all reference labels to end-to-end misses as
    soon as any requested complete/partial record makes scoring available.

    Row IDs must be unique within each split. A prediction without a split can
    match an ID only when it identifies one requested source row unambiguously.
    Extra prediction rows are reported and do not change the requested sample.
    Records are model outputs supplied by the caller; this function cannot infer
    provenance. Do not pass offline fixtures as evidence of real model accuracy.

    For a type-restricted experiment, pass its selected registry and reference
    label view. Coverage denominators use that registry. Per-type scores include
    selected types and any unexpected submitted types so invalid labels remain
    visible; omitted types with no submitted labels are not reported.

    Optional diagnostics.candidate_values maps concept IDs to transcript-derived
    value lists. Alternatively, diagnostics.candidate_pools maps STRING/NUMERIC
    types to shared transcript-derived value lists without repeating each list
    for every concept. Explicit per-concept lists override shared pools.
    Candidate coverage is evaluated posthoc against normalized valid
    STRING/NUMERIC references, independent of emitted observations; missing
    diagnostics are unavailable, not counted as candidate misses. Their expected
    and unavailable counts are explicit. Other diagnostics/metadata are retained.
    """
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("rows must be a list of source row objects.")
    if not isinstance(predictions, list) or not all(isinstance(row, dict) for row in predictions):
        raise ValueError("predictions must be a list of prediction record objects.")
    sources: dict[tuple[str, str | int], dict] = {}
    for row in rows:
        identity = _identity(row)
        if identity in sources:
            raise ValueError(f"Duplicate source row identity {identity!r}; evaluate it only once.")
        if "observations" not in row:
            raise ValueError(f"Source row {identity!r} is missing observations.")
        sources[identity] = row
    records: dict[tuple[str, str | int], dict] = {}
    prediction_issues: list[dict] = []
    for record in predictions:
        identity = _identity(record)
        if "split" not in record:
            matches = [key for key in sources if key[1] == identity[1]]
            if len(matches) > 1:
                raise ValueError(
                    f"Prediction ID {identity[1]!r} needs a split to disambiguate rows."
                )
            if matches:
                identity = matches[0]
        if identity not in sources:
            prediction_issues.append(
                {
                    "split": identity[0],
                    "row_id": identity[1],
                    "code": "unrequested_row",
                    "message": "Prediction lies outside the requested source sample.",
                }
            )
            continue
        if identity in records:
            raise ValueError(f"Duplicate prediction record for row {identity!r}.")
        if record.get("status") not in ("complete", "partial", "failed"):
            raise ValueError(
                f"Prediction {identity!r} status must be complete, partial, or failed."
            )
        for field in ("observations", "audit", "failures"):
            if not isinstance(record.get(field), list):
                raise ValueError(f"Prediction {identity!r}.{field} must be a list.")
        records[identity] = record
    available = any(record["status"] in ("complete", "partial") for record in records.values())
    coverage = {
        "requested_rows": len(rows),
        "prediction_rows": len(records),
        "complete_rows": 0,
        "partial_rows": 0,
        "failed_rows": 0,
        "missing_rows": len(rows) - len(records),
        "fully_audited_complete_rows": 0,
        "expected_concepts": len(rows) * len(registry.concepts),
        "audited_concepts": 0,
    }
    audit_counts = {"emitted": 0, "absent": 0, "review": 0, "failed": 0}
    submitted = 0
    valid_predictions = 0
    failures: list[dict] = []
    failure_counts: Counter[str] = Counter()
    reference_changes: list[dict] = []
    reference_issues: list[dict] = []
    diagnostic_rows: list[dict] = []
    candidates = {"expected": 0, "evaluated": 0, "covered": 0, "unavailable": 0}
    raw_rows: list[tuple[dict, list[_Label], list[_Label]]] = []
    normalized_rows: list[tuple[dict, list[_Label], list[_Label]]] = []
    by_id = registry.by_id
    for identity, row in sources.items():
        context = {"split": identity[0], "row_id": identity[1]}
        normalized = normalize_references(
            row["observations"], registry, split=identity[0], row_id=str(identity[1])
        )
        reference_changes.extend(normalized.changes)
        reference_issues.extend(normalized.issues)
        raw_items, _ = _reference_items(row["observations"])
        gold_raw = [_label(item, registry) for item in raw_items]
        gold_normalized = [_label(item, registry) for item in normalized.observations]
        record = records.get(identity)
        predicted: list[_Label] = []
        candidate_values: dict = {}
        candidate_pools: dict = {}
        if record is not None:
            status = record["status"]
            coverage[f"{status}_rows"] += 1
            observations = record["observations"]
            ids = Counter(
                item["id"]
                for item in observations
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            )
            for index, item in enumerate(observations):
                errors = _observation_errors(item, registry)
                duplicate = (
                    isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                    and (ids[item["id"]] > 1)
                )
                if duplicate:
                    errors.append(("id", "Duplicate predicted ID; all occurrences rejected."))
                submitted += 1
                if not errors:
                    valid_predictions += 1
                for field, message in errors:
                    prediction_issues.append(
                        {
                            **context,
                            "index": index,
                            "code": "invalid_observation",
                            "field": field,
                            "message": message,
                            "observation": deepcopy(item),
                        }
                    )
                if status != "failed":
                    predicted.append(_label(item, registry, invalid=bool(errors)))
            if status == "failed" and observations:
                prediction_issues.append(
                    {
                        **context,
                        "code": "failed_row_emissions",
                        "message": "Failed-row observations are not scored as successful emissions.",
                        "observations": deepcopy(observations),
                    }
                )
            audit: dict[str, str] = {}
            duplicate_audits: set[str] = set()
            for entry in record["audit"]:
                concept_id = entry.get("id") if isinstance(entry, dict) else None
                decision = entry.get("status") if isinstance(entry, dict) else None
                if (
                    not isinstance(concept_id, str)
                    or concept_id not in by_id
                    or (not isinstance(decision, str) or decision not in audit_counts)
                ):
                    prediction_issues.append(
                        {
                            **context,
                            "code": "invalid_audit",
                            "entry": deepcopy(entry),
                            "message": "Audit entries require a known concept ID and valid status.",
                        }
                    )
                elif concept_id in audit:
                    duplicate_audits.add(concept_id)
                    prediction_issues.append(
                        {
                            **context,
                            "code": "duplicate_audit",
                            "entry": deepcopy(entry),
                            "message": "Duplicate concept audit entries do not count as coverage.",
                        }
                    )
                else:
                    audit[concept_id] = decision
            for concept_id in duplicate_audits:
                audit.pop(concept_id)
            coverage["audited_concepts"] += len(audit)
            for decision in audit.values():
                audit_counts[decision] += 1
            if status == "complete":
                if len(audit) == len(by_id):
                    coverage["fully_audited_complete_rows"] += 1
                else:
                    prediction_issues.append(
                        {
                            **context,
                            "code": "incomplete_schema_audit",
                            "message": "Complete record did not audit every schema concept exactly once.",
                            "missing_ids": sorted(by_id.keys() - audit.keys()),
                        }
                    )
            emitted_ids = {
                label.concept_id
                for label in predicted
                if label.key is not None and label.concept_id is not None
            }
            audit_emitted = {
                concept_id for concept_id, decision in audit.items() if decision == "emitted"
            }
            if emitted_ids != audit_emitted:
                prediction_issues.append(
                    {
                        **context,
                        "code": "emission_audit_mismatch",
                        "message": "Valid emitted observations and emitted audit IDs disagree.",
                        "observation_only_ids": sorted(emitted_ids - audit_emitted),
                        "audit_only_ids": sorted(audit_emitted - emitted_ids),
                    }
                )
            for failure in record["failures"]:
                failures.append({**context, "failure": deepcopy(failure)})
                if isinstance(failure, dict):
                    kind = failure.get("kind", failure.get("stage", "unspecified"))
                    failure_counts[str(kind)] += 1
                else:
                    failure_counts["malformed"] += 1
                    prediction_issues.append(
                        {
                            **context,
                            "code": "invalid_failure",
                            "failure": deepcopy(failure),
                            "message": "Failure entries must be objects.",
                        }
                    )
            diagnostics = record.get("diagnostics", {})
            diagnostic_rows.append(
                {
                    **context,
                    "diagnostics": deepcopy(diagnostics),
                    "metadata": deepcopy(record.get("metadata", {})),
                }
            )
            if isinstance(diagnostics, dict):
                pools = diagnostics.get("candidate_pools", {})
                if isinstance(pools, dict):
                    for kind, choices in pools.items():
                        if kind in ("STRING", "NUMERIC") and isinstance(choices, list):
                            candidate_pools[kind] = choices
                        else:
                            prediction_issues.append(
                                {
                                    **context,
                                    "code": "invalid_candidate_pools",
                                    "message": "Candidate pools require STRING/NUMERIC keys and lists.",
                                }
                            )
                else:
                    prediction_issues.append(
                        {
                            **context,
                            "code": "invalid_candidate_pools",
                            "message": "diagnostics.candidate_pools must be an object.",
                        }
                    )
                values = diagnostics.get("candidate_values", {})
                if isinstance(values, dict):
                    candidate_values = values
                    for concept_id, choices in values.items():
                        concept = by_id.get(concept_id) if isinstance(concept_id, str) else None
                        if (
                            concept is None
                            or concept.value_type not in ("STRING", "NUMERIC")
                            or not isinstance(choices, list)
                        ):
                            prediction_issues.append(
                                {
                                    **context,
                                    "id": concept_id,
                                    "code": "invalid_candidate_values",
                                    "message": "Candidate values require a known STRING/NUMERIC ID "
                                    "mapped to a list.",
                                    "value": deepcopy(choices),
                                }
                            )
                else:
                    prediction_issues.append(
                        {
                            **context,
                            "code": "invalid_candidate_values",
                            "message": "diagnostics.candidate_values must map IDs to candidate lists.",
                        }
                    )
            else:
                prediction_issues.append(
                    {
                        **context,
                        "code": "invalid_diagnostics",
                        "message": "diagnostics must be an object when supplied.",
                    }
                )
        for label in gold_normalized:
            if label.key is None or label.value_type not in ("STRING", "NUMERIC"):
                continue
            candidates["expected"] += 1
            values = candidate_values.get(label.concept_id, candidate_pools.get(label.value_type))
            if not isinstance(values, list):
                candidates["unavailable"] += 1
                continue
            candidates["evaluated"] += 1
            candidate_keys = []
            for value in values:
                candidate = deepcopy(label.raw)
                assert isinstance(candidate, dict)
                candidate["value"] = value
                candidate_label = _label(candidate, registry)
                if candidate_label.key is not None:
                    candidate_keys.append(candidate_label.key)
                else:
                    prediction_issues.append(
                        {
                            **context,
                            "id": label.concept_id,
                            "code": "invalid_candidate_value",
                            "message": "Candidate value does not match its STRING/NUMERIC type.",
                            "value": deepcopy(value),
                        }
                    )
            candidates["covered"] += label.key in candidate_keys
        raw_rows.append((context, gold_raw, predicted))
        normalized_rows.append((context, gold_normalized, predicted))
    expected = coverage["expected_concepts"]
    value_types = tuple(
        kind for kind in VALUE_TYPES if any(concept.value_type == kind for concept in registry.concepts)
    )
    return {
        "available": available,
        "description": "Local exact-match baseline; not the official SYNUR scorer.",
        "value_types": list(value_types),
        "raw": _score(raw_rows, available, value_types),
        "normalized": _score(normalized_rows, available, value_types),
        "coverage": {
            **coverage,
            "row_rate": len(records) / len(rows) if rows else None,
            "audit_rate": coverage["audited_concepts"] / expected if expected else None,
        },
        "counts": {
            **audit_counts,
            "failures": len(failures),
            "failures_by_kind": dict(failure_counts),
            "review_rate": audit_counts["review"] / expected if expected else None,
            "failed_concept_rate": audit_counts["failed"] / expected if expected else None,
            "normalization_changes": len(reference_changes),
            "reference_issues": len(reference_issues),
            "prediction_issues": len(prediction_issues),
        },
        "schema_validity": {
            "submitted": submitted,
            "valid": valid_predictions,
            "invalid": submitted - valid_predictions,
            "rate": valid_predictions / submitted if submitted else None,
        },
        "reference_changes": reference_changes,
        "reference_issues": reference_issues,
        "prediction_issues": prediction_issues,
        "failures": failures,
        "diagnostics": {
            "rows": diagnostic_rows,
            "candidate_coverage": {
                **candidates,
                "uncovered": candidates["evaluated"] - candidates["covered"],
                "rate": candidates["covered"] / candidates["evaluated"]
                if candidates["evaluated"]
                else None,
            },
        },
    }
