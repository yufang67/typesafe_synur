"""Strict prediction contracts and a separate, auditable reference-cleanup view."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

ValueType = Literal["SINGLE_SELECT", "MULTI_SELECT", "STRING", "NUMERIC"]
VALUE_TYPES: tuple[ValueType, ...] = ("SINGLE_SELECT", "MULTI_SELECT", "STRING", "NUMERIC")
ObservationValue = str | int | float | list[str]


class Observation(TypedDict):
    id: str
    name: str
    value_type: ValueType
    value: ObservationValue


@dataclass(frozen=True)
class Concept:
    id: str
    name: str
    value_type: ValueType
    value_enum: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "value_type": self.value_type,
            "value_enum": list(self.value_enum),
        }


@dataclass(frozen=True)
class SchemaRegistry:
    concepts: tuple[Concept, ...]

    @property
    def by_id(self) -> dict[str, Concept]:
        return {concept.id: concept for concept in self.concepts}

    @classmethod
    def from_entries(cls, entries: object) -> SchemaRegistry:
        if not isinstance(entries, list):
            raise ValueError("Schema must be a list of concept objects.")
        concepts: list[Concept] = []
        ids: set[str] = set()
        for index, entry in enumerate(entries):
            prefix = f"Schema entry {index}"
            if not isinstance(entry, dict):
                raise ValueError(f"{prefix} must be an object.")
            for field in ("id", "name"):
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    raise ValueError(f"{prefix}.{field} must be a nonempty string.")
            if entry["id"] in ids:
                raise ValueError(f"{prefix}: duplicate concept ID {entry['id']!r}.")
            value_type = entry.get("value_type")
            if value_type not in VALUE_TYPES:
                raise ValueError(f"{prefix}.value_type must be one of {VALUE_TYPES}.")
            enum = entry.get("value_enum", [])
            if not isinstance(enum, list) or any(
                not isinstance(value, str) or not value for value in enum
            ):
                raise ValueError(f"{prefix}.value_enum must be a list of nonempty strings.")
            if len(enum) != len(set(enum)):
                raise ValueError(f"{prefix}.value_enum contains duplicate values.")
            if value_type in ("SINGLE_SELECT", "MULTI_SELECT") and not enum:
                raise ValueError(f"{prefix}.value_enum must not be empty for a select concept.")
            if value_type in ("STRING", "NUMERIC") and enum:
                raise ValueError(f"{prefix}.value_enum must be empty for {value_type}.")
            concepts.append(
                Concept(entry["id"], entry["name"], cast(ValueType, value_type), tuple(enum))
            )
            ids.add(entry["id"])
        return cls(tuple(concepts))

    def to_entries(self) -> list[dict]:
        return [concept.to_dict() for concept in self.concepts]


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (isinstance(value, int) or math.isfinite(value))
    )


def _observation_errors(raw: object, registry: SchemaRegistry) -> list[tuple[str, str]]:
    if not isinstance(raw, dict):
        return [("", "must be an object with id, name, value_type, and value")]
    errors: list[tuple[str, str]] = []
    expected = {"id", "name", "value_type", "value"}
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing:
        errors.append(("", f"missing required fields: {', '.join(sorted(missing))}"))
    if extra:
        errors.append(("", f"unexpected fields: {', '.join(sorted(map(str, extra)))}"))
    concept_id = raw.get("id")
    concept = registry.by_id.get(concept_id) if isinstance(concept_id, str) else None
    if concept is None:
        errors.append(("id", f"unknown concept ID {concept_id!r}; use an exact schema string ID"))
        return errors
    if raw.get("name") != concept.name:
        errors.append(("name", f"must equal schema name {concept.name!r}"))
    if raw.get("value_type") != concept.value_type:
        errors.append(("value_type", f"must equal schema type {concept.value_type!r}"))
    if "value" not in raw:
        return errors
    value = raw["value"]
    if concept.value_type == "NUMERIC":
        if not _finite_number(value):
            errors.append(("value", "must be a finite int or float, never a bool"))
    elif concept.value_type == "STRING":
        if not isinstance(value, str) or not value.strip():
            errors.append(("value", "must be a nonempty string"))
    elif concept.value_type == "SINGLE_SELECT":
        if not isinstance(value, str) or value not in concept.value_enum:
            errors.append(("value", f"must exactly match one of {concept.value_enum!r}"))
    elif not isinstance(value, list) or not value:
        errors.append(("value", "must be a nonempty list of enum strings"))
    elif any(not isinstance(item, str) or item not in concept.value_enum for item in value):
        errors.append(("value", f"all items must exactly match {concept.value_enum!r}"))
    elif len(value) != len(set(value)):
        errors.append(("value", "must not contain duplicate enum selections"))
    return errors


def validate_observations(raw: object, registry: SchemaRegistry) -> list[Observation]:
    """Validate without coercion; reject even identical repeated prediction IDs.

    The returned observations are deep copies. Sidecars such as confidence and
    source spans belong outside these four-field observations.
    """
    if not isinstance(raw, list):
        raise ValueError("Predicted observations must be a list, not encoded JSON or an object.")
    seen: set[str] = set()
    result: list[Observation] = []
    for index, item in enumerate(raw):
        errors = _observation_errors(item, registry)
        if errors:
            details = "; ".join(f"{field or 'shape'}: {message}" for field, message in errors)
            raise ValueError(f"Observation {index}: {details}.")
        observation = cast(Observation, item)
        if observation["id"] in seen:
            raise ValueError(
                f"Observation {index}: duplicate concept ID {observation['id']!r}; "
                "repeated or conflicting predictions must be resolved before emission."
            )
        seen.add(observation["id"])
        result.append(deepcopy(observation))
    return result


@dataclass
class ReferenceNormalization:
    observations: list[dict]
    changes: list[dict]
    issues: list[dict]
    raw: object


def _reference_items(raw: object) -> tuple[list, str | None]:
    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError) as error:
            return [deepcopy(raw)], f"Reference observations are not valid JSON: {error}"
    if not isinstance(value, list):
        return [deepcopy(value)], "Reference observations must decode to a list."
    return deepcopy(value), None


def _encoding_match(value: str, enum: tuple[str, ...]) -> str | None:
    matches: set[str] = set()
    for encoding in ("latin-1", "cp1252"):
        try:
            repaired = value.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired != value and repaired in enum:
            matches.add(repaired)
    return next(iter(matches)) if len(matches) == 1 else None


def normalize_references(
    raw: object, registry: SchemaRegistry, *, split: str = "", row_id: str = ""
) -> ReferenceNormalization:
    """Copy references, applying only metadata-confirmed, deterministic repairs.

    Numeric STRING conversion uses Python's exact ``str(value)`` representation;
    SINGLE_SELECT conversion additionally requires exact enum membership. Enum
    encoding repair reverses one Latin-1/CP1252-to-UTF-8 mojibake step only when
    exactly one distinct allowed value results. No case/space/synonym cleanup is
    performed. Repeated and conflicting labels are retained and reported.

    Non-object labels are retained as ``{"_raw": original}``. An invalid outer
    container is retained as one malformed label with ``label_count_unknown``;
    its true label count cannot be inferred. ``raw`` preserves the original input.
    """
    items, container_error = _reference_items(raw)
    observations: list[dict] = []
    changes: list[dict] = []
    issues: list[dict] = []
    context = {"split": split, "row_id": row_id}
    if container_error:
        issues.append(
            {
                **context,
                "index": None,
                "field": "",
                "code": "label_count_unknown",
                "message": container_error,
                "value": deepcopy(raw),
            }
        )
    seen: dict[str, dict] = {}
    by_id = registry.by_id
    for index, item in enumerate(items):
        observation = item if isinstance(item, dict) else {"_raw": item}
        location = {**context, "index": index}

        def change(field: str, old: object, new: object, reason: str) -> None:
            changes.append(
                {
                    **location,
                    "id": observation.get("id"),
                    "field": field,
                    "original": deepcopy(old),
                    "normalized": deepcopy(new),
                    "reason": reason,
                }
            )

        concept_id = observation.get("id")
        concept = by_id.get(concept_id) if isinstance(concept_id, str) else None
        if (
            concept is None
            and isinstance(concept_id, str)
            and len(concept_id) > 1
            and concept_id.startswith("0")
            and concept_id.isascii()
            and concept_id.isdigit()
        ):
            matches = [
                entry
                for entry in registry.concepts
                if entry.id.isascii()
                and entry.id.isdigit()
                and entry.id.lstrip("0") == concept_id.lstrip("0")
            ]
            if len(matches) == 1:
                candidate = matches[0]
                if (
                    observation.get("name") == candidate.name
                    and observation.get("value_type") == candidate.value_type
                ):
                    change("id", concept_id, candidate.id, "unambiguous_zero_padded_id")
                    observation["id"] = candidate.id
                    concept = candidate
        if (
            concept is not None
            and observation.get("name") == concept.name
            and observation.get("value_type") == concept.value_type
        ):
            value = observation.get("value")
            if _finite_number(value) and (
                concept.value_type == "STRING"
                or (concept.value_type == "SINGLE_SELECT" and str(value) in concept.value_enum)
            ):
                change("value", value, str(value), "numeric_to_exact_string")
                observation["value"] = str(value)
            value = observation.get("value")
            if "value" in observation and concept.value_type in ("SINGLE_SELECT", "MULTI_SELECT"):
                values = value if isinstance(value, list) else [value]
                repaired_values = deepcopy(values)
                for position, selection in enumerate(values):
                    if isinstance(selection, str) and selection not in concept.value_enum:
                        repaired = _encoding_match(selection, concept.value_enum)
                        if repaired is not None:
                            field = f"value[{position}]" if isinstance(value, list) else "value"
                            change(field, selection, repaired, "unique_enum_encoding_repair")
                            repaired_values[position] = repaired
                observation["value"] = (
                    repaired_values if isinstance(value, list) else repaired_values[0]
                )
        for field, message in _observation_errors(observation, registry):
            issues.append(
                {
                    **location,
                    "id": observation.get("id"),
                    "field": field,
                    "code": "invalid_observation",
                    "message": message,
                    "value": deepcopy(observation.get(field) if field else observation),
                }
            )
        current_id = observation.get("id")
        if isinstance(current_id, str):
            if current_id in seen:
                conflict = observation != seen[current_id]
                issues.append(
                    {
                        **location,
                        "id": current_id,
                        "field": "id",
                        "code": "conflicting_id" if conflict else "duplicate_id",
                        "message": "Repeated reference concept retained, not deduplicated.",
                        "value": deepcopy(observation),
                    }
                )
            else:
                seen[current_id] = observation
        observations.append(observation)
    return ReferenceNormalization(observations, changes, issues, deepcopy(raw))
