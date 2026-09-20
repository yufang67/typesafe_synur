"""Direct JEV decisions -> auditable, schema-validated SYNUR observations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from typesafe_sdk import Choice, ChoiceAnswer, NoulAnswer

from synur.candidates import (
    Candidate,
    candidate_value_pools,
    numeric_candidates,
    phrase_candidates,
    phrase_starts,
    text_candidates,
)
from synur.jev import ModelAdapter, ModelCallError
from synur.observations import Concept, Observation, SchemaRegistry, validate_observations
from synur.questions import (
    QuestionSpec,
    build_state,
    candidate_question,
    enum_questions,
    pack_questions,
    status_questions,
)

PROMPT_VERSION = "direct-jev-v1"


@dataclass(frozen=True)
class Settings:
    choice_confidence: float = 0.8
    support_probability: float = 0.8
    reject_probability: float = 0.2
    batch_size: int = 32
    max_request_bytes: int = 60_000
    max_state_question_bytes: int = 30_000

    def __post_init__(self) -> None:
        probabilities = (
            self.choice_confidence,
            self.support_probability,
            self.reject_probability,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in probabilities
        ):
            raise ValueError("Thresholds must be finite numeric probabilities, not booleans.")
        if not (
            0 <= self.choice_confidence <= 1
            and 0 <= self.reject_probability < self.support_probability <= 1
        ):
            raise ValueError("Thresholds must be finite probabilities with reject < support.")
        budgets = (self.batch_size, self.max_request_bytes, self.max_state_question_bytes)
        if any(type(value) is not int or value < 1 for value in budgets):
            raise ValueError("Question and byte budgets must be positive integers.")


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _validate_answer(spec: QuestionSpec, answer: object) -> ChoiceAnswer | NoulAnswer:
    if isinstance(spec.question, Choice):
        if not isinstance(answer, ChoiceAnswer):
            raise ValueError("Expected a Choice answer")
        if answer.choice not in spec.question.criteria:
            raise ValueError("Choice selected an unknown option")
        if set(answer.probabilities) != set(spec.question.criteria):
            raise ValueError("Choice probability keys do not match options")
        values = list(answer.probabilities.values())
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError("Invalid Choice probabilities")
        if not math.isclose(sum(values), 1, abs_tol=0.02):
            raise ValueError("Choice probabilities do not sum approximately to one")
        if answer.probabilities[answer.choice] < max(values):
            raise ValueError("Selected Choice is not a highest-probability option")
        if not math.isfinite(answer.confidence) or not 0 <= answer.confidence <= 1:
            raise ValueError("Invalid Choice confidence")
        return answer
    if not isinstance(answer, NoulAnswer):
        raise ValueError("Expected a Noul answer")
    if not math.isfinite(answer.noul) or not 0 <= answer.noul <= 1:
        raise ValueError("Invalid Noul probability")
    return answer


def _source_candidates(
    transcript: str, registry: SchemaRegistry
) -> tuple[list[Candidate], list[Candidate], list[dict]]:
    kinds = {concept.value_type for concept in registry.concepts}
    numbers, issues = numeric_candidates(transcript) if "NUMERIC" in kinds else ([], [])
    clauses = text_candidates(transcript) if "STRING" in kinds else []
    return numbers, clauses, issues


def preview(transcript: str, registry: SchemaRegistry, settings: Settings | None = None) -> dict:
    """Compile the entire presence stage and representative follow-ups without a client."""
    config = settings or Settings()
    state = build_state(transcript, registry)
    specs = status_questions(registry)
    batches = pack_questions(
        state,
        specs,
        max_questions=config.batch_size,
        max_request_bytes=config.max_request_bytes,
        max_state_question_bytes=config.max_state_question_bytes,
    )
    numeric, clauses, issues = _source_candidates(transcript, registry)
    examples = {}
    for kind in ("SINGLE_SELECT", "MULTI_SELECT", "NUMERIC", "STRING"):
        concept = next((item for item in registry.concepts if item.value_type == kind), None)
        if concept is None:
            continue
        if kind in {"SINGLE_SELECT", "MULTI_SELECT"}:
            examples[kind] = [spec.preview() for spec in enum_questions(concept)]
        else:
            candidates = numeric if kind == "NUMERIC" else clauses
            examples[kind] = (
                candidate_question(
                    concept,
                    candidates,
                    stage="number" if kind == "NUMERIC" else "clause",
                    round_index=0,
                ).preview()
                if candidates
                else {"reason": "no_source_candidates"}
            )
    return {
        "state": state,
        "questions": [spec.preview() for spec in specs],
        "concept_count": len(specs),
        "status_batches": len(batches),
        "value_question_examples": examples,
        "numeric_candidates": [item.to_dict() for item in numeric],
        "text_candidates": [item.to_dict() for item in clauses],
        "candidate_issues": issues,
        "settings": asdict(config),
        "live_calls_made": False,
    }


def extract(
    transcript: str,
    registry: SchemaRegistry,
    adapter: ModelAdapter,
    *,
    row_id: str = "custom",
    settings: Settings | None = None,
) -> dict:
    config = settings or Settings()
    state = build_state(transcript, registry)
    audit = {
        concept.id: {
            "id": concept.id,
            "name": concept.name,
            "status": "pending",
            "reason": None,
            "decisions": [],
            "evidence": [],
        }
        for concept in registry.concepts
    }
    failures: list[dict] = []
    requests: list[dict] = []
    observations: list[Observation] = []
    numbers, clauses, candidate_issues = _source_candidates(transcript, registry)

    def fail(spec: QuestionSpec, message: str) -> None:
        audit[spec.concept_id].update(status="failed", reason=message)
        failures.append({"concept_id": spec.concept_id, "question_id": spec.id, "error": message})

    def ask(specs: list[QuestionSpec]) -> dict[str, ChoiceAnswer | NoulAnswer]:
        answers = {}
        try:
            batches = pack_questions(
                state,
                specs,
                max_questions=config.batch_size,
                max_request_bytes=config.max_request_bytes,
                max_state_question_bytes=config.max_state_question_bytes,
            )
        except ValueError as exc:
            for spec in specs:
                fail(spec, str(exc))
            return answers
        for batch in batches:
            question_map = {spec.id: spec.question for spec in batch}
            request = {
                "question_ids": list(question_map),
                "questions_hash": _hash([spec.preview() for spec in batch]),
            }
            try:
                reply = adapter.ask(state, question_map)
            except ModelCallError as exc:
                requests.append({**request, "status": "failed", "error": str(exc)})
                for spec in batch:
                    fail(spec, str(exc))
                continue
            requests.append(
                {
                    **request,
                    "status": "returned",
                    "model": reply.model,
                    "usage": reply.usage,
                    "request_id": reply.request_id,
                }
            )
            unknown = set(reply.answers) - set(question_map)
            for spec in batch:
                try:
                    if unknown:
                        raise ValueError("Response included unexpected question IDs")
                    if spec.id not in reply.answers:
                        raise ValueError("Response omitted this question ID")
                    answer = _validate_answer(spec, reply.answers[spec.id])
                except ValueError as exc:
                    fail(spec, str(exc))
                    continue
                answers[spec.id] = answer
                audit[spec.concept_id]["decisions"].append(
                    {"question_id": spec.id, **answer.model_dump(mode="json")}
                )
        return answers

    def accepted_choice(spec: QuestionSpec, answer: ChoiceAnswer | NoulAnswer | None) -> str | None:
        if audit[spec.concept_id]["status"] == "failed":
            return None
        if not isinstance(answer, ChoiceAnswer):
            fail(spec, "Expected a validated Choice response")
            return None
        if answer.confidence < config.choice_confidence:
            audit[spec.concept_id].update(status="review", reason="low_choice_confidence")
            return None
        return answer.choice

    def emit(concept: Concept, value: str | int | float | list[str], evidence: list[dict]) -> None:
        result = {
            "id": concept.id,
            "name": concept.name,
            "value_type": concept.value_type,
            "value": value,
        }
        try:
            observation = validate_observations([result], registry)[0]
        except ValueError as exc:
            audit[concept.id].update(status="failed", reason=str(exc))
            failures.append({"concept_id": concept.id, "error": str(exc)})
            return
        observations.append(observation)
        audit[concept.id].update(status="emitted", reason=None, evidence=evidence)

    statuses = status_questions(registry)
    initial = ask(statuses)
    supported = []
    for spec in statuses:
        choice = accepted_choice(spec, initial.get(spec.id))
        if choice == "supported":
            supported.append(registry.by_id[spec.concept_id])
        elif choice == "not_stated":
            audit[spec.concept_id].update(status="absent", reason="not_stated")
        elif choice is not None:
            audit[spec.concept_id].update(status="review", reason=choice)

    enums = [
        concept for concept in supported if concept.value_type in {"SINGLE_SELECT", "MULTI_SELECT"}
    ]
    enum_specs = [spec for concept in enums for spec in enum_questions(concept)]
    enum_answers = ask(enum_specs)
    for concept in enums:
        specs = [spec for spec in enum_specs if spec.concept_id == concept.id]
        if audit[concept.id]["status"] == "failed":
            continue
        if concept.value_type == "SINGLE_SELECT":
            spec = specs[0]
            choice = accepted_choice(spec, enum_answers.get(spec.id))
            if choice in spec.values:
                emit(concept, spec.values[choice], [])
            elif choice is not None:
                audit[concept.id].update(status="review", reason=f"value_{choice}")
        else:
            selected = []
            uncertain = False
            for spec in specs:
                answer = enum_answers.get(spec.id)
                if not isinstance(answer, NoulAnswer):
                    fail(spec, "Missing validated Noul response")
                    break
                if answer.noul >= config.support_probability:
                    selected.append(spec.values["true"])
                elif answer.noul > config.reject_probability:
                    uncertain = True
            if audit[concept.id]["status"] != "failed":
                if uncertain or not selected:
                    audit[concept.id].update(
                        status="review",
                        reason="uncertain_members" if uncertain else "no_supported_members",
                    )
                else:
                    emit(concept, selected, [])

    # Each round batches independent candidate selections; the next round narrows chosen branches.
    jobs: list[tuple[Concept, str, list[Candidate]]] = []
    selected_clauses: dict[str, Candidate] = {}
    for concept in supported:
        if concept.value_type not in {"STRING", "NUMERIC"}:
            continue
        candidates = numbers if concept.value_type == "NUMERIC" else clauses
        if not candidates:
            audit[concept.id].update(status="review", reason="no_source_candidates")
        else:
            jobs.append(
                (concept, "number" if concept.value_type == "NUMERIC" else "clause", candidates)
            )
    round_index = 0
    while jobs:
        specs = [
            candidate_question(concept, pool, stage=stage, round_index=round_index)
            for concept, stage, pool in jobs
        ]
        answers = ask(specs)
        next_jobs = []
        for (concept, stage, pool), spec in zip(jobs, specs, strict=True):
            choice = accepted_choice(spec, answers.get(spec.id))
            if choice is None:
                continue
            if choice not in spec.branches:
                audit[concept.id].update(status="review", reason=f"candidate_{choice}")
                continue
            branch = list(spec.branches[choice])
            if not branch:
                raise RuntimeError("Candidate hierarchy contained an empty branch")
            if len(branch) > 1:
                if len(branch) >= len(pool):
                    raise RuntimeError("Candidate hierarchy did not narrow")
                next_jobs.append((concept, stage, list(branch)))
                continue
            candidate = branch[0]
            if stage == "clause":
                selected_clauses[concept.id] = candidate
                phrases = phrase_starts(transcript, candidate)
                if phrases:
                    next_jobs.append((concept, "phrase_start", phrases))
                else:
                    audit[concept.id].update(status="review", reason="no_phrase_candidates")
            elif stage == "phrase_start":
                phrases = [
                    span
                    for span in phrase_candidates(transcript, selected_clauses[concept.id])
                    if span.start == candidate.start
                ]
                if not phrases:
                    raise RuntimeError("Selected start token has no matching phrase")
                next_jobs.append((concept, "phrase", phrases))
            else:
                emit(concept, candidate.value, [candidate.to_dict()])
        jobs = next_jobs
        round_index += 1

    if any(item["status"] == "pending" for item in audit.values()):
        raise RuntimeError("Extraction left unaccounted schema concepts")
    return {
        "id": row_id,
        "status": ("partial" if observations else "failed") if failures else "complete",
        "observations": observations,
        "audit": list(audit.values()),
        "failures": failures,
        "requests": requests,
        "candidate_issues": candidate_issues,
        "diagnostics": {
            "candidate_pools": {
                kind: values
                for kind, values in candidate_value_pools(transcript, numbers, clauses).items()
                if any(concept.value_type == kind for concept in registry.concepts)
            }
        },
        "metadata": {
            "prompt_version": PROMPT_VERSION,
            "state_hash": _hash(state),
            "schema_hash": _hash(registry.to_entries()),
            "settings": asdict(config),
            "concept_count": len(registry.concepts),
            "candidate_policy": "transcript-only scalars and hierarchical clause/contiguous-span selection",
            "actual_models": sorted({item["model"] for item in requests if "model" in item}),
        },
    }


def save_run(output_dir: Path, predictions: list[dict], metrics: dict, metadata: dict) -> None:
    """Write a new run directory only; never overwrite a previous experiment."""
    output_dir = Path(output_dir)
    payloads = {
        "predictions.jsonl": "".join(
            json.dumps(
                {key: row[key] for key in ("id", "status", "observations")},
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in predictions
        ),
        "audit.jsonl": "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in predictions
        ),
        "failures.jsonl": "".join(
            json.dumps({"id": row["id"], **failure}, ensure_ascii=False, allow_nan=False) + "\n"
            for row in predictions
            for failure in row["failures"]
        ),
        "metrics.json": json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False),
        "run.json": json.dumps(
            {
                **metadata,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "typesafe_sdk": importlib.metadata.version("typesafe-sdk"),
                "prediction_records": len(predictions),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, content in payloads.items():
        path = output_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
