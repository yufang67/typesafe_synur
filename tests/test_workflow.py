import json
from dataclasses import replace

import pytest
from typesafe_sdk import Choice, ChoiceAnswer, NoulAnswer

from synur.candidates import (
    Candidate,
    candidate_coverage,
    numeric_candidates,
    parse_number,
    phrase_candidates,
    phrase_starts,
    text_candidates,
)
from synur.experiment import Settings, extract, preview, save_run
from synur.jev import JevAdapter, ModelCallError, ModelReply
from synur.observations import SchemaRegistry
from synur.questions import (
    RULES,
    build_state,
    candidate_question,
    enum_questions,
    pack_questions,
    status_questions,
)

TEXT = "Patient denies nausea. Urine is dark and cloudy. Urine output 150 mL. Heart sounds regular."


@pytest.fixture
def registry():
    return SchemaRegistry.from_entries(
        [
            {
                "id": "30",
                "name": "Nausea",
                "value_type": "SINGLE_SELECT",
                "value_enum": ["Yes", "No"],
            },
            {
                "id": "139",
                "name": "Urine appearance",
                "value_type": "MULTI_SELECT",
                "value_enum": ["dark", "cloudy", "clear"],
            },
            {"id": "73", "name": "Urine output", "value_type": "NUMERIC"},
            {"id": "1", "name": "Heart sounds", "value_type": "STRING"},
            {"id": "19", "name": "Temperature", "value_type": "NUMERIC"},
        ]
    )


def choice_answer(question, selected, confidence=1.0):
    return ChoiceAnswer(
        choice=selected,
        confidence=confidence,
        probabilities={key: float(key == selected) for key in question.criteria},
    )


class FixtureAdapter:
    """Controlled responses only, not an approximation of model accuracy."""

    def __init__(self):
        self.calls = []

    def ask(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for qid, question in questions.items():
            if qid.endswith("_status"):
                selected = (
                    "supported"
                    if qid in {"obs_30_status", "obs_139_status", "obs_73_status", "obs_1_status"}
                    else "not_stated"
                )
            elif qid == "obs_30_value":
                selected = next(
                    key
                    for key, value in question.criteria.items()
                    if isinstance(value, dict) and value.get("schema_value") == "No"
                )
            elif qid.startswith("obs_139_member"):
                answers[qid] = NoulAnswer(
                    noul=0.9 if question.instructions["schema_value"] in {"dark", "cloudy"} else 0.1
                )
                continue
            else:
                target = "150" if qid.startswith("obs_73") else "regular"
                groups = {
                    key: value["source_candidates"]
                    for key, value in question.criteria.items()
                    if isinstance(value, dict)
                }
                exact = [
                    key
                    for key, group in groups.items()
                    if any(item["text"] == target for item in group)
                ]
                enclosing = [
                    key
                    for key, group in groups.items()
                    if any(target in item["text"] for item in group)
                ]
                selected = (exact or enclosing or ["ambiguous"])[0]
            answers[qid] = choice_answer(question, selected)
        return ModelReply(answers=answers, model="fixture-not-jev", usage={"input_tokens": 0})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("150", 150),
        ("-2.5", -2.5),
        ("1,250.5", 1250.5),
        ("one hundred and fifty", 150),
        ("ninety two", 92),
        ("minus zero point five", -0.5),
        ("two thousand and five", 2005),
    ],
)
def test_number_parser(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["120/80", "3 to 5", "one and two", "12,34", "1e3", "NaN"])
def test_number_parser_rejects_ambiguous(text):
    with pytest.raises(ValueError):
        parse_number(text)


def test_numeric_candidates_offsets_duplicates_and_composites():
    text = "BP 120/80; age 72; pulse 72; range 3-5; three out of five; 1e3; 12,34; 1,250."
    candidates, issues = numeric_candidates(text)
    assert [item.value for item in candidates] == [72, 72, 1250]
    assert candidates[0].start != candidates[1].start
    assert len(issues) == 5
    assert all(text[item.start : item.end] == item.text for item in candidates)


def test_phrase_spans_preserve_source():
    text = "[Clinician] Heart sounds regular, pulse 72."
    clauses = text_candidates(text)
    assert clauses[0].text == "Heart sounds regular"
    phrases = phrase_candidates(text, clauses[0])
    assert [item.text for item in phrase_starts(text, clauses[0])] == ["Heart", "sounds", "regular"]
    assert any(item.value == "regular" for item in phrases)
    assert all(text[item.start : item.end] == item.text for item in phrases)
    numbers, _ = numeric_candidates(text)
    report = candidate_coverage(
        text,
        [
            {"value_type": "STRING", "value": "regular"},
            {"value_type": "NUMERIC", "value": 72},
        ],
        numbers,
        clauses,
    )
    assert report["STRING"]["coverage"] == report["NUMERIC"]["coverage"] == 1


def test_native_request_and_no_gold_leak(registry):
    state = build_state(TEXT, registry)
    assert set(state) == {"context", "schema"}
    specs = status_questions(registry)
    assert len(specs) == len(registry.concepts)
    assert len({spec.id for spec in specs}) == len(specs)
    for spec in specs:
        assert spec.question.instructions["rules"] == RULES
        assert spec.question.instructions["concept"]["id"] == spec.concept_id
    assert len(enum_questions(registry.by_id["139"])) == 3
    result = preview(TEXT, registry)
    assert result["concept_count"] == 5
    assert result["live_calls_made"] is False
    assert "reference" not in json.dumps(state)


@pytest.mark.parametrize("count", [1, 252, 253, 1000])
def test_choice_limit_and_hierarchy_never_drop_candidates(registry, count):
    candidates = [Candidate(i, i + 1, str(i), i, "number") for i in range(count)]
    spec = candidate_question(registry.by_id["73"], candidates, stage="number", round_index=0)
    assert len(spec.question.criteria) <= 255
    assert [item for group in spec.branches.values() for item in group] == candidates
    if count == 252:
        assert len(spec.question.criteria) == 255
    if count > 252:
        assert all(len(group) < count for group in spec.branches.values())


def test_question_batches_and_budget_errors(registry):
    specs = status_questions(registry)
    batches = pack_questions(build_state(TEXT, registry), specs, max_questions=2)
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [spec.id for batch in batches for spec in batch] == [spec.id for spec in specs]
    with pytest.raises(ValueError, match="No source content"):
        pack_questions(build_state(TEXT, registry), specs, max_state_question_bytes=10)
    with pytest.raises(ValueError, match="unique"):
        pack_questions({}, [specs[0], specs[0]])


def test_full_extraction_all_types_and_explicit_negative(registry):
    adapter = FixtureAdapter()
    result = extract(TEXT, registry, adapter, row_id="fixture")
    assert result["status"] == "complete"
    values = {item["id"]: item["value"] for item in result["observations"]}
    assert values == {"30": "No", "139": ["dark", "cloudy"], "73": 150, "1": "regular"}
    assert len(result["audit"]) == 5
    assert result["audit"][-1]["status"] == "absent"
    evidence = next(item["evidence"] for item in result["audit"] if item["id"] == "1")
    assert evidence[0]["text"] == "regular"
    assert all(set(state) == {"context", "schema"} for state, _ in adapter.calls)
    assert result["metadata"]["actual_models"] == ["fixture-not-jev"]
    from synur.evaluation import evaluate

    report = evaluate(
        [{"id": "fixture", "transcript": TEXT, "observations": result["observations"]}],
        [result],
        registry,
    )
    assert report["diagnostics"]["candidate_coverage"]["rate"] == 1
    assert report["diagnostics"]["candidate_coverage"]["unavailable"] == 0


@pytest.mark.parametrize("include_numeric", [False, True])
def test_selected_scope_only_predicts_enabled_types(registry, monkeypatch, include_numeric):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled observation types must not generate source candidates")

    if not include_numeric:
        monkeypatch.setattr("synur.experiment.numeric_candidates", forbidden)
    monkeypatch.setattr("synur.experiment.text_candidates", forbidden)
    enabled_types = {"SINGLE_SELECT", "MULTI_SELECT"}
    if include_numeric:
        enabled_types.add("NUMERIC")
    selected = SchemaRegistry(
        tuple(concept for concept in registry.concepts if concept.value_type in enabled_types)
    )
    request = preview(TEXT, selected)
    assert request["concept_count"] == (4 if include_numeric else 2)
    assert set(request["value_question_examples"]) == enabled_types
    assert bool(request["numeric_candidates"]) == include_numeric
    assert request["text_candidates"] == []
    assert request["candidate_issues"] == []
    adapter = FixtureAdapter()
    result = extract(TEXT, selected, adapter)
    assert result["status"] == "complete"
    expected = {"30": "No", "139": ["dark", "cloudy"]}
    if include_numeric:
        expected["73"] = 150
    assert {item["id"]: item["value"] for item in result["observations"]} == expected
    assert {item["id"] for item in result["audit"]} == (
        {"30", "139", "73", "19"} if include_numeric else {"30", "139"}
    )
    assert result["candidate_issues"] == []
    assert result["diagnostics"]["candidate_pools"] == (
        {"NUMERIC": [150]} if include_numeric else {}
    )
    for state, questions in adapter.calls:
        assert {entry["value_type"] for entry in state["schema"]} == enabled_types
        assert all(
            question.instructions["concept"]["value_type"] in enabled_types
            for question in questions.values()
        )
    from synur.reporting import build_transcript_report

    report = build_transcript_report(
        [{"id": result["id"], "transcript": TEXT, "observations": result["observations"]}],
        [result], selected,
    )
    for item in report["transcripts"][0]["predicted_observations"]:
        concept_id = item["observation"]["id"]
        provenance = item["provenance"]
        assert provenance["audit_status"] == "recorded"
        assert provenance["audit"] == [
            entry for entry in result["audit"] if entry["id"] == concept_id
        ]
        assert provenance["requests"]
        assert provenance["metadata"]["actual_models"] == ["fixture-not-jev"]
        evidence = provenance["audit"][0]["evidence"]
        if concept_id == "73":
            assert len(evidence) == 1
            assert TEXT[evidence[0]["start"]:evidence[0]["end"]] == "150"
        else:
            assert evidence == []


@pytest.mark.parametrize(("confidence", "expected"), [(0.8, "emitted"), (0.7999, "review")])
def test_choice_threshold_boundary(registry, confidence, expected):
    class ThresholdAdapter(FixtureAdapter):
        def ask(self, state, questions):
            reply = super().ask(state, questions)
            if "obs_30_value" in reply.answers:
                reply.answers["obs_30_value"] = reply.answers["obs_30_value"].model_copy(
                    update={"confidence": confidence}
                )
            return reply

    result = extract(TEXT, registry, ThresholdAdapter())
    assert next(item for item in result["audit"] if item["id"] == "30")["status"] == expected


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.8, "emitted"),
        (0.7999, "review"),
        (0.2001, "review"),
        (0.2, "emitted"),
    ],
)
def test_noul_threshold_boundary(registry, probability, expected):
    class ThresholdAdapter(FixtureAdapter):
        def ask(self, state, questions):
            reply = super().ask(state, questions)
            if "obs_139_member_0" in reply.answers:
                reply.answers["obs_139_member_0"] = NoulAnswer(noul=probability)
            return reply

    result = extract(TEXT, registry, ThresholdAdapter())
    assert next(item for item in result["audit"] if item["id"] == "139")["status"] == expected


@pytest.mark.parametrize("corruption", ["missing", "extra", "wrong_type", "nan", "bad_option"])
def test_corrupt_response_is_failure_not_absence(registry, corruption):
    class CorruptAdapter(FixtureAdapter):
        def ask(self, state, questions):
            reply = super().ask(state, questions)
            key = next(iter(reply.answers))
            if corruption == "missing":
                del reply.answers[key]
            elif corruption == "extra":
                reply.answers["unknown_id"] = reply.answers[key]
            elif corruption == "wrong_type":
                reply.answers[key] = NoulAnswer(noul=1.0)
            elif isinstance(reply.answers[key], ChoiceAnswer):
                patch = (
                    {"confidence": float("nan")} if corruption == "nan" else {"choice": "invalid"}
                )
                reply.answers[key] = reply.answers[key].model_copy(update=patch)
            return reply

    result = extract(TEXT, registry, CorruptAdapter())
    assert result["status"] in {"partial", "failed"}
    assert result["failures"]
    assert result["audit"][0]["status"] == "failed"


def test_model_failure_is_explicit(registry):
    class FailedAdapter:
        def ask(self, state, questions):
            raise ModelCallError("Offline fixture failure")

    result = extract(TEXT, registry, FailedAdapter())
    assert result["status"] == "failed"
    assert len(result["failures"]) == 5
    assert all(item["status"] == "failed" for item in result["audit"])
    assert result["observations"] == []


def test_missing_candidates_are_review_not_absence(registry):
    result = extract("No numeric measurements available.", registry, FixtureAdapter())
    item = next(item for item in result["audit"] if item["id"] == "73")
    assert item["status"] == "review"
    assert item["reason"] == "no_source_candidates"


def test_sdk_live_gate_and_placeholders(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ModelCallError, match="disabled"):
        JevAdapter()
    with pytest.raises(ModelCallError, match="Set TYPESAFE_API_KEY"):
        JevAdapter(enabled=True)
    with pytest.raises(ModelCallError, match="placeholder"):
        JevAdapter(enabled=True, api_key="REPLACE_WITH_YOUR_KEY")
    with pytest.raises(ModelCallError, match="HTTPS"):
        JevAdapter(enabled=True, api_key="fixture-key", base_url="http://unapproved.invalid")


def test_sdk_wire_contract_without_network(monkeypatch, registry):
    import httpx2
    from typesafe_sdk import TypeSafeClient

    import synur.jev

    captured = []

    def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert set(payload) == {"state", "questions", "model"}
        assert request.url.path.endswith("/v1/systemone")
        answers = {
            key: choice_answer(Choice.model_validate(value), "not_stated").model_dump()
            for key, value in payload["questions"].items()
        }
        return httpx2.Response(
            200,
            json={
                "model": "fixture-model",
                "usage": {"input_tokens": 3, "output_tokens": 0},
                "answers": answers,
            },
            headers={"x-typesafe-request-id": "fixture-request"},
        )

    def client_factory(**kwargs):
        return TypeSafeClient(**kwargs, transport=httpx2.MockTransport(handler))

    monkeypatch.setattr(synur.jev, "TypeSafeClient", client_factory)
    with JevAdapter(enabled=True, api_key="fixture-key") as adapter:
        result = extract(TEXT, registry, adapter)
    assert result["status"] == "complete"
    assert result["observations"] == []
    assert result["requests"][0]["request_id"] == "fixture-request"
    assert captured[0]["state"] == build_state(TEXT, registry)


def test_run_exports_no_overwrite(tmp_path, registry):
    result = extract(TEXT, registry, FixtureAdapter())
    target = tmp_path / "run"
    save_run(target, [result], {"available": False}, {"dataset_revision": "fixture"})
    assert json.loads((target / "predictions.jsonl").read_text())["observations"]
    assert json.loads((target / "run.json").read_text())["prediction_records"] == 1
    with pytest.raises(FileExistsError):
        save_run(target, [], {}, {})


def test_configuration_validation():
    with pytest.raises(ValueError):
        Settings(choice_confidence=float("nan"))
    with pytest.raises(ValueError):
        replace(Settings(), batch_size=0)
    with pytest.raises(ValueError):
        replace(Settings(), batch_size=1.5)
    with pytest.raises(ValueError):
        replace(Settings(), support_probability=True)


def test_full_local_registry_with_controlled_responses():
    from pathlib import Path

    from synur.dataset import load_dataset

    data_dir = Path(__file__).resolve().parents[1] / "data" / "synur"
    if not (data_dir / "manifest.json").is_file():
        pytest.skip("Separate dataset download required for full-schema integration.")
    dataset = load_dataset(data_dir)
    registry = SchemaRegistry.from_entries(dataset.schema_entries)
    result = extract(TEXT, registry, FixtureAdapter())
    assert result["status"] == "complete"
    assert len(result["audit"]) == len(registry.concepts) == 193
    assert len({item["id"] for item in result["audit"]}) == 193
    assert {item["id"] for item in result["observations"]} == {"30", "139", "73", "1"}
    assert not result["failures"]
