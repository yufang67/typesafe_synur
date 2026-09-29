import json
import traceback

import httpx2
import pytest
from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient

from synur.experiment import extract
from synur.jev import (
    PI_SCORER_BASE_URL,
    JevAdapter,
    ModelCallError,
    configure_api_key,
    resolve_model_config,
)
from synur.observations import SchemaRegistry
from synur.questions import build_state

SYNTHETIC_KEY = "synthetic-contract-key"
SYNTHETIC_PRIVATE_BODY = "synthetic-private-response-content"


@pytest.fixture(autouse=True)
def clean_provider_environment(monkeypatch):
    for variable in (
        "SYNUR_MODEL_PROVIDER", "TYPESAFE_MODEL", "TYPESAFE_DEFAULT_MODEL",
        "TYPESAFE_BASE_URL", "TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def mock_sdk(monkeypatch):
    clients = []

    def install(handler):
        def factory(**kwargs):
            client = TypeSafeClient(
                **kwargs, transport=httpx2.MockTransport(handler), retry=RetryPolicy(max_retries=0)
            )
            clients.append(client)
            return client

        monkeypatch.setattr("synur.jev.TypeSafeClient", factory)
        return clients

    return install


def test_default_profiles_and_environment_selection(monkeypatch):
    external = resolve_model_config()
    assert (external.provider, external.model, external.base_url) == ("typesafe", "jev-1.13.0", None)
    monkeypatch.setenv("SYNUR_MODEL_PROVIDER", "pi-scorer")
    pi = resolve_model_config()
    assert (pi.provider, pi.model, pi.base_url) == ("pi-scorer", "pi-scorer", PI_SCORER_BASE_URL)
    assert resolve_model_config(provider="typesafe") == external


@pytest.mark.parametrize("provider", ["typesafe", "pi-scorer"])
def test_model_precedence(provider, monkeypatch):
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "documented-default")
    assert resolve_model_config(provider=provider).model == "documented-default"
    monkeypatch.setenv("TYPESAFE_MODEL", "legacy-project-override")
    assert resolve_model_config(provider=provider).model == "legacy-project-override"
    assert resolve_model_config(provider=provider, model="explicit").model == "explicit"
    for invalid in ("", " "):
        with pytest.raises(ModelCallError, match="nonempty"):
            resolve_model_config(provider=provider, model=invalid)
        monkeypatch.setenv("TYPESAFE_MODEL", invalid)
        with pytest.raises(ModelCallError, match="nonempty"):
            resolve_model_config(provider=provider)


@pytest.mark.parametrize("path", ["", "/", "/invocations", "/invocations/", "/v1/systemone"])
def test_pi_endpoint_normalization_and_url_precedence(path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", PI_SCORER_BASE_URL + path)
    assert resolve_model_config(provider="pi-scorer").base_url == PI_SCORER_BASE_URL
    assert resolve_model_config(
        provider="pi-scorer", base_url="https://authorized.invalid/invocations"
    ).base_url == "https://authorized.invalid"
    assert resolve_model_config(provider="typesafe").base_url == PI_SCORER_BASE_URL + path


@pytest.mark.parametrize("url", [
    "http://example.invalid", "https://", "https://user:secret@example.invalid",
    "https://example.invalid?key=synthetic", "https://example.invalid/#fragment",
    "https://example.invalid:invalid", "https://[invalid", "https://bad host.invalid",
    "https://example.invalid/other-route",
])
def test_pi_rejects_invalid_urls_without_echoing_them(url):
    with pytest.raises(ModelCallError) as caught:
        resolve_model_config(provider="pi-scorer", base_url=url)
    assert url not in str(caught.value)


def test_unknown_provider_never_prompts_or_constructs_client(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid provider reached live setup")

    monkeypatch.setattr("synur.jev.TypeSafeClient", forbidden)
    monkeypatch.setattr("getpass.getpass", forbidden)
    monkeypatch.setenv("SYNUR_MODEL_PROVIDER", "invalid-provider")
    for call in (
        resolve_model_config,
        lambda: configure_api_key(enabled=True),
        lambda: JevAdapter(enabled=True, api_key=SYNTHETIC_KEY),
    ):
        with pytest.raises(ModelCallError, match="SYNUR_MODEL_PROVIDER"):
            call()


@pytest.mark.parametrize("provider", ["typesafe", "pi-scorer"])
def test_no_client_without_gate_and_valid_key(provider, monkeypatch):
    monkeypatch.setattr(
        "synur.jev.TypeSafeClient", lambda **kwargs: pytest.fail("Unexpected SDK construction")
    )
    with pytest.raises(ModelCallError, match="disabled"):
        JevAdapter(provider=provider, api_key=SYNTHETIC_KEY)
    with pytest.raises(ModelCallError, match="Set TYPESAFE_API_KEY"):
        JevAdapter(provider=provider, enabled=True)
    for key in ("", " ", "REPLACE_WITH_YOUR_KEY"):
        with pytest.raises(ModelCallError):
            JevAdapter(provider=provider, enabled=True, api_key=key)


def choice_reply(question, selected, confidence=0.87):
    count = len(question["criteria"])
    return {
        "type": "choice", "choice": selected, "confidence": confidence,
        "probabilities": {
            key: 0.91 if key == selected else 0.09 / (count - 1)
            for key in question["criteria"]
        },
    }


def test_pi_sdk_wire_and_all_enabled_types_end_to_end(mock_sdk, tmp_path):
    registry = SchemaRegistry.from_entries([
        {"id": "1", "name": "Synthetic category", "value_type": "SINGLE_SELECT",
         "value_enum": ["A", "B"]},
        {"id": "2", "name": "Synthetic members", "value_type": "MULTI_SELECT",
         "value_enum": ["X", "Y"]},
        {"id": "3", "name": "Synthetic number", "value_type": "NUMERIC"},
    ])
    text = "Synthetic example: category B, member X, number 42."
    captured = []

    def handler(request):
        assert str(request.url) == PI_SCORER_BASE_URL + "/v1/systemone"
        assert request.method == "POST"
        # The official SDK owns authentication; no custom Azure transport is substituted.
        assert request.headers["authorization"] == f"Bearer {SYNTHETIC_KEY}"
        payload = json.loads(request.content)
        assert set(payload) == {"state", "questions", "model"}
        assert payload["model"] == "pi-scorer"
        assert payload["state"] == build_state(text, registry)
        captured.append(payload)
        answers = {}
        for qid, question in payload["questions"].items():
            if question["type"] == "noul":
                assert question["criteria"] == {
                    "true": "This exact schema value is supported.",
                    "false": "This value is not supported, or is explicitly denied.",
                }
                assert question["instructions"]["concept"]["id"] == "2"
                answers[qid] = {
                    "type": "noul",
                    "noul": 0.93 if question["instructions"]["schema_value"] == "X" else 0.07,
                }
            else:
                selected = "supported" if qid.endswith("_status") else (
                    "value_1" if qid == "obs_1_value" else "candidate_0"
                )
                answers[qid] = choice_reply(question, selected)
        return httpx2.Response(200, json={
            "model": "pi-server-reported-model", "answers": answers,
            "usage": {"input_tokens": 123, "output_tokens": 7},
        }, headers={"x-typesafe-request-id": "synthetic-pi-request"})

    clients = mock_sdk(handler)
    with JevAdapter(
        provider="pi-scorer", enabled=True, api_key=SYNTHETIC_KEY,
        base_url=PI_SCORER_BASE_URL + "/invocations",
    ) as adapter:
        result = extract(text, registry, adapter, row_id="synthetic")
    assert clients[0]._http_client.is_closed
    assert len(captured) == 3
    assert result["status"] == "complete"
    assert [item["value"] for item in result["observations"]] == ["B", ["X"], 42]
    assert result["metadata"]["actual_providers"] == ["pi-scorer"]
    assert result["metadata"]["actual_models"] == ["pi-server-reported-model"]
    assert all(request["usage"] == {"input_tokens": 123, "output_tokens": 7}
               for request in result["requests"])
    assert all(request["request_id"] == "synthetic-pi-request" for request in result["requests"])
    assert result["audit"][0]["decisions"][0]["confidence"] == 0.87
    assert result["audit"][0]["decisions"][0]["probabilities"]["supported"] == 0.91
    assert SYNTHETIC_KEY not in json.dumps(result)

    from synur.experiment import save_run

    save_run(tmp_path / "pi-run", [result], {"available": False}, {"provider": "pi-scorer"})
    audit = json.loads((tmp_path / "pi-run" / "audit.jsonl").read_text())
    assert audit["requests"] == result["requests"]
    assert audit["metadata"]["actual_providers"] == ["pi-scorer"]


def synthetic_response():
    return {
        "model": "pi-scorer", "usage": {"input_tokens": 1, "output_tokens": 0},
        "answers": {
            "obs_1_status": {
                "type": "choice", "choice": "not_stated", "confidence": 0.95,
                "probabilities": {
                    "supported": 0.01, "not_stated": 0.97, "ambiguous": 0.01, "conflicting": 0.01,
                },
            },
        },
    }


@pytest.mark.parametrize("corruption", [
    "confidence", "probabilities", "model", "usage", "answers", "missing_id", "extra_id",
    "wrong_type", "bad_option", "bad_sum", "bad_keys", "not_max", "nan", "low_confidence",
])
def test_pi_malformed_answers_fail_and_confidence_is_not_invented(corruption, mock_sdk):
    body = synthetic_response()
    answer = body["answers"]["obs_1_status"]
    if corruption in {"confidence", "probabilities"}:
        del answer[corruption]
    elif corruption in {"model", "usage", "answers"}:
        del body[corruption]
    elif corruption == "missing_id":
        body["answers"].clear()
    elif corruption == "extra_id":
        body["answers"]["unexpected"] = dict(answer)
    elif corruption == "wrong_type":
        body["answers"]["obs_1_status"] = {"type": "noul", "noul": 0.8}
    elif corruption == "bad_option":
        answer["choice"] = SYNTHETIC_PRIVATE_BODY
    elif corruption == "bad_sum":
        answer["probabilities"]["not_stated"] = 0.6
    elif corruption == "bad_keys":
        answer["probabilities"].pop("supported")
    elif corruption == "not_max":
        answer["choice"] = "supported"
    elif corruption == "nan":
        answer["confidence"] = float("nan")
    elif corruption == "low_confidence":
        answer["confidence"] = 0.79
    body["private_extra"] = SYNTHETIC_PRIVATE_BODY
    mock_sdk(lambda _: httpx2.Response(200, content=json.dumps(body).encode()))
    registry = SchemaRegistry.from_entries([{"id": "1", "name": "Example", "value_type": "NUMERIC"}])
    with JevAdapter(provider="pi-scorer", enabled=True, api_key=SYNTHETIC_KEY) as adapter:
        result = extract("Synthetic example.", registry, adapter)
    assert result["observations"] == []
    if corruption == "low_confidence":
        assert result["status"] == "complete"
        assert result["audit"][0]["status"] == "review"
        assert result["audit"][0]["reason"] == "low_choice_confidence"
    else:
        assert result["status"] == "failed"
        assert result["failures"]
        assert result["audit"][0]["status"] == "failed"
    assert SYNTHETIC_PRIVATE_BODY not in json.dumps(result)
    assert SYNTHETIC_KEY not in json.dumps(result)


@pytest.mark.parametrize("failure", [401, 403, 429, 500, "timeout", "connection", "invalid_json"])
def test_pi_errors_are_sanitized_including_tracebacks(failure, mock_sdk, capsys):
    def handler(request):
        detail = SYNTHETIC_KEY + SYNTHETIC_PRIVATE_BODY
        if failure == "timeout":
            raise httpx2.ReadTimeout(detail, request=request)
        if failure == "connection":
            raise httpx2.ConnectError(detail, request=request)
        if failure == "invalid_json":
            return httpx2.Response(200, text=detail)
        return httpx2.Response(failure, json={"error": detail})

    mock_sdk(handler)
    with JevAdapter(provider="pi-scorer", enabled=True, api_key=SYNTHETIC_KEY) as adapter:
        with pytest.raises(ModelCallError, match="Pi Scorer.*inference failed") as caught:
            adapter.ask({}, {"q": Choice(instructions="Synthetic?", criteria={"a": "", "b": ""})})
    rendered = "".join(traceback.format_exception(caught.value))
    output = capsys.readouterr()
    assert SYNTHETIC_KEY not in rendered + output.out + output.err
    assert SYNTHETIC_PRIVATE_BODY not in rendered + output.out + output.err


def test_pi_does_not_invent_optional_request_id_or_token_counts(mock_sdk):
    body = {"model": "pi-scorer", "usage": {}, "answers": {"q": {"type": "noul", "noul": 0.6}}}
    mock_sdk(lambda _: httpx2.Response(200, json=body))
    with JevAdapter(provider="pi-scorer", enabled=True, api_key=SYNTHETIC_KEY) as adapter:
        result = adapter.ask({}, {"q": Noul(instructions="Synthetic?")})
    assert result.request_id is None
    assert result.usage == {"input_tokens": None, "output_tokens": None}
    assert result.answers["q"].noul == 0.6


@pytest.mark.parametrize("noul", [None, True, "0.9", -0.1, 1.1, float("nan"), float("inf")])
def test_pi_malformed_member_probabilities_fail_instead_of_emitting(noul, mock_sdk):
    def handler(request):
        payload = json.loads(request.content)
        answers = {
            qid: choice_reply(question, "supported") if question["type"] == "choice"
            else {"type": "noul", "noul": noul}
            for qid, question in payload["questions"].items()
        }
        return httpx2.Response(200, content=json.dumps({
            "model": "pi-scorer", "usage": {}, "answers": answers,
        }).encode())

    mock_sdk(handler)
    registry = SchemaRegistry.from_entries([
        {"id": "1", "name": "Synthetic members", "value_type": "MULTI_SELECT", "value_enum": ["A"]},
    ])
    with JevAdapter(provider="pi-scorer", enabled=True, api_key=SYNTHETIC_KEY) as adapter:
        result = extract("Synthetic example.", registry, adapter)
    assert result["status"] == "failed"
    assert result["observations"] == []
    assert result["audit"][0]["status"] == "failed"
