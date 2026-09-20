import getpass
import os
import warnings
from pathlib import Path

import nbformat
import pytest

from synur.jev import ModelCallError, configure_api_key

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "synur_observation_extraction.ipynb"


def test_disabled_does_not_prompt_or_change_environment(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(getpass, "getpass", lambda _: pytest.fail("Unexpected credential prompt"))
    assert configure_api_key() is None
    assert "TYPESAFE_API_KEY" not in os.environ


def test_reuses_environment_key_without_prompt(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-existing-credential")
    monkeypatch.setattr(getpass, "getpass", lambda _: pytest.fail("Unexpected credential prompt"))
    assert configure_api_key(enabled=True) is None
    assert os.environ["TYPESAFE_API_KEY"] == "synthetic-existing-credential"


@pytest.mark.parametrize("existing", [None, "", "   ", "REPLACE_WITH_YOUR_KEY"])
def test_masked_prompt_populates_environment_without_output(monkeypatch, capsys, existing):
    if existing is None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TYPESAFE_API_KEY", existing)
    calls = []

    def masked_input(prompt):
        calls.append(prompt)
        return " synthetic-entered-credential "

    monkeypatch.setattr(getpass, "getpass", masked_input)
    assert configure_api_key(enabled=True) is None
    assert calls == ["TypeSafe API key (hidden): "]
    assert os.environ["TYPESAFE_API_KEY"] == "synthetic-entered-credential"
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("entered", ["", "   ", "REPLACE_WITH_YOUR_KEY"])
def test_invalid_entry_is_not_saved(monkeypatch, entered):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(getpass, "getpass", lambda _: entered)
    with pytest.raises(ModelCallError):
        configure_api_key(enabled=True)
    assert "TYPESAFE_API_KEY" not in os.environ


@pytest.mark.parametrize("error", ["echo_warning", "eof"])
def test_unavailable_masked_input_fails_without_echo_fallback(monkeypatch, error):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    def unavailable(_):
        if error == "eof":
            raise EOFError()
        warnings.warn("No terminal echo control", getpass.GetPassWarning)
        pytest.fail("Must fail before an unmasked input fallback")

    monkeypatch.setattr(getpass, "getpass", unavailable)
    with pytest.raises(ModelCallError, match="Masked key input is unavailable"):
        configure_api_key(enabled=True)
    assert "TYPESAFE_API_KEY" not in os.environ


def test_setup_cell_runs_without_exposing_entered_key(monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(getpass, "getpass", lambda _: "synthetic-cell-credential")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(cell for cell in notebook.cells if cell.id == "api-key-setup")
    namespace = {"LIVE_CALLS": True}
    exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    assert os.environ["TYPESAFE_API_KEY"] == "synthetic-cell-credential"
    captured = capsys.readouterr()
    assert "configured for this kernel session" in captured.out
    assert "synthetic-cell-credential" not in captured.out + captured.err
    assert "synthetic-cell-credential" not in cell.source
