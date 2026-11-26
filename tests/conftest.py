import pytest

from frakobla_llm import config


@pytest.fixture(autouse=True)
def isolated_trust_file(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(config, "TRUSTED_FILE", tmp_path_factory.mktemp("trust") / "trusted.json")
