"""The product is called Docent ("Talk to your documents"): in the config, the OpenAPI docs and every prompt in which
the assistant speaks as itself, so it answers "who are you?" with its name (docs/DESIGN.md, decided 2026-10-10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.branding import IDENTITY, PRODUCT_NAME, TAGLINE, TAGLINE_HI
from app.main import create_app
from app.providers.registry import build_container
from app.services import prompts

from .conftest import mock_http

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("en", "hi")


def test_names():
    assert (PRODUCT_NAME, TAGLINE, TAGLINE_HI) == ("Docent", "Talk to your documents", "अपने दस्तावेज़ों से बात करें")


@pytest.mark.parametrize("language", LANGUAGES)
def test_assistant_prompts_introduce_docent(language):
    spoken = [
        prompts.answer_system_prompt(language),
        prompts.answer_system_prompt(language, mixed=True),
        prompts.general_system_prompt(language),
        prompts.live_system_prompt(language, documents="sources"),
        prompts.conversation_system_prompt(language),
        prompts.clarification_system_prompt(language),
    ]
    for system in spoken:
        assert system.startswith(f"You are {PRODUCT_NAME},"), system[:80]


@pytest.mark.parametrize("language", LANGUAGES)
def test_conversation_prompt_answers_who_are_you_as_docent(language):
    system = prompts.conversation_system_prompt(language)
    assert IDENTITY in system
    assert "say you are Docent" in system


@pytest.mark.parametrize("name", ["local", "docker", "cloud"])
def test_config_profiles_use_the_product_name(name):
    config = json.loads((ROOT / "config" / f"{name}.config.json").read_text())
    assert config["client"]["app_title"] == PRODUCT_NAME


def test_openapi_is_titled_docent(load_local):
    s = load_local()
    app = create_app(s, build_container(s, http=mock_http()), preload_models=False)
    with TestClient(app) as client:
        info = client.get("/openapi.json").json()["info"]
    assert info["title"] == PRODUCT_NAME and info["description"] == TAGLINE


def test_no_generic_name_left_in_the_sources():
    old = "document voice agent"
    for folder in ("backend/app", "config", "frontend/app", "frontend/components", "frontend/lib"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".ts", ".tsx", ".md"}:
                assert old not in path.read_text(encoding="utf-8").casefold(), path
