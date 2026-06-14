from __future__ import annotations

import ast
from pathlib import Path

import pytest


FORBIDDEN_DEFAULT_DEPENDENCIES = {
    "accelerate",
    "av",
    "librosa",
    "soundfile",
    "stempeg",
    "tensorboard",
    "torchaudio",
    "wandb",
}


def _dependency_name(requirement: str) -> str:
    for separator in ("<", ">", "=", "!", "~", "[", ";"):
        requirement = requirement.split(separator, 1)[0]
    return requirement.strip().lower().replace("_", "-")


def test_default_dependencies_stay_minimal_and_audio_extras_are_optional():
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    default_dependencies = {
        _dependency_name(requirement)
        for requirement in pyproject["project"]["dependencies"]
    }
    optional_av_dependencies = {
        _dependency_name(requirement)
        for requirement in pyproject["project"]["optional-dependencies"]["av"]
    }

    assert default_dependencies.isdisjoint(FORBIDDEN_DEFAULT_DEPENDENCIES)
    assert optional_av_dependencies == {"av"}


def test_pymss_core_is_resolved_from_package_index_not_local_editable():
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    uv_sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})

    assert "pymss-core" not in uv_sources
    assert "pymss-core>=0.1.4" in pyproject["project"]["dependencies"]


def test_pymsst_source_does_not_import_optional_audio_or_logging_dependencies():
    forbidden_imports = FORBIDDEN_DEFAULT_DEPENDENCIES | {"pyav"}
    imported: set[str] = set()
    for path in sorted(Path("src/pymsst").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0].lower().replace("_", "-") for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0].lower().replace("_", "-"))

    assert imported.isdisjoint(forbidden_imports)
