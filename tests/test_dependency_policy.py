from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

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
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

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
    speaker_dependencies = {
        _dependency_name(requirement)
        for requirement in pyproject["project"]["optional-dependencies"]["speakers"]
    }
    assert speaker_dependencies == {"soundfile", "scipy", "librosa"}


def test_pymss_core_is_resolved_from_package_index_not_local_editable():
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    uv_sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})

    assert "pymss-core" not in uv_sources
    assert "pymss-core>=0.1.4" in pyproject["project"]["dependencies"]


def test_optional_dependencies_are_confined_to_lazy_speaker_feature_imports():
    forbidden_imports = FORBIDDEN_DEFAULT_DEPENDENCIES | {"pyav"}
    allowed = {"data/speakers.py": {"soundfile", "librosa"}}
    source = ROOT / "src/pymsst"
    for path in sorted(source.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            names = set()
            if isinstance(node, ast.Import):
                names = {alias.name.split(".", 1)[0].lower().replace("_", "-") for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".", 1)[0].lower().replace("_", "-")}
            optional = names & forbidden_imports
            if not optional:
                continue
            assert optional <= allowed.get(path.relative_to(source).as_posix(), set()), path
            ancestor = parents.get(node)
            while ancestor is not None and not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ancestor = parents.get(ancestor)
            assert ancestor is not None, f"optional import must be inside a function: {path}"


def test_cli_loads_when_speaker_audio_extras_are_unavailable():
    import subprocess
    import sys

    code = ("import sys; sys.modules.update({'soundfile': None, 'librosa': None}); "
            "from pymsst.cli import build_parser; build_parser()")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
