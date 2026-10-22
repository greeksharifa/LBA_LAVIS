import runpy
from pathlib import Path

import setuptools


def test_package_includes_subqa_runtime_dependency(monkeypatch):
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))
    assert "subqa" in captured["packages"]
    assert not any(name == "tests" or name.startswith("tests.") for name in captured["packages"])
