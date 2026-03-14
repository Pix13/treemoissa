# tests/test_main_cli.py
"""Tests for treemoissa CLI argument parsing and mode detection."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_default_mode_is_llm():
    """Without --model, LLM mode should be used."""
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir"])
    assert args.model is None
    assert args.llm_host == "localhost:8080"


def test_llm_host_single():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir", "--llm-host", "10.0.0.1:9090"])
    assert args.llm_host == "10.0.0.1:9090"


def test_llm_host_multiple():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir", "--llm-host", "10.0.0.1:8080,10.0.0.2:8080"])
    assert args.llm_host == "10.0.0.1:8080,10.0.0.2:8080"


def test_llm_concurrency():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir", "--llm-concurrency", "3"])
    assert args.llm_concurrency == 3


def test_model_flag_selects_ml_mode():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir", "--model", "yolov8l"])
    assert args.model == "yolov8l"


def test_old_llm_flag_removed():
    """--llm should no longer be accepted."""
    from treemoissa.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["input_dir", "output_dir", "--llm"])


def test_old_llm_url_removed():
    """--llm-url should no longer be accepted."""
    from treemoissa.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["input_dir", "output_dir", "--llm-url", "http://x:8080"])


def test_default_concurrency():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir"])
    assert args.llm_concurrency == 1


def test_default_confidence():
    from treemoissa.main import _parse_args
    args = _parse_args(["input_dir", "output_dir"])
    assert args.confidence == 0.35


def test_invalid_model_rejected():
    from treemoissa.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["input_dir", "output_dir", "--model", "invalid"])


def test_ml_import_error(monkeypatch):
    """--model with missing ML deps should print error and exit."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    from treemoissa.main import _run_ml_pipeline
    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(SystemExit):
        _run_ml_pipeline(Path("/tmp"), Path("/tmp"), 0.35, "yolov8l")


def test_main_imports_without_torch():
    """main.py must import cleanly without torch."""
    import importlib
    import treemoissa.main
    # If we got here, no ImportError was raised at module level
    assert hasattr(treemoissa.main, "_parse_args")
