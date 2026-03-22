"""Tests for runserver VRAM detection and model selection."""

import subprocess
from unittest.mock import MagicMock, patch

from treemoissa.runserver import _detect_vram_mb, _select_best_model


# --- _detect_vram_mb ---


def test_detect_vram_parses_nvidia_smi():
    """_detect_vram_mb returns VRAM in MB when nvidia-smi succeeds."""
    mock_result = MagicMock()
    mock_result.stdout = "8192\n"
    with patch("treemoissa.runserver.subprocess.run", return_value=mock_result) as mock_run:
        result = _detect_vram_mb()
        assert result == 8192
        mock_run.assert_called_once()


def test_detect_vram_returns_none_on_file_not_found():
    """_detect_vram_mb returns None when nvidia-smi is not available."""
    with patch("treemoissa.runserver.subprocess.run", side_effect=FileNotFoundError):
        assert _detect_vram_mb() is None


def test_detect_vram_returns_none_on_error():
    """_detect_vram_mb returns None when nvidia-smi returns an error."""
    with patch(
        "treemoissa.runserver.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
    ):
        assert _detect_vram_mb() is None


def test_detect_vram_multi_gpu_picks_first():
    """With multiple GPUs, _detect_vram_mb returns the first GPU's VRAM."""
    mock_result = MagicMock()
    mock_result.stdout = "8192\n16384\n"
    with patch("treemoissa.runserver.subprocess.run", return_value=mock_result):
        assert _detect_vram_mb() == 8192


# --- _select_best_model ---


def test_select_model_8gb_picks_9b():
    """8 GB VRAM should select the 9B model."""
    repo, filename, display = _select_best_model(8192, "Q4_K_M")
    assert "9B" in repo
    assert "Q4_K_M" in filename


def test_select_model_24gb_picks_27b():
    """24 GB VRAM should select the 27B model."""
    repo, filename, display = _select_best_model(24576, "Q4_K_M")
    assert "27B" in repo


def test_select_model_4gb_picks_4b():
    """4 GB VRAM should select the 4B model."""
    repo, filename, display = _select_best_model(4096, "Q4_K_M")
    assert "4B" in repo


def test_select_model_3gb_picks_2b():
    """3 GB VRAM should select the 2B model."""
    repo, filename, display = _select_best_model(3072, "Q4_K_M")
    assert "2B" in repo


def test_select_model_2gb_picks_08b():
    """2 GB VRAM should select the 0.8B model."""
    repo, filename, display = _select_best_model(2048, "Q4_K_M")
    assert "0.8B" in repo


def test_select_model_none_vram_defaults_9b():
    """When VRAM is None (detection failed), default to 9B."""
    repo, filename, display = _select_best_model(None, "Q4_K_M")
    assert "9B" in repo
    assert "VRAM unknown" in display


def test_select_model_quant_override():
    """--quant should change the filename pattern."""
    repo, filename, display = _select_best_model(8192, "Q8_0")
    assert "Q8_0" in filename
    assert "9B" in repo


def test_select_model_tiny_vram_still_returns():
    """Even very small VRAM should return the smallest model with warning."""
    repo, filename, display = _select_best_model(512, "Q4_K_M")
    assert "0.8B" in repo
    assert "warning" in display.lower()
