"""Tests for treemoissa.utils."""

from treemoissa.utils import _sanitize


def test_sanitize_lowercases():
    assert _sanitize("Porsche") == "porsche"


def test_sanitize_replaces_spaces():
    assert _sanitize("Aston Martin") == "aston_martin"


def test_sanitize_removes_unsafe_chars():
    assert _sanitize("911/turbo") == "911turbo"


def test_sanitize_strips_whitespace():
    assert _sanitize("  supra  ") == "supra"


def test_sanitize_empty_returns_unknown():
    assert _sanitize("") == "unknown"


def test_sanitize_only_unsafe_chars_returns_unknown():
    assert _sanitize("///") == "unknown"


def test_sanitize_preserves_hyphens():
    assert _sanitize("Rolls-Royce") == "rolls-royce"


def test_sanitize_mixed_special_chars():
    assert _sanitize("911 GT3 (RS)") == "911_gt3_rs"
