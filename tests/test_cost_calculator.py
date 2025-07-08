import pytest
from translate_srt import TranslationDisplay


def test_calculate_cost_valid_model():
    """Test cost calculation with valid model and token counts"""
    display = TranslationDisplay("gpt-4")
    cost = display.calculate_cost(1000, 1000)
    assert cost == 0.09  # (0.03 * 1 + 0.06 * 1)


def test_calculate_cost_unknown_model():
    """Test cost calculation with unknown model"""
    display = TranslationDisplay("unknown-model")
    cost = display.calculate_cost(1000, 1000)
    assert cost == 0.0


def test_calculate_cost_zero_tokens():
    """Test cost calculation with zero tokens"""
    display = TranslationDisplay("gpt-4")
    cost = display.calculate_cost(0, 0)
    assert cost == 0.0
