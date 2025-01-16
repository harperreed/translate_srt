import pytest
from translate_srt import TranslationDisplay, MODEL_PRICING

@pytest.fixture
def display():
    """Fixture to create a TranslationDisplay instance"""
    return TranslationDisplay("gpt-4-1106-preview")

def test_display_update_header(display):
    """Test header updates with progress"""
    display.update_header(5, 10, 1000, 0.25)
    header_content = display.layout["header"].renderable.content
    assert "5/10" in header_content
    assert "1,000" in header_content
    assert "$0.25" in header_content
    assert "gpt-4-1106-preview" in header_content

def test_display_update_stats(display):
    """Test statistics display updates"""
    display.update_stats(100, 150, 1000, 1500)
    stats_content = display.layout["stats"].renderable.content
    assert "1,000" in stats_content  # Total prompt tokens
    assert "1,500" in stats_content  # Total completion tokens
    assert "2,500" in stats_content  # Total tokens (1000 + 1500)
    
    # Test cost calculation
    input_rate, output_rate = MODEL_PRICING["gpt-4-1106-preview"]
    expected_cost = (1000/1000 * input_rate) + (1500/1000 * output_rate)
    cost_str = f"${expected_cost:.2f}"
    assert cost_str in stats_content

def test_display_progress_bar(display):
    """Test progress bar calculations"""
    display.update_progress(7, 10)
    progress_content = display.layout["progress"].renderable.content
    
    # Check percentage
    assert "70.0%" in progress_content
    assert "7/10" in progress_content
    
    # Check progress bar visualization
    filled_chars = "█" * 35  # 70% of 50 width
    assert filled_chars in progress_content

def test_display_cost_formatting(display):
    """Test cost display formatting"""
    # Test with different token counts
    prompt_tokens = 1234
    completion_tokens = 5678
    
    total_cost = display.calculate_cost(prompt_tokens, completion_tokens)
    assert isinstance(total_cost, float)
    
    # Verify the cost calculation
    input_rate, output_rate = MODEL_PRICING["gpt-4-1106-preview"]
    expected_cost = (prompt_tokens/1000 * input_rate) + (completion_tokens/1000 * output_rate)
    assert total_cost == pytest.approx(expected_cost)
    
    # Test formatting in header display
    display.update_header(1, 10, prompt_tokens + completion_tokens, total_cost)
    header_content = display.layout["header"].renderable.content
    assert f"${total_cost:.4f}" in header_content

def test_display_unknown_model():
    """Test display handling of unknown model"""
    unknown_display = TranslationDisplay("unknown-model")
    cost = unknown_display.calculate_cost(1000, 2000)
    assert cost == 0.0
    
    # Should still format stats without cost information
    unknown_display.update_stats(100, 150, 1000, 1500)
    stats_content = unknown_display.layout["stats"].renderable.content
    assert "2,500" in stats_content  # Should show total tokens
