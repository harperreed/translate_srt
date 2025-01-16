import pytest
from unittest.mock import Mock, patch, MagicMock
import srt
from datetime import timedelta
from translate_srt import translate_srt, TranslationDisplay, validate_srt_format
from rich.console import Console
from rich.live import Live

@pytest.mark.asyncio
async def test_openai_client_mocked():
    """Test translation with mocked OpenAI client"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Translated text"))]
    
    with patch('translate_srt.read_srt') as mock_read:
        with patch('translate_srt.client.chat.completions.create', return_value=mock_response) as mock_create:
            with patch('translate_srt.write_srt') as mock_write:
                # Create sample subtitle
                subtitle = srt.Subtitle(
                    index=1,
                    start=timedelta(seconds=0),
                    end=timedelta(seconds=2),
                    content="Test content"
                )
                mock_read.return_value = [subtitle]
                
                # Run translation with mocked dependencies
                await translate_srt(
                "input.srt",
                "output.srt",
                "English",
                "Spanish",
                "gpt-4",
                quiet=True
            )
            
            # Verify OpenAI API was called with correct parameters
            assert mock_create.called
            call_args = mock_create.call_args[1]
            assert call_args['model'] == "gpt-4"
            assert any("English" in msg['content'] for msg in call_args['messages'])
            assert any("Spanish" in msg['content'] for msg in call_args['messages'])

@pytest.mark.asyncio
async def test_file_operations_mocked(tmp_path):
    """Test file operations with mocked filesystem"""
    input_file = tmp_path / "input.srt"
    output_file = tmp_path / "output.srt"
    
    # Create test input file
    test_content = """1
00:00:01,000 --> 00:00:04,000
Test subtitle

"""
    input_file.write_text(test_content)
    
    with patch('translate_srt.client.chat.completions.create') as mock_create:
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Translated subtitle"))]
        mock_create.return_value = mock_response
            
        # Mock translate_text to return a string instead of coroutine
        async def mock_translate_text(*args, **kwargs):
            return "Translated subtitle"
                
        with patch('translate_srt.translate_text', side_effect=mock_translate_text):
            # Run translation
            await translate_srt(
            str(input_file),
            str(output_file),
            "English",
            "Spanish",
            "gpt-4",
            quiet=True
        )
        
        # Verify output file was created with correct content
            assert output_file.exists()
            output_content = output_file.read_text()
            assert "Translated subtitle" in output_content
            assert "00:00:01,000 --> 00:00:04,000" in output_content

def test_progress_display_mocked():
    """Test progress display with mocked console"""
    with patch('rich.live.Live') as mock_live:
        with patch('rich.console.Console') as mock_console:
            display = TranslationDisplay("gpt-4")
            
            # Test display updates
            display.update_header(1, 10, 1000, 0.25)
            display.update_progress(5, 10)
            display.update_stats(100, 150, 1000, 1500)
            
            # Verify layout updates
            assert len(display.layout["header"].renderable.renderable) > 0
            
            # Test cost calculation
            cost = display.calculate_cost(1000, 1500)
            assert isinstance(cost, float)
            assert cost > 0

def test_srt_validation_mocked(tmp_path):
    """Test SRT validation with mocked filesystem"""
    # Test valid SRT
    valid_content = """1
00:00:01,000 --> 00:00:04,000
Test subtitle

"""
    validate_srt_format(valid_content)
    
    # Test invalid SRT
    invalid_content = """Invalid SRT format"""
    with pytest.raises(SystemExit):
        validate_srt_format(invalid_content)
