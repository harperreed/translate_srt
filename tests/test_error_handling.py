import pytest
from unittest.mock import patch, Mock
from openai import APIError, RateLimitError, APIConnectionError
from translate_srt import translate_text, MAX_RETRIES

@pytest.fixture
def mock_openai():
    with patch('translate_srt.client.chat.completions.create') as mock:
        yield mock

@pytest.mark.asyncio
async def test_translation_api_error(mock_openai):
    """Test handling of API errors"""
    mock_openai.side_effect = APIError(
        message="API Error",
        request=None,
        body={"error": {"message": "API Error"}},
    )
    
    with pytest.raises(RuntimeError) as exc_info:
        await translate_text("Test text", "English", "Spanish", "gpt-4")
    
    assert "OpenAI API error" in str(exc_info.value)
    assert mock_openai.call_count == 1

@pytest.mark.asyncio
async def test_translation_rate_limit(mock_openai):
    """Test handling of rate limit errors"""
    mock_openai.side_effect = RateLimitError(
        message="Rate limit exceeded",
        response=Mock(status_code=429),
        body={"error": {"message": "Rate limit exceeded"}}
    )
    
    with pytest.raises(RuntimeError) as exc_info:
        await translate_text("Test text", "English", "Spanish", "gpt-4")
    
    assert "rate limit exceeded" in str(exc_info.value).lower()
    assert mock_openai.call_count == MAX_RETRIES

@pytest.mark.asyncio
async def test_translation_connection_error(mock_openai):
    """Test handling of connection errors"""
    mock_openai.side_effect = APIConnectionError(
        message="Connection failed",
        request=None
    )
    
    with pytest.raises(RuntimeError) as exc_info:
        await translate_text("Test text", "English", "Spanish", "gpt-4")
    
    assert "Failed to connect" in str(exc_info.value)
    assert mock_openai.call_count == MAX_RETRIES

@pytest.mark.asyncio
async def test_translation_invalid_response(mock_openai):
    """Test handling of invalid API responses"""
    # Mock a response with missing required fields
    mock_response = Mock()
    mock_response.choices = []
    mock_openai.return_value = mock_response
    
    with pytest.raises(RuntimeError) as exc_info:
        await translate_text("Test text", "English", "Spanish", "gpt-4")
    
    assert "Unexpected error" in str(exc_info.value)
    assert mock_openai.call_count == 1
