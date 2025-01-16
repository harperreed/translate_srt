import pytest
import tempfile
import os
import time
from translate_srt import read_srt, RateLimiter, TranslationDisplay


@pytest.fixture
def valid_srt_content():
    return """1
00:00:01,000 --> 00:00:04,000
Hello world!

2
00:00:04,001 --> 00:00:08,000
This is a test subtitle.
Multiple lines.

"""


@pytest.fixture
def temp_srt_file(valid_srt_content):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False, encoding="utf-8"
    ) as f:
        f.write(valid_srt_content)
    yield f.name
    os.unlink(f.name)


def test_read_srt_valid_file(temp_srt_file, valid_srt_content):
    """Test reading a valid SRT file"""
    subtitles = read_srt(temp_srt_file)
    assert len(subtitles) == 2
    assert subtitles[0].content == "Hello world!"
    assert subtitles[1].content == "This is a test subtitle.\nMultiple lines."


def test_read_srt_file_not_found():
    """Test handling of non-existent file"""
    with pytest.raises(SystemExit):
        read_srt("nonexistent_file.srt")


def test_read_srt_invalid_encoding(temp_srt_file):
    """Test handling of files with invalid encoding"""
    # Create a file with invalid UTF-8
    with open(temp_srt_file, "wb") as f:
        f.write(b"\xff\xfe\x00\x00Invalid UTF-8 content")

    with pytest.raises(SystemExit):
        read_srt(temp_srt_file)


def test_read_srt_malformed_content(temp_srt_file):
    """Test handling of malformed SRT content"""
    malformed_content = """1
00:00:01,000 --> 00:00:04,000
Missing blank line
2
Invalid timestamp --> Invalid
Text content
"""
    with open(temp_srt_file, "w", encoding="utf-8") as f:
        f.write(malformed_content)

    with pytest.raises(SystemExit):
        read_srt(temp_srt_file)


def test_rate_limiter_initial_state():
    """Test initial state of rate limiter"""
    limiter = RateLimiter(window_size=60, max_requests=3)
    assert limiter.can_make_request() == True
    assert len(limiter.request_times) == 0


def test_rate_limiter_request_tracking():
    """Test request tracking over time window"""
    limiter = RateLimiter(window_size=60, max_requests=3)
    
    # Make 3 requests (should be allowed)
    for _ in range(3):
        assert limiter.can_make_request() == True
        limiter.add_request()
    
    # 4th request should be denied
    assert limiter.can_make_request() == False
    assert len(limiter.request_times) == 3


def test_rate_limiter_window_expiry():
    """Test requests expire after window time"""
    limiter = RateLimiter(window_size=1, max_requests=2)  # 1 second window
    
    # Make 2 requests
    for _ in range(2):
        assert limiter.can_make_request() == True
        limiter.add_request()
    
    # Wait for window to expire
    time.sleep(1.1)
    
    # Should be allowed to make new requests
    assert limiter.can_make_request() == True
    assert len(limiter.request_times) == 0


def test_rate_limiter_batch_delay():
    """Test batch delay triggering"""
    limiter = RateLimiter(window_size=60, max_requests=3)
    
    # Make 2 quick requests
    for _ in range(2):
        limiter.add_request()
    
    # Should trigger batch delay
    assert limiter.should_batch_delay() == True
    
    # Wait a bit
    time.sleep(0.1)
    
    # One more request
    limiter.add_request()
    
    # Should still want to delay as we're at limit
    assert limiter.should_batch_delay() == True


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
