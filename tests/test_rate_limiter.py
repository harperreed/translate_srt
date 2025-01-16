import pytest
import time
from translate_srt import RateLimiter


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
