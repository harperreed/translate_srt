import pytest
from translate_srt import validate_language, SUPPORTED_LANGUAGES

def test_validate_language_supported():
    """Test validation of supported language"""
    # Test a few supported languages
    for lang in ["English", "Spanish", "Japanese"]:
        validate_language(lang)  # Should not raise any exception

def test_validate_language_unsupported():
    """Test validation of unsupported language"""
    with pytest.raises(ValueError) as exc_info:
        validate_language("InvalidLanguage")
    assert "Unsupported language" in str(exc_info.value)
    assert all(lang in str(exc_info.value) for lang in SUPPORTED_LANGUAGES)

def test_validate_language_case_sensitivity():
    """Test language validation case sensitivity"""
    # Test that case matters
    with pytest.raises(ValueError):
        validate_language("english")  # Should fail because it's not "English"
    with pytest.raises(ValueError):
        validate_language("ENGLISH")  # Should fail because it's not "English"

def test_validate_language_whitespace():
    """Test language validation with whitespace"""
    # Test whitespace handling
    with pytest.raises(ValueError):
        validate_language(" English")  # Leading space
    with pytest.raises(ValueError):
        validate_language("English ")  # Trailing space
    with pytest.raises(ValueError):
        validate_language(" English ")  # Both leading and trailing space
