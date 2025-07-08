import pytest
import sys
import asyncio
from translate_srt import translate_srt, SUPPORTED_LANGUAGES
from unittest.mock import patch

def test_cli_valid_arguments(monkeypatch, tmp_path):
    """Test CLI with valid arguments"""
    input_file = tmp_path / "input.srt"
    output_file = tmp_path / "output.srt"
    input_file.write_text("1\n00:00:01,000 --> 00:00:04,000\nTest subtitle\n\n")
    
    test_args = [
        "translate_srt.py",
        str(input_file),
        str(output_file),
        "--from", "English",
        "--to", "Spanish",
        "--model", "gpt-4-1106-preview"
    ]
    
    with patch('translate_srt.translate_srt') as mock_translate:
        monkeypatch.setattr(sys, 'argv', test_args)
        import translate_srt.__main__
        asyncio.run(translate_srt.__main__.main())
        assert mock_translate.called
        call_args = mock_translate.call_args[0]
        assert str(input_file) == call_args[0]
        assert str(output_file) == call_args[1]
        assert "English" == call_args[2]
        assert "Spanish" == call_args[3]
        assert "gpt-4-1106-preview" == call_args[4]

def test_cli_missing_required_args(capsys):
    """Test CLI with missing required arguments"""
    test_args = ["translate_srt.py"]
    
    with pytest.raises(SystemExit) as exc_info:
        with patch('sys.argv', test_args):
            import translate_srt.__main__
    
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "required" in captured.err.lower()
    assert "input_file" in captured.err
    assert "output_file" in captured.err

def test_cli_invalid_file_paths(tmp_path):
    """Test CLI with invalid file paths"""
    nonexistent_file = tmp_path / "nonexistent.srt"
    output_file = tmp_path / "output.srt"
    
    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(translate_srt(
            str(nonexistent_file),
            str(output_file),
            "English",
            "Spanish",
            "gpt-4-1106-preview"
        ))
    
    assert exc_info.value.code == 1

def test_cli_invalid_language_codes():
    """Test CLI with invalid language codes"""
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(translate_srt(
            "input.srt",
            "output.srt",
            "InvalidLanguage",
            "Spanish",
            "gpt-4-1106-preview"
        ))
    
    assert "Unsupported language" in str(exc_info.value)
    assert all(lang in str(exc_info.value) for lang in SUPPORTED_LANGUAGES)

    # Test invalid target language
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(translate_srt(
            "input.srt",
            "output.srt",
            "English", 
            "InvalidLanguage",
            "gpt-4-1106-preview"
        ))
    
    assert "Unsupported language" in str(exc_info.value)
