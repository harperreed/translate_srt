import pytest
import tempfile
import os
from translate_srt import read_srt


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
    with pytest.raises(ValueError):
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
First subtitle
2
Invalid timestamp
Third subtitle"""
    with open(temp_srt_file, "w", encoding="utf-8") as f:
        f.write(malformed_content)

    with pytest.raises(SystemExit):
        read_srt(temp_srt_file)
