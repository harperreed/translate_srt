# 🚀 SRT Subtitle Translator

Welcome to the SRT Subtitle Translator! This project allows you to translate SRT (SubRip Subtitle) files between different languages using the OpenAI API. The tool is designed to help content creators, translators, and anyone needing to manage subtitles in multiple languages.

## 📜 Summary of the Project

The SRT Subtitle Translator reads SRT files, translates the subtitles from the source language to the target language, and outputs the translated subtitles into a new SRT file. The translated subtitles retain the timing and structure of the original files, allowing for seamless integration.

Key features include:

- Support for multiple languages
- Error handling and validation for input SRT files
- Token counting for OpenAI API cost estimation
- Progress tracking with a user-friendly interface
- Dry run option for analyzing token usage without performing translation

## 📦 How to Use

### Prerequisites

To use this project, ensure you have the following installed:

- Python 3.12 or higher
- Dependencies listed in `pyproject.toml`
- [uv](https://docs.astral.sh/uv/#getting-started) installed for running the script

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/harperreed/srt-subtitle-translator.git
    cd srt-subtitle-translator
    ```

2. Set up your OpenAI API Key in an environment variable or a `.env` file:
    ```plaintext
    OPENAI_API_KEY=your-api-key-here
    ```

### Running the Translator

#### Do a dry run

To translate an SRT file, use the following command:

```bash
uv run translate_srt.py input_file.srt output_file.srt --from "English" --to "Japanese" --dry-run
```

#### Translate the SRT file

To translate an SRT file, use the following command:

```bash
uv run translate_srt.py input_file.srt output_file.srt --from "English" --to "Japanese"
```

### Available Arguments:

- `input_file`: Path to the input SRT file.
- `output_file`: Path to save the translated SRT file.
- `--from`: Source language (default: English).
- `--to`: Target language (default: Japanese).
- `--model`: OpenAI model to use (default: gpt-4o-mini).
- `--dry-run`: Analyze token usage without performing translation.
- `--quiet`: Use simple progress bar instead of the TUI interface.

## ⚙️ Tech Info

- **Programming Language**: Python
- **Dependencies**:

    - `openai`: OpenAI API client
    - `python-dotenv`: Load environment variables from a `.env` file
    - `rich`: Beautiful console output and progress tracking
    - `srt`: Handling of SRT subtitle files
    - `tiktoken`: Token encoding for OpenAI API

- **File Structure**:

    ```
    .
    ├── .gitignore
    ├── .python-version
    ├── issues.md
    ├── missing-tests.md
    ├── pyproject.toml
    ├── translate_srt.py
    └── tests/
        └── test_srt_reader.py
    ```

- **Testing**: The project includes tests located in the `tests` directory, leveraging `pytest` for unit testing.

### 📝 Additional Information

- This project is designed with error handling and input validation in mind to provide a robust user experience.
- For any issues or feature requests, please refer to the `issues.md` file and feel free to contribute!

Feel free to reach out with any questions or feedback! Happy translating! 🎉

---

For more information, check out [my GitHub profile](https://github.com/harperreed) or connect with me for collaboration.
