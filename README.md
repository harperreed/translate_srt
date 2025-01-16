# 📄 Subs Translator

## 📝 Summary of Project
Welcome to the **Subs Translator**! 🎉 This project is designed to help you seamlessly translate subtitle files from Japanese to English using OpenAI's language model. The goal of this repository is to provide a simple and efficient command-line tool to convert subtitle files while preserving their original timing and formatting.

## 🚀 How to Use
To get started with the Subs Translator, follow these simple steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/harperreed/subs-translator.git
   cd subs-translator
   ```

2. **Set Up Your Environment**:
   Make sure you have Python 3.12 or higher installed. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Required Packages**:
   Install necessary dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Obtain Your OpenAI API Key**:
   - Sign up at [OpenAI](https://openai.com/) if you haven't.
   - Set your OpenAI API key in a `.env` file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   
5. **Prepare Your Subtitle Files**:
   Ensure you have your Japanese subtitle file ready in `.srt` format.

6. **Run the Translation**:
   You can run the translation script by modifying the `input_file` and `output_file` variables in `subs-translate.py` to point to your subtitle files.

   Example:
   ```python
   input_file = 'path/to/your/input_file.ja.srt'
   output_file = 'path/to/your/output_file.en.srt'
   ```

   Then execute:
   ```bash
   python subs-translate.py
   ```

7. **Check the Output**:
   Once the translation is complete, check your output file at the specified location! 🎊

## 🔧 Tech Info
- **Programming Language**: Python
- **Dependencies**: 
  - `srt` for handling subtitle files
  - `openai` for accessing OpenAI's language models
  - `dotenv` for environment variable management
  - `tqdm` for progress bars on translations
- **Project Structure**:
  ```
  ├── .gitignore
  ├── .python-version
  ├── pyproject.toml
  └── subs-translate.py
  ```

- **OpenAI Model Used**: Currently set to interact with `gpt-4o-mini` for text translation.

Feel free to explore the code or contribute to the project! Your contributions and feedback are welcome. If you have any questions or issues, please raise an issue in this repository. Happy translating! 🌟
