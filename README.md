```markdown
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
  - openai >= 1.59.7
  - python-dotenv >= 1.0.1
  - rich >= 13.9.4
  - srt >= 3.5.3

For further technical specifications, please refer to the code within the repository. 

--- 
📅 Last updated on: 2025-01-16
```

