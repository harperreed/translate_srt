import srt
import argparse
import sys
import time
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich import print as rprint

# Supported languages
SUPPORTED_LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese", 
    "Russian", "Japanese", "Chinese", "Korean", "Arabic"
]

console = Console()

load_dotenv()

client = OpenAI()

# Set your OpenAI API key


def read_srt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return list(srt.parse(file))
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Input file '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read SRT file: {str(e)}")
        sys.exit(1)

def write_srt(file_path, subtitles):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(srt.compose(subtitles))

def translate_text(text, source_lang, target_lang, model, max_retries=3, retry_delay=5):
    # Check input text length (rough estimate: 4 chars per token)
    if len(text) > 4000:  # ~1000 tokens
        raise ValueError("Text too long for translation. Please break into smaller chunks.")
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You are a professional translator. Translate the following {source_lang} text to {target_lang}. Maintain the original meaning and nuance as much as possible."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
            
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            raise RuntimeError("OpenAI API rate limit exceeded. Please try again later.")
            
        except APIConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise RuntimeError("Failed to connect to OpenAI API. Please check your internet connection.")
            
        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
            
        except Exception as e:
            raise RuntimeError(f"Unexpected error during translation: {str(e)}")

def validate_language(lang):
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang}\nSupported languages: {', '.join(SUPPORTED_LANGUAGES)}")

def translate_srt(input_file, output_file, source_lang, target_lang, model):
    validate_language(source_lang)
    validate_language(target_lang)
    
    if source_lang == target_lang:
        raise ValueError("Source and target languages must be different")
        
    with console.status("[bold green]Reading subtitles...") as status:
        subtitles = read_srt(input_file)
        total_subs = len(subtitles)
        status.update(f"[bold green]Found {total_subs} subtitles to translate")
    
    translated_subtitles = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Translating subtitles...", total=total_subs)
        
        for sub in subtitles:
            translated_content = translate_text(sub.content, source_lang, target_lang, model)
            
            translated_sub = srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=translated_content
            )
            
            translated_subtitles.append(translated_sub)
            progress.advance(task)
    
    with console.status("[bold green]Writing output file..."):
        write_srt(output_file, translated_subtitles)

if __name__ == '__main__':
    console.print("[bold blue]SRT Subtitle Translator[/bold blue]")
    console.print("Translates subtitles between languages using OPENAI\n")

    parser = argparse.ArgumentParser(description='Translate SRT subtitles between languages')
    parser.add_argument('input_file', help='Input SRT file path')
    parser.add_argument('output_file', help='Output SRT file path')
    parser.add_argument('--from', dest='source_lang', default='Japanese',
                      help=f'Source language (default: Japanese). Supported: {", ".join(SUPPORTED_LANGUAGES)}')
    parser.add_argument('--to', dest='target_lang', default='English',
                      help=f'Target language (default: English). Supported: {", ".join(SUPPORTED_LANGUAGES)}')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use (default: gpt-4o-mini)')
    
    args = parser.parse_args()
    
    try:
        translate_srt(args.input_file, args.output_file, args.source_lang, args.target_lang, args.model)
        console.print(f"\n[bold green]✓[/bold green] Translation completed successfully!")
        console.print(f"[dim]Output saved to:[/dim] {args.output_file}")
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        sys.exit(1)
