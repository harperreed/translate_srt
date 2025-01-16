import srt
import argparse
import sys
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich import print as rprint

console = Console()

load_dotenv()

client = OpenAI()
from tqdm import tqdm

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

def translate_text(text):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a professional translator. Translate the following Japanese text to English. Maintain the original meaning and nuance as much as possible."},
        {"role": "user", "content": text}
    ])

    return response.choices[0].message.content.strip()

def translate_srt(input_file, output_file):
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
            translated_content = translate_text(sub.content)
            
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
    console.print("Translates Japanese subtitles to English using GPT-4\n")

    parser = argparse.ArgumentParser(description='Translate SRT subtitles from Japanese to English')
    parser.add_argument('input_file', help='Input SRT file path')
    parser.add_argument('output_file', help='Output SRT file path')
    
    args = parser.parse_args()
    
    try:
        translate_srt(args.input_file, args.output_file)
        console.print(f"\n[bold green]✓[/bold green] Translation completed successfully!")
        console.print(f"[dim]Output saved to:[/dim] {args.output_file}")
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        sys.exit(1)
