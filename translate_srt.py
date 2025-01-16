import srt
import argparse
import sys
import signal
import random
import asyncio
from datetime import datetime, timedelta
from typing import List, Sequence, Dict, Tuple
from openai import OpenAI
import tiktoken
from openai import APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel

# Supported languages
SUPPORTED_LANGUAGES: List[str] = [
    "English",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Russian",
    "Japanese",
    "Chinese",
    "Korean",
    "Arabic",
]

console = Console()

load_dotenv()

client = OpenAI()

# Model pricing per 1K tokens (input, output)
# Rate limiting settings
MAX_RETRIES = 5
MIN_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 60  # seconds
BATCH_SIZE = 10  # number of requests before forced delay
BATCH_DELAY = 2  # seconds between batches
REQUEST_WINDOW = 60  # rolling window in seconds
MAX_REQUESTS_PER_WINDOW = 50  # max requests per window

MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4-1106-preview": (
        0.01,
        0.03,
    ),  # $10.00 / 1M input tokens, $30.00 / 1M output tokens
    "gpt-4": (0.03, 0.06),  # $30.00 / 1M input tokens, $60.00 / 1M output tokens
    "gpt-4-32k": (0.06, 0.12),  # $60.00 / 1M input tokens, $120.00 / 1M output tokens
    "gpt-3.5-turbo-1106": (
        0.0015,
        0.002,
    ),  # $1.50 / 1M input tokens, $2.00 / 1M output tokens
    "gpt-3.5-turbo": (
        0.0015,
        0.002,
    ),  # $1.50 / 1M input tokens, $2.00 / 1M output tokens
    "gpt-3.5-turbo-16k": (
        0.003,
        0.004,
    ),  # $3.00 / 1M input tokens, $4.00 / 1M output tokens
    "gpt-4o": (0.0025, 0.01),  # $2.50 / 1M input tokens, $10.00 / 1M output tokens
    "gpt-4o-mini": (
        0.00015,
        0.0006,
    ),  # $0.15 / 1M input tokens, $0.60 / 1M output tokens
    "o1": (0.015, 0.06),  # $15.00 / 1M input tokens, $60.00 / 1M output tokens
    "o1-mini": (0.003, 0.012),  # $3.00 / 1M input tokens, $12.00 / 1M output tokens
}


def validate_srt_format(content: str) -> None:
    """Validate SRT file format requirements.

    Args:
        content: The content of the SRT file as a string

    Raises:
        SystemExit: If the SRT format is invalid
    """
    if not content.strip():
        console.print("[red]Error:[/red] SRT file is empty")
        sys.exit(1)

    # Check basic structure
    entries = [e for e in content.strip().split("\n\n") if e.strip()]
    if not entries:
        console.print("[red]Error:[/red] No subtitle entries found")
        sys.exit(1)

    for i, entry in enumerate(entries, 1):
        lines = [line for line in entry.strip().split("\n") if line.strip()]
        if len(lines) < 3:
            console.print(
                f"[red]Error:[/red] Invalid entry format at subtitle {i}: Missing required components"
            )
            sys.exit(1)

        # Validate entry has proper blank line separation
        if i < len(entries):
            entry_parts = content.split("\n\n")
            if i < len(entry_parts) and not entry_parts[i-1].strip():
                console.print(f"[red]Error:[/red] Empty subtitle entry at position {i}")
                sys.exit(1)

        # Validate index number
        try:
            index = int(lines[0])
            if index != i:
                console.print(
                    f"[red]Error:[/red] Invalid subtitle index at entry {i}: Expected {i}, got {index}"
                )
                sys.exit(1)
        except ValueError:
            console.print(
                f"[red]Error:[/red] Invalid subtitle index at entry {i}: Must be a number"
            )
            sys.exit(1)

        # Validate timestamp format
        try:
            timestamp = lines[1]
            if " --> " not in timestamp:
                console.print(
                    f"[red]Error:[/red] Invalid timestamp format at entry {i}: Missing separator ' --> '"
                )
                sys.exit(1)

            start, end = timestamp.split(" --> ")
            try:
                srt.srt_timestamp_to_timedelta(start)
                srt.srt_timestamp_to_timedelta(end)
            except (ValueError, AttributeError):
                console.print(
                    f"[red]Error:[/red] Invalid timestamp format at entry {i}: Must be in HH:MM:SS,mmm format"
                )
                sys.exit(1)
        except IndexError:
            console.print(
                f"[red]Error:[/red] Invalid timestamp format at entry {i}: Missing timestamp line"
            )
            sys.exit(1)

        # Validate subtitle text
        if not "".join(lines[2:]).strip():
            console.print(f"[red]Error:[/red] Empty subtitle text at entry {i}")
            sys.exit(1)


def read_srt(file_path: str) -> List[srt.Subtitle]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Validate SRT format
        validate_srt_format(content)

        # Parse after validation
        return list(srt.parse(content))
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Input file '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read SRT file: {str(e)}")
        sys.exit(1)


def write_srt(file_path: str, subtitles: Sequence[srt.Subtitle]) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(srt.compose(subtitles))
    except PermissionError:
        raise RuntimeError(
            f"Permission denied when writing to '{file_path}'. Check file permissions."
        )
    except OSError as e:
        raise RuntimeError(f"Failed to write output file '{file_path}': {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while writing output file: {str(e)}")


class RateLimiter:
    def __init__(
        self,
        window_size: int = REQUEST_WINDOW,
        max_requests: int = MAX_REQUESTS_PER_WINDOW,
    ):
        self.window_size = window_size
        self.max_requests = max_requests
        self.request_times = []
        self.batch_count = 0

    def can_make_request(self) -> bool:
        now = datetime.now()
        # Remove requests outside the window
        self.request_times = [
            t for t in self.request_times if now - t < timedelta(seconds=self.window_size)
        ]
        return len(self.request_times) < self.max_requests

    def add_request(self):
        self.request_times.append(datetime.now())
        self.batch_count += 1

    def should_batch_delay(self) -> bool:
        return self.batch_count >= 2  # Trigger after 2 requests


def calculate_retry_delay(attempt: int) -> float:
    """Calculate exponential backoff with jitter."""
    exp_delay = min(MAX_RETRY_DELAY, MIN_RETRY_DELAY * (2**attempt))
    jitter = random.uniform(0, 0.1 * exp_delay)  # 10% jitter
    return exp_delay + jitter


# Global rate limiter instance
rate_limiter = RateLimiter()


async def translate_text(
    text: str, source_lang: str, target_lang: str, model: str
) -> str:
    # Check input text length (rough estimate: 4 chars per token)
    if len(text) > 4000:  # ~1000 tokens
        raise ValueError(
            "Text too long for translation. Please break into smaller chunks."
        )

    for attempt in range(MAX_RETRIES):
        try:
            # Check rate limits
            while not rate_limiter.can_make_request():
                await asyncio.sleep(1)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Translate the following {source_lang} text to {target_lang}. Maintain the original meaning and nuance as much as possible.",
                    },
                    {"role": "user", "content": text},
                ],
            )

            rate_limiter.add_request()

            # Add batch delay if needed
            if rate_limiter.should_batch_delay():
                await asyncio.sleep(BATCH_DELAY)

            return response.choices[0].message.content.strip()

        except RateLimitError:
            if attempt < MAX_RETRIES - 1:
                delay = calculate_retry_delay(attempt)
                await asyncio.sleep(delay)
                continue
            raise RuntimeError(
                "OpenAI API rate limit exceeded. Please try again later."
            )

        except APIConnectionError:
            if attempt < MAX_RETRIES - 1:
                delay = calculate_retry_delay(attempt)
                await asyncio.sleep(delay)
                continue
            raise RuntimeError(
                "Failed to connect to OpenAI API. Please check your internet connection."
            )

        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error during translation: {str(e)}")


def validate_language(lang: str) -> None:
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {lang}\nSupported languages: {', '.join(SUPPORTED_LANGUAGES)}"
        )


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in the text for the specified model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


class TranslationDisplay:
    def __init__(self, model: str):
        self.layout = Layout()
        self.model = model
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="stats", size=4),
            Layout(name="progress", size=3),
        )
        self.layout["main"].split_row(
            Layout(name="source", ratio=1), Layout(name="target", ratio=1)
        )

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        if self.model not in MODEL_PRICING:
            return 0.0
        input_cost = (prompt_tokens / 1000) * MODEL_PRICING[self.model][0]
        output_cost = (completion_tokens / 1000) * MODEL_PRICING[self.model][1]
        return input_cost + output_cost

    def update_header(
        self, current: int, total: int, total_tokens: int, total_cost: float
    ):
        self.layout["header"].update(
            Panel(
                f"[bold blue]Translating subtitle {current}/{total} using [/bold blue][green]{self.model}[/green]\n"
                f"Total tokens: [yellow]{total_tokens:,}[/yellow] | "
                f"Total cost: [green]${total_cost:.4f}[/green]",
                title="SRT Translator",
            )
        )

    def update_source(self, text: str):
        self.layout["source"].update(
            Panel(text, title="Source Text", border_style="blue")
        )

    def update_target(self, text: str):
        self.layout["target"].update(
            Panel(text, title="Translation", border_style="green")
        )

    def update_stats(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_prompt_tokens: int,
        total_completion_tokens: int,
    ):
        if self.model in MODEL_PRICING:
            input_rate, output_rate = MODEL_PRICING[self.model]
            avg_rate = (input_rate + output_rate) / 2  # Simple average for display
            total_tokens = total_prompt_tokens + total_completion_tokens
            total_cost = self.calculate_cost(
                total_prompt_tokens, total_completion_tokens
            )

            self.layout["stats"].update(
                Panel(
                    f"Prompt Tokens: [blue]{total_prompt_tokens:,}[/blue], "
                    f"Completion Tokens: [blue]{total_completion_tokens:,}[/blue] | "
                    f"Total Tokens: [yellow]{total_tokens:,}[/yellow] "
                    f"([green]${total_cost:.2f}[/green] @ 1k/${avg_rate:.3f})",
                    border_style="cyan",
                )
            )
        else:
            total_tokens = total_prompt_tokens + total_completion_tokens
            self.layout["stats"].update(
                Panel(
                    f"Prompt Tokens: [blue]{total_prompt_tokens:,}[/blue], "
                    f"Completion Tokens: [blue]{total_completion_tokens:,}[/blue] | "
                    f"Total Tokens: [yellow]{total_tokens:,}[/yellow]",
                    border_style="cyan",
                )
            )

    def update_progress(self, current: int, total: int):
        progress_width = 50  # Width of the progress bar in characters
        filled = int(progress_width * current / total)
        bar = "█" * filled + "░" * (progress_width - filled)
        percentage = current / total * 100

        self.layout["progress"].update(
            Panel(
                f"{bar} {percentage:>5.1f}% ({current}/{total})", border_style="yellow"
            )
        )


def display_final_summary(
    total_subs: int,
    completed_subs: int,
    total_prompt_tokens: int,
    total_completion_tokens: int,
    model: str,
    interrupted: bool = False,
) -> None:
    """Display a summary of the translation run."""
    total_tokens = total_prompt_tokens + total_completion_tokens

    console.print("\n[bold yellow]Translation Summary:[/bold yellow]")
    console.print(
        f"Status: {'[red]Interrupted[/red]' if interrupted else '[green]Completed[/green]'}"
    )
    console.print(
        f"Progress: [blue]{completed_subs}/{total_subs}[/blue] subtitles processed"
    )
    console.print(f"Total prompt tokens: [yellow]{total_prompt_tokens:,}[/yellow]")
    console.print(
        f"Total completion tokens: [yellow]{total_completion_tokens:,}[/yellow]"
    )
    console.print(f"Total tokens: [bold yellow]{total_tokens:,}[/bold yellow]")

    if model in MODEL_PRICING:
        input_rate, output_rate = MODEL_PRICING[model]
        input_cost = (total_prompt_tokens / 1000) * input_rate
        output_cost = (total_completion_tokens / 1000) * output_rate
        total_cost = input_cost + output_cost

        console.print(f"Input cost: [green]${input_cost:.4f}[/green]")
        console.print(f"Output cost: [green]${output_cost:.4f}[/green]")
        console.print(f"Total cost: [bold green]${total_cost:.4f}[/bold green]")


def translate_srt(
    input_file: str,
    output_file: str,
    source_lang: str,
    target_lang: str,
    model: str,
    dry_run: bool = False,
    quiet: bool = False,
) -> None:
    validate_language(source_lang)
    validate_language(target_lang)

    if source_lang == target_lang:
        raise ValueError("Source and target languages must be different")

    with console.status("[bold green]Reading subtitles...") as status:
        subtitles = read_srt(input_file)
        total_subs = len(subtitles)
        status.update(f"[bold green]Found {total_subs} subtitles to translate")

    if quiet and not dry_run:
        translate_srt_quiet(subtitles, output_file, source_lang, target_lang, model)
        return
    elif dry_run:
        total_tokens = 0
        prompt_tokens = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("[cyan]Analyzing subtitles...", total=total_subs)

            for sub in subtitles:
                # Count tokens in the system message
                system_msg = f"You are a professional translator. Translate the following {source_lang} text to {target_lang}. Maintain the original meaning and nuance as much as possible."
                prompt_tokens += count_tokens(system_msg, model)

                # Count tokens in the subtitle content
                content_tokens = count_tokens(sub.content, model)
                prompt_tokens += content_tokens

                # Estimate response tokens (assuming translation is ~1.5x the input)
                response_tokens = int(content_tokens * 1.5)
                total_tokens += prompt_tokens + response_tokens

                progress.advance(task)

        # Calculate cost estimates
        if model not in MODEL_PRICING:
            console.print(
                f"\n[yellow]Warning:[/yellow] Cost estimation not available for model '{model}'"
            )
            # cost_estimate = "Unknown"
        else:
            input_cost = (prompt_tokens / 1000) * MODEL_PRICING[model][0]
            output_cost = ((total_tokens - prompt_tokens) / 1000) * MODEL_PRICING[
                model
            ][1]
            total_cost = input_cost + output_cost

        console.print("\n[bold green]Dry Run Summary:[/bold green]")
        console.print(f"Model: [blue]{model}[/blue]")
        console.print(f"Number of subtitles: [yellow]{total_subs:,}[/yellow]")

        if model in MODEL_PRICING:
            input_rate, output_rate = MODEL_PRICING[model]
            estimated_output_tokens = total_tokens - prompt_tokens
            input_cost = (prompt_tokens / 1000) * input_rate
            output_cost = (estimated_output_tokens / 1000) * output_rate

            console.print(
                f"Input tokens: [blue]{prompt_tokens:,}[/blue] (${input_cost:.4f} @ ${input_rate:.3f}/1K tokens)"
            )
            console.print(
                f"Estimated output tokens: [blue]{estimated_output_tokens:,}[/blue] (${output_cost:.4f} @ ${output_rate:.3f}/1K tokens)"
            )
            console.print(f"Estimated total tokens: [yellow]{total_tokens:,}[/yellow]")
            console.print(
                f"[bold green]Estimated total cost: ${input_cost + output_cost:.4f}[/bold green]"
            )
        else:
            console.print(f"Estimated prompt tokens: {prompt_tokens:,}")
            console.print(f"Estimated total tokens: {total_tokens:,}")
            console.print(
                "[yellow]Note: Cost estimation not available for this model[/yellow]"
            )
        return

    translated_subtitles = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    display = TranslationDisplay(model)

    # Setup signal handler
    def signal_handler(sig, frame):
        display_final_summary(
            total_subs,
            len(translated_subtitles),
            total_prompt_tokens,
            total_completion_tokens,
            model,
            interrupted=True,
        )
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        with Live(display.layout, refresh_per_second=4, screen=True):
            for sub in subtitles:
                # Count tokens for this subtitle
                system_msg = f"You are a professional translator. Translate the following {source_lang} text to {target_lang}. Maintain the original meaning and nuance as much as possible."
                prompt_tokens = count_tokens(system_msg, model) + count_tokens(
                    sub.content, model
                )

                # Update display with current subtitle
                display.update_header(sub.index, total_subs, total_tokens, total_cost)
                display.update_source(sub.content)
                display.update_progress(sub.index, total_subs)

                # Perform translation
                translated_content = await translate_text(
                    sub.content, source_lang, target_lang, model
                )

                # Update token counts and display
                completion_tokens = count_tokens(translated_content, model)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens = total_prompt_tokens + total_completion_tokens

                # Calculate total cost
                total_cost = display.calculate_cost(
                    total_prompt_tokens, total_completion_tokens
                )

                # Update all display components
                display.update_header(sub.index, total_subs, total_tokens, total_cost)
                display.update_target(translated_content)
                display.update_stats(
                    prompt_tokens,
                    completion_tokens,
                    total_prompt_tokens,
                    total_completion_tokens,
                )
                display.update_progress(sub.index, total_subs)

                translated_sub = srt.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=translated_content,
                )

                translated_subtitles.append(translated_sub)

        # Write output file and show summary after successful translation
        with console.status("[bold green]Writing output file..."):
            write_srt(output_file, translated_subtitles)

        # Show final summary
    except Exception as e:
        console.print(f"\n[bold red]Error during translation:[/bold red] {str(e)}")
        raise
    display_final_summary(
        total_subs,
        len(translated_subtitles),
        total_prompt_tokens,
        total_completion_tokens,
        model,
    )


def translate_srt_quiet(
    subtitles: List[srt.Subtitle],
    output_file: str,
    source_lang: str,
    target_lang: str,
    model: str,
) -> None:
    """Translate subtitles with simple progress bar instead of TUI."""
    translated_subtitles = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_subs = len(subtitles)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Translating subtitles...", total=total_subs)

        for sub in subtitles:
            # Count tokens for this subtitle
            system_msg = f"You are a professional translator. Translate the following {source_lang} text to {target_lang}. Maintain the original meaning and nuance as much as possible."
            prompt_tokens = count_tokens(system_msg, model) + count_tokens(
                sub.content, model
            )

            # Perform translation
            translated_content = translate_text(
                sub.content, source_lang, target_lang, model
            )

            # Update token counts
            completion_tokens = count_tokens(translated_content, model)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            translated_sub = srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=translated_content,
            )

            translated_subtitles.append(translated_sub)
            progress.advance(task)

    # Write output file
    write_srt(output_file, translated_subtitles)

    # Show final summary
    display_final_summary(
        total_subs,
        len(translated_subtitles),
        total_prompt_tokens,
        total_completion_tokens,
        model,
    )


if __name__ == "__main__":
    console.print("[bold blue]SRT Subtitle Translator[/bold blue]")
    console.print("Translates subtitles between languages using OPENAI\n")

    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles between languages"
    )
    parser.add_argument("input_file", help="Input SRT file path")
    parser.add_argument("output_file", help="Output SRT file path")
    parser.add_argument(
        "--from",
        dest="source_lang",
        default="English",
        help=f'Source language (default: English). Supported: {", ".join(SUPPORTED_LANGUAGES)}',
    )
    parser.add_argument(
        "--to",
        dest="target_lang",
        default="Japanese",
        help=f'Target language (default: English). Supported: {", ".join(SUPPORTED_LANGUAGES)}',
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help=f'OpenAI model to use (default: gpt-4o-mini). Supported: {", ".join(MODEL_PRICING.keys())}',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze token usage without performing translation",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Use simple progress bar instead of TUI"
    )

    args = parser.parse_args()

    try:
        translate_srt(
            args.input_file,
            args.output_file,
            args.source_lang,
            args.target_lang,
            args.model,
            args.dry_run,
            args.quiet,
        )
        console.print(
            "\n[bold green]✓[/bold green] Translation completed successfully!"
        )
        console.print(f"[dim]Output saved to:[/dim] {args.output_file}")
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        sys.exit(1)
