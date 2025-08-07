from pathlib import Path
from typing import Any

# --- Rich for colorful output ---
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ._configuration import find_and_load_config
from ._file_scanning import get_files_to_check, is_text_file
from ._tokenization import get_tokenizer


def check_file(file_path: Path, config: dict[str, Any], console: Console) -> bool:
    """Check a single file and print a rich table with results. Return True if any check failed."""
    try:
        content = file_path.read_text(encoding="utf-8")
        file_size_kb = file_path.stat().st_size / 1024
    except Exception as e:
        console.print(f"[bold red]Error reading {file_path}: {e}[/bold red]")
        return True

    metrics = {
        "Lines": (len(content.splitlines()), config.get("max_lines")),
        "Chars": (len(content), config.get("max_chars")),
        "Words": (len(content.split()), config.get("max_words")),
        "KB": (round(file_size_kb, 2), config.get("max_kb")),
    }

    any_failed = False

    table = Table(
        show_header=True,
        header_style="bold magenta",
        title=f"[cyan]{file_path.absolute()}[/cyan]",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status", justify="center")

    for name, (value, threshold) in metrics.items():
        if threshold is None:
            status = Text("SKIP", style="yellow")
            table.add_row(name, f"{value:,}", "-", status)
        else:
            failed = value > threshold
            if failed:
                any_failed = True
            status = (
                Text("FAIL", style="bold red")
                if failed
                else Text("PASS", style="bold green")
            )
            table.add_row(name, f"{value:,}", f"{threshold:,}", status)

    for tok_conf in config.get("tokenizers", []):
        threshold = tok_conf.get("max_tokens") or config.get("max_tokens")
        name = tok_conf.get("path") or tok_conf.get("model") or tok_conf.get("type")
        tokenizer_func = get_tokenizer(tok_conf)

        if not tokenizer_func:
            status = Text("ERROR", style="bold red")
            table.add_row(
                f"Tokens ({name})", "-", f"{threshold:,}" if threshold else "-", status
            )
            any_failed = True
            continue

        if not threshold:
            status = Text("SKIP", style="yellow")
            table.add_row(f"Tokens ({name})", "-", "-", status)
            continue

        try:
            token_count = tokenizer_func(content)
            failed = token_count > threshold
            if failed:
                any_failed = True
            status = (
                Text("FAIL", style="bold red")
                if failed
                else Text("PASS", style="bold green")
            )
            table.add_row(
                f"Tokens ({name})", f"{token_count:,}", f"{threshold:,}", status
            )
        except Exception as e:
            status = Text("ERROR", style="bold red")
            table.add_row(
                f"Tokens ({name})", f"API Error: {e}", f"{threshold:,}", status
            )
            any_failed = True

    if any_failed:
        console.print(table)

    return any_failed


def check_files(
    paths: list[Path | str],
    honor_gitignore: bool = True,
    override_config: dict[str, Any] | None = None,
) -> bool:
    """Check files in the given paths, optionally honoring .gitignore."""
    console = Console()

    config, project_root = find_and_load_config(Path.cwd())
    if override_config:
        for key, value in override_config.items():
            config[key] = value

    if not honor_gitignore:
        config["honor_gitignore"] = False

    paths = [Path(p) for p in paths]

    for path in paths:
        if not path.exists():
            console.print(f"[bold red]Path does not exist: {path}[/bold red]")
            raise ValueError(f"Path does not exist: {path}")

    files_to_process = get_files_to_check(paths, config, project_root)
    text_files = {f for f in files_to_process if is_text_file(f)}

    if not text_files:
        console.print("[bold yellow]No text files found to check.[/bold yellow]")
        return True

    console.print(
        f"--- [bold]lintok[/bold]: Checking {len(text_files)} text file(s) ---"
    )

    failed_files = 0
    for file_path in sorted(list(text_files)):
        if check_file(file_path, config, console):
            failed_files += 1

    if failed_files > 0:
        console.print(
            f"\n--- Summary: [bold red]{failed_files} file(s) failed[/bold red]. ---"
        )
        return False
    else:
        console.print("\n--- Summary: [bold green]All files passed[/bold green]. ---")
        return True
