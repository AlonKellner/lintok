#!/usr/bin/env python3

"""lintok: A compact, colorful linter for checking file size metrics."""

import argparse
import contextlib
import importlib
import mimetypes
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pathspec

# --- Required Dependencies ---
import toml

# --- Rich for colorful output ---
from rich.console import Console
from rich.table import Table
from rich.text import Text

# --- Configuration ---
DEFAULT_CONFIG = {
    "max_lines": None,
    "max_chars": None,
    "max_words": None,
    "max_kb": None,
    "max_tokens": None,  # Global default for tokenizers
    # Default tokenizer provides an out-of-the-box check.
    # This list is REPLACED by user config, not merged.
    "tokenizers": [
        # This now relies on the global max_tokens if set
        {"type": "tiktoken", "model": "cl100k_base"},
    ],
    "exclude": ["*.pyc", "*.log", ".git/", ".idea/", "__pycache__/"],
    "honor_gitignore": True,
}

# --- Caches ---
_hf_tokenizer_cache = {}
_tiktoken_cache = {}
_anthropic_client = None


def find_and_load_config(start_path: Path) -> tuple[dict[str, Any], Path | None]:
    """Find and load config from pyproject.toml, searching upwards."""
    config = DEFAULT_CONFIG.copy()
    project_root = None

    current_path = start_path.resolve()

    for path in [current_path, *list(current_path.parents)]:
        pyproject_path = path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                pyproject_data = toml.load(pyproject_path)
                if "tool" in pyproject_data and "lintok" in pyproject_data["tool"]:
                    project_root = path
                    user_config = pyproject_data["tool"]["lintok"]
                    # If user defines tokenizers, it replaces the default list.
                    if "tokenizers" in user_config:
                        config["tokenizers"] = user_config.pop("tokenizers")

                    for key, value in user_config.items():
                        config[key] = value
                    break
            except Exception as e:
                print(
                    f"[bold red]Warning:[/bold red] Could not parse {pyproject_path}: {e}",
                    file=sys.stderr,
                )

    return config, project_root


def get_files_to_check(
    paths: list[str], config: dict[str, Any], project_root: Path | None
) -> set[Path]:
    """Gather all files from paths, handling directories, and apply exclusion patterns."""
    all_files: set[Path] = set()
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        if path.is_dir():
            all_files.update(p.resolve() for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            all_files.add(path.resolve())

    # --- Apply .gitignore rules ---
    if config.get("honor_gitignore") and project_root:
        gitignore_file = project_root / ".gitignore"
        if gitignore_file.is_file():
            with gitignore_file.open("r") as f:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", f)

            files_to_ignore_by_gitignore = set()
            for file in all_files:
                try:
                    # If relative_to succeeds, the file is in the project.
                    relative_path = file.relative_to(project_root)
                    if spec.match_file(relative_path):
                        files_to_ignore_by_gitignore.add(file)
                except ValueError:
                    # This file is outside the project root, so gitignore doesn't apply.
                    pass
            all_files -= files_to_ignore_by_gitignore

    # --- Apply manual exclude patterns ---
    exclude_patterns = config.get("exclude", [])
    if not exclude_patterns:
        return all_files

    manually_excluded_files: set[Path] = set()
    for file_path in all_files:
        # Get relative path safely
        relative_path_str = None
        if project_root:
            with contextlib.suppress(ValueError):
                relative_path_str = str(file_path.relative_to(project_root))

        for pattern in exclude_patterns:
            # Re-implement original logic safely
            should_exclude = False
            # Match against filename glob (e.g. "*.log")
            if file_path.match(pattern) or (
                relative_path_str and Path(relative_path_str).match(pattern)
            ):
                should_exclude = True

            if should_exclude:
                manually_excluded_files.add(file_path)
                break

    return all_files - manually_excluded_files


def is_text_file(file_path: Path) -> bool:
    """Check if a file is likely a text file using the mimetypes module."""
    mime_type, encoding = mimetypes.guess_type(file_path)

    if mime_type:
        return mime_type.startswith("text/")

    try:
        with open(file_path, encoding="utf-8") as f:
            f.read(1024)
        return True
    except (OSError, UnicodeDecodeError):
        return False


def _get_huggingface_tokenizer(config: dict[str, Any]) -> Callable[[str], int] | None:
    """Lazy load and return a Hugging Face tokenizer function."""
    try:
        transformers = importlib.import_module("transformers")
        path = config.get("path")
        if not path:
            return None

        tokenizer = _hf_tokenizer_cache.setdefault(
            path, transformers.AutoTokenizer.from_pretrained(path)
        )
        return lambda text: len(tokenizer.encode(text))
    except ImportError:
        print(
            "[bold red]Error:[/bold red] 'transformers' is required for this tokenizer. Run 'pip install transformers'."
        )
        return None
    except Exception as e:
        print(
            f"[bold red]Error loading Hugging Face tokenizer '{config.get('path')}': {e}[/bold red]"
        )
        return None


def _get_tiktoken_tokenizer(config: dict[str, Any]) -> Callable[[str], int] | None:
    """Lazy load and return a tiktoken tokenizer function."""
    try:
        tiktoken = importlib.import_module("tiktoken")
        model = config.get("model")
        if not model:
            return None

        encoding = _tiktoken_cache.setdefault(model, tiktoken.get_encoding(model))
        return lambda text: len(encoding.encode(text))
    except ImportError:
        print(
            "[bold red]Error:[/bold red] 'tiktoken' is required for this tokenizer. Run 'pip install tiktoken'."
        )
        return None
    except Exception as e:
        print(
            f"[bold red]Error loading tiktoken model '{config.get('model')}': {e}[/bold red]"
        )
        return None


def _get_anthropic_tokenizer(config: dict[str, Any]) -> Callable[[str], int] | None:
    """Lazy load and return the Anthropic tokenizer function."""
    global _anthropic_client  # noqa: PLW0603
    model = config.get("model")
    if not model:
        print(
            "[bold red]Error:[/bold red] 'model' is required for the anthropic tokenizer."
        )
        return None

    if _anthropic_client is None:
        try:
            anthropic = importlib.import_module("anthropic")
            _anthropic_client = anthropic.Anthropic()
        except ImportError:
            print(
                "[bold red]Error:[/bold red] 'anthropic' is required for this tokenizer. Run 'pip install anthropic'."
            )
            return None
        except Exception as e:
            print(f"[bold red]Error initializing Anthropic client: {e}[/bold red]")
            return None

    # Return a function that calls the correct method with the required structure
    def count_tokens(text: str) -> int:
        return _anthropic_client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}]
        ).input_tokens

    return count_tokens


def get_tokenizer(config: dict[str, Any]) -> Callable[[str], int] | None:
    """Load a tokenizer function based on its configuration."""
    tokenizer_type = config.get("type", "huggingface")
    if tokenizer_type == "huggingface":
        return _get_huggingface_tokenizer(config)
    elif tokenizer_type == "tiktoken":
        return _get_tiktoken_tokenizer(config)
    elif tokenizer_type == "anthropic":
        return _get_anthropic_tokenizer(config)
    return None


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
        title=f"[cyan]{file_path.name}[/cyan]",
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


def main() -> None:
    """Run the main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="lintok: A compact, colorful linter for checking file size metrics."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Paths to check (files or directories). Defaults to current directory.",
    )
    parser.add_argument(
        "--no-gitignore",
        dest="honor_gitignore",
        action="store_false",
        help="Do not honor .gitignore files.",
    )
    parser.set_defaults(honor_gitignore=True)

    args = parser.parse_args()
    console = Console()

    config, project_root = find_and_load_config(Path.cwd())

    if not args.honor_gitignore:
        config["honor_gitignore"] = False

    files_to_process = get_files_to_check(args.paths, config, project_root)
    text_files = {f for f in files_to_process if is_text_file(f)}

    if not text_files:
        console.print("[bold yellow]No text files found to check.[/bold yellow]")
        sys.exit(0)

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
        sys.exit(1)
    else:
        console.print("\n--- Summary: [bold green]All files passed[/bold green]. ---")
        sys.exit(0)


if __name__ == "__main__":
    main()
