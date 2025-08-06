import sys
from pathlib import Path
from typing import Any

import toml

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
