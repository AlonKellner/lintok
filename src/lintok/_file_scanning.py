import contextlib
import mimetypes
from pathlib import Path
from typing import Any

import pathspec


def _gather_files(paths: list[str]) -> set[Path]:
    """Collect all files from the given paths."""
    all_files: set[Path] = set()
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        if path.is_dir():
            all_files.update(p.resolve() for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            all_files.add(path.resolve())
    return all_files


def _apply_gitignore(
    files: set[Path], config: dict[str, Any], project_root: Path | None
) -> set[Path]:
    """Remove files ignored by .gitignore if configured."""
    if not (config.get("honor_gitignore") and project_root):
        return files
    gitignore_file = project_root / ".gitignore"
    if not gitignore_file.is_file():
        return files
    with gitignore_file.open("r") as f:
        spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
    files_to_ignore = set()
    for file in files:
        try:
            relative_path = file.relative_to(project_root)
            if spec.match_file(relative_path):
                files_to_ignore.add(file)
        except ValueError:
            pass
    return files - files_to_ignore


def _apply_manual_exclude(
    files: set[Path], exclude_patterns: list[str], project_root: Path | None
) -> set[Path]:
    """Remove files matching manual exclude patterns."""
    if not exclude_patterns:
        return files
    manually_excluded_files: set[Path] = set()
    for file_path in files:
        relative_path_str = None
        if project_root:
            with contextlib.suppress(ValueError):
                relative_path_str = str(file_path.relative_to(project_root))
        for pattern in exclude_patterns:
            if file_path.match(pattern) or (
                relative_path_str and Path(relative_path_str).match(pattern)
            ):
                manually_excluded_files.add(file_path)
                break
    return files - manually_excluded_files


def get_files_to_check(
    paths: list[str], config: dict[str, Any], project_root: Path | None
) -> set[Path]:
    """Gather all files from paths, handling directories, and apply exclusion patterns."""
    all_files = _gather_files(paths)
    all_files = _apply_gitignore(all_files, config, project_root)
    exclude_patterns = config.get("exclude", [])
    all_files = _apply_manual_exclude(all_files, exclude_patterns, project_root)
    return all_files


def is_text_file(file_path: Path) -> bool:
    """Check if a file is likely a text file using the mimetypes module."""
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type:
        return mime_type.startswith("text/")

    try:
        with open(file_path, encoding="utf-8") as f:
            f.read(1024)
        return True
    except (OSError, UnicodeDecodeError):
        return False
