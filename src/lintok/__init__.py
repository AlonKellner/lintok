"""Lintok is a linter for file sizes in tokens, for smaller files and faster agent edits."""

from ._file_checking import check_files

__all__ = [
    "check_files",
]
