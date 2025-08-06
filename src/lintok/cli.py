"""Lintok CLI entry point for checking file size metrics."""

import argparse
import sys

from ._file_checking import check_files


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
    if check_files(args.paths, honor_gitignore=args.honor_gitignore):
        sys.exit(0)
    else:
        sys.exit(1)
