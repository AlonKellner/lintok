"""Simple single file test cases for the lintok file checking functionality."""

import lintok


def test_check_small_file() -> None:
    """Test that the check_file function passes with a small file."""
    # Assuming check_file takes a file path and returns True if the file is small enough
    result = lintok.check_files(["tests/sample_small_file.txt"])
    assert result is True, "Expected the file to be small enough"


def test_check_large_file() -> None:
    """Test that the check_file function fails with a large file."""
    # Test with a file that is too large
    result = lintok.check_files(
        ["tests/sample_large_file.txt"], override_config={"exclude": None}
    )
    assert result is False, "Expected the file to be too large"
