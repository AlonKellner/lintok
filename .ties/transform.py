"""Transform functions for `ties`."""

import yaml


class IndentedDumper(yaml.Dumper):
    """Custom YAML Dumper that indents lists to match the yamlfmt standard."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:  # noqa: ARG002
        """Increase indent."""
        # The key is to call the superclass's method with indentless=False.
        # This ensures that list items are indented.
        return super().increase_indent(flow, False)


def trivy_yaml(gitignore: str) -> str:
    """Transform for a `.gitignore` -> `trivy.toml` tie."""
    lines = [line.strip() for line in gitignore.split("\n")]
    lines = [line for line in lines if (not line.startswith("#")) and (len(line) > 0)]
    lines = [f"**/{line}" for line in lines]
    file_lines = [line for line in lines if not line.endswith("/")]
    dir_lines = [line for line in lines if line.endswith("/")]
    trivy_config = {
        "fs": {
            "skip-dirs": sorted(dir_lines),
            "skip-files": sorted(file_lines),
        }
    }
    return yaml.dump(
        trivy_config,
        Dumper=IndentedDumper,
        default_flow_style=False,
    )
