import importlib
from collections.abc import Callable
from typing import Any

# --- Caches ---
_hf_tokenizer_cache = {}
_tiktoken_cache = {}
_anthropic_client = None


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
