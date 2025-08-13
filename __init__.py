# __init__.py
from __future__ import annotations

__all__ = [
    "CompressedTrie",
    "normalize_token",
    "Model",
]

__version__ = "0.1.0"

from .Trie import CompressedTrie, normalize_token  # noqa: E402
from .Model import Model  # noqa: E402