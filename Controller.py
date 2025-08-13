# Controller.py
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

from Utilities import LoadFile  # kept for compatibility with your View
from Model import Model

# Tkinter import (Python 3) with a Python 2 fallback just in case
try:
    import tkinter as tk
except Exception:  # pragma: no cover
    try:
        import Tkinter as tk  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tkinter/Tkinter is not available") from exc


DEFAULT_FILE = os.getenv("WORDS_FILE", "words_598153.txt")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "10"))
DEFAULT_FUZZY = os.getenv("FUZZY", "0") in {"1", "true", "True"}
DEFAULT_MAX_EDITS = int(os.getenv("MAX_EDITS", "1"))

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class Controller:
    """
    Wires up the Model and the View.
    Keeps legacy method names (Construct/List/Contains/SwitchCommand) so
    your existing View continues to work.
    """

    def __init__(self, filename: str, *, top_k: int = DEFAULT_TOP_K, fuzzy: bool = DEFAULT_FUZZY,
                 max_edits: int = DEFAULT_MAX_EDITS) -> None:
        self.filename = filename
        self.model = Model()
        self.top_k = top_k
        self.fuzzy = fuzzy
        self.max_edits = max_edits

        self.window = tk.Tk()
        self.window.title("Intelligent Auto-Complete")
        # Lazy import to avoid circulars if View imports Controller
        from View import View  # noqa: WPS433

        self.view = View(self.window, self)

    # -----------------------
    # App lifecycle
    # -----------------------

    def run(self) -> None:
        self.window.mainloop()

    # -----------------------
    # Legacy API (preserved)
    # -----------------------

    def Construct(self) -> int:
        """Legacy wrapper → build the trie from the file."""
        logger.info("Building trie from %s ...", self.filename)
        return self.model.construct(self.filename)

    def Contains(self, tag: str) -> bool:
        """Legacy wrapper → exact match check."""
        return self.model.contains(tag)

    def List(self, tag: str) -> List[str]:
        """
        Legacy wrapper → suggestions.
        Uses controller-level defaults for top-k and fuzzy settings.
        """
        return self.model.list(tag, k=self.top_k, fuzzy=self.fuzzy, max_edits=self.max_edits)

    def SwitchCommand(self) -> None:
        """Legacy wrapper → toggle prefix vs infix mode."""
        self.model.switch_command()

    def LoadFile(self) -> List[str]:
        """Legacy wrapper → iterate file as list (used by some Views)."""
        return list(LoadFile(self.filename))

    # -----------------------
    # Modern API (optional)
    # -----------------------

    def construct(self) -> int:
        return self.Construct()

    def contains(self, tag: str) -> bool:
        return self.Contains(tag)

    def suggestions(self, tag: str, *, k: int | None = None, fuzzy: bool | None = None,
                    max_edits: int | None = None) -> List[str]:
        return self.model.list(
            tag,
            k=k if k is not None else self.top_k,
            fuzzy=self.fuzzy if fuzzy is None else fuzzy,
            max_edits=self.max_edits if max_edits is None else max_edits,
        )

    def switch_command(self) -> None:
        self.SwitchCommand()


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intelligent Auto-Complete Controller")
    p.add_argument("-f", "--file", default=DEFAULT_FILE, help="Word list file")
    p.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K suggestions")
    p.add_argument("--fuzzy", action="store_true", default=DEFAULT_FUZZY, help="Enable fuzzy search")
    p.add_argument("--max-edits", type=int, default=DEFAULT_MAX_EDITS, help="Max edits for fuzzy")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    app = Controller(
        args.file,
        top_k=args.top_k,
        fuzzy=args.fuzzy,
        max_edits=args.max_edits,
    )
    app.run()