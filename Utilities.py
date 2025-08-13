# Utilities.py
from __future__ import annotations

import logging
import os
from typing import Generator, Tuple, Union

logger = logging.getLogger(__name__)


def CenterWindow(parent, width: int = 600, height: int = 80) -> None:
    """
    Center the given Tkinter/Tk window on the current screen.

    Args:
        parent: A Tkinter/Tk window instance.
        width: Desired window width in pixels.
        height: Desired window height in pixels.
    """
    try:
        sw = parent.winfo_screenwidth()
        sh = parent.winfo_screenheight()
    except Exception as exc:
        logger.warning("Could not get screen size: %s", exc)
        return

    x = max(0, int((sw - width) / 2))
    y = max(0, int((sh - height) / 2))
    parent.geometry(f"{width}x{height}+{x}+{y}")


def LoadFile(
    filename: str,
    *,
    with_scores: bool = False,
    encoding: str = "utf-8",
) -> Union[Generator[str, None, None], Generator[Tuple[str, int], None, None]]:
    """
    Lazily read a file line by line.

    Args:
        filename: Path to the file to read.
        with_scores: If True, yields (word, score) tuples for tab-separated lines.
                     If a score is missing or invalid, defaults to 1.
        encoding: File encoding (default UTF-8 with 'replace' for errors).

    Yields:
        str: Each stripped line if with_scores=False.
        (str, int): (word, score) tuples if with_scores=True.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    logger.debug("Loading file: %s", filename)

    try:
        with open(filename, "r", encoding=encoding, errors="replace") as buf:
            for raw_line in buf:
                line = raw_line.strip()
                if not line:
                    continue

                if with_scores:
                    if "\t" in line:
                        word, score_s = line.split("\t", 1)
                        try:
                            score = int(score_s)
                        except ValueError:
                            score = 1
                        yield word, score
                    else:
                        yield line, 1
                else:
                    yield line
    except Exception as exc:
        logger.exception("Error loading file %s: %s", filename, exc)
        raise