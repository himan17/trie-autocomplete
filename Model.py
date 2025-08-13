# Model.py
from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Tuple

from Utilities import LoadFile
from Trie import CompressedTrie, normalize_token

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _try_init_redis() -> Optional[object]:
    """
    Lazily create a Redis client if redis-py is available and env vars are set.
    Returns a client-like object implementing .get/.set or None.
    """
    host = os.getenv("REDIS_HOST")
    if not host:
        return None
    try:
        import redis  # type: ignore

        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        client = redis.Redis(host=host, port=port, db=db)
        # quick ping
        client.ping()
        logger.info("Connected to Redis at %s:%d db=%d", host, port, db)
        return client
    except Exception as exc:
        logger.warning("Redis unavailable (%s). Proceeding without cache.", exc)
        return None


class Model:
    """
    High-level model wrapping a CompressedTrie.

    Features:
    - Bulk construction from word list files (supports `word` or `word<TAB>score`)
    - Top-K autocomplete
    - Optional fuzzy search (Levenshtein) with small edit radii
    - Optional Redis caching (configured via REDIS_* env vars)
    - Prefix-only (default) or infix mode toggle
    """

    def __init__(self, cache: Optional[object] = None) -> None:
        self._cache = cache if cache is not None else _try_init_redis()
        self.trie = CompressedTrie(loader=None, cache=self._cache)
        self.prefix_only: bool = True

    # -----------------------
    # Build / load
    # -----------------------

    def construct(self, filename: str) -> int:
        """
        Populate the trie from a file.

        Supported line formats:
          - 'word'
          - 'word\\t<score>'
        Returns number of tokens added.
        """
        added = 0
        buf = LoadFile(filename)
        try:
            for raw in buf:
                line = raw.strip()
                if not line:
                    continue
                # Allow optional tab-separated score/frequency
                if "\t" in line:
                    word, score_s = line.split("\t", 1)
                    word = normalize_token(word)
                    try:
                        score = int(score_s)
                    except ValueError:
                        score = 1
                else:
                    word = normalize_token(line)
                    score = 1

                if not word:
                    continue

                self.trie.add(word, score=score)
                added += 1
        finally:
            try:
                buf.close()
            except Exception:
                pass

        logger.info("Constructed trie with %d entries from %s", added, filename)
        return added

    # -----------------------
    # Queries
    # -----------------------

    def list(self, query: str, k: int = 10, *, fuzzy: bool = False, max_edits: int = 1) -> List[str]:
        """
        Get suggestions for a query.

        - Prefix mode (default): top-K completions for the prefix.
        - Infix mode: returns words containing the substring (lightweight fallback).
        - Fuzzy mode: Levenshtein within max_edits; returns top-K by (score, distance).

        Returns a list of words (strings).
        """
        q = normalize_token(query)
        if not q:
            return []

        if fuzzy:
            hits = self.trie.fuzzy_search(q, max_edits=max_edits, k=k)
            # hits: List[(word, score, dist)]
            # Sort by higher score, then lower edit distance, then lexicographic
            hits.sort(key=lambda x: (-x[1], x[2], x[0]))
            return [w for (w, _s, _d) in hits[:k]]

        if self.prefix_only:
            pairs = self.trie.autocomplete(q, k=k)
            # pairs: List[(word, score)] already ranked by score then lexicographic
            return [w for (w, _s) in pairs]

        # Infix mode: since full infix trie indexing is heavy, do a pragmatic approach:
        # - Use a broad prefix (first char of query, or the whole query if short)
        # - Fallback to scanning autocomplete beneath that prefix
        # - Filter by substring containment
        # This keeps latency reasonable while still supporting infix.
        seed_prefix = q[0] if q else ""
        seed_candidates: List[Tuple[str, int]] = []
        if seed_prefix:
            seed_candidates = self.trie.autocomplete(seed_prefix, k=max(5 * k, 200))
        else:
            # degenerate case: no prefix; try the query directly
            seed_candidates = self.trie.autocomplete(q, k=max(5 * k, 200))

        filtered = [w for (w, _s) in seed_candidates if q in w]
        # If not enough, try fuzzy to fill
        if len(filtered) < k and len(q) >= 2:
            fuzz = self.trie.fuzzy_search(q, max_edits=1, k=max(k, 50))
            for (w, _s, _d) in fuzz:
                if q in w:
                    filtered.append(w)

        # Deduplicate preserving order
        seen = set()
        out: List[str] = []
        for w in filtered:
            if w not in seen:
                out.append(w)
                seen.add(w)
            if len(out) >= k:
                break
        return out

    def contains(self, word: str) -> bool:
        return self.trie.contains(word)

    # -----------------------
    # Controls
    # -----------------------

    def switch_command(self) -> None:
        """Toggle between prefix-only and infix modes."""
        self.prefix_only = not self.prefix_only
        logger.info("Search mode: %s", "PREFIX" if self.prefix_only else "INFIX")