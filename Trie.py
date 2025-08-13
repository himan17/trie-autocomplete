# Trie.py
from __future__ import annotations

import heapq
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Reasonable default logging; override in app if desired.
    logging.basicConfig(level=logging.INFO)

# -------------------------
# Public types / interfaces
# -------------------------

ChildSpec = Tuple[str, bool, int]
# (label, is_terminal, score)
# - label: edge label from parent to child (string, may be >1 char)
# - is_terminal: whether a word ends at this child
# - score: ranking score / frequency for the word that ends exactly at this child
#          (ignored if not terminal; can be 0/1 if unknown)


LoaderFn = Callable[[str], Iterable[ChildSpec]]
# Given the absolute path string from root (i.e., concatenation of labels),
# return children specs to materialize lazily. If no children, return empty iterable.


CacheLike = object
# Any object implementing .get(key: str) -> Optional[str] and .set(key: str, value: str, ex: Optional[int]) -> None
# (e.g., redis-py client). We use JSON-encoded payloads (UTF-8 strings).


# -------------------------
# Compressed Trie Node
# -------------------------

@dataclass
class CompressedTrieNode:
    """
    A node in a *compressed* trie (radix tree).
    Each outgoing edge is labeled by a *string* (not a single character).
    """
    label: str = ""  # Edge label from parent to this node
    is_terminal: bool = False
    score: int = 1  # Ranking score for terminal words; default 1 if unknown
    children: Dict[str, "CompressedTrieNode"] = field(default_factory=dict)
    path: str = ""  # Absolute path (word prefix) from root to this node
    _loaded: bool = False  # For lazy loading

    # Optional loader to populate children on demand (attached on the root and inherited)
    _loader: Optional[LoaderFn] = None

    def attach_loader(self, loader: Optional[LoaderFn]) -> None:
        """Attach a loader to the subtree rooted here (propagated as needed)."""
        self._loader = loader

    # --- Lazy loading helpers ---

    def _ensure_loaded(self) -> None:
        """If a loader exists and this node hasn't been expanded, load children."""
        if self._loader is None or self._loaded:
            return
        try:
            specs = list(self._loader(self.path))
            for lbl, is_term, sc in specs:
                if not lbl:
                    continue
                first = lbl[0]
                child = CompressedTrieNode(
                    label=lbl,
                    is_terminal=is_term,
                    score=sc if is_term else 1,
                    path=self.path + lbl,
                )
                child.attach_loader(self._loader)
                self.children[first] = child
            logger.debug("Lazy-loaded %d children at path='%s'", len(specs), self.path)
        except Exception as exc:
            logger.exception("Loader failed at path='%s': %s", self.path, exc)
        finally:
            self._loaded = True

    # --- Mutation ---

    def insert(self, word: str, score: int = 1) -> None:
        """
        Insert a word with optional score (frequency). Supports apostrophes and [a-z].
        Path compression is performed as needed.
        """
        p = self
        i = 0
        while i < len(word):
            p._ensure_loaded()
            ch = word[i]
            nxt = p.children.get(ch)
            if nxt is None:
                # No edge starting with ch → create new compressed edge with the rest of the word
                new_node = CompressedTrieNode(
                    label=word[i:],
                    is_terminal=True,
                    score=score,
                    path=p.path + word[i:],
                )
                new_node.attach_loader(p._loader)
                p.children[word[i]] = new_node
                return

            # There is an edge; match as far as possible on its label
            lbl = nxt.label
            j = 0
            while i + j < len(word) and j < len(lbl) and word[i + j] == lbl[j]:
                j += 1

            if j == len(lbl):
                # Edge label fully matched → move down
                p = nxt
                i += j
                continue

            # Partial match within the edge label → split node
            # Common prefix lbl[:j], remainder lbl[j:], new word remainder word[i+j:]
            common = lbl[:j]
            remainder_existing = lbl[j:]
            remainder_new = word[i + j:]

            # Create an intermediate node for the common part
            mid = CompressedTrieNode(
                label=common,
                is_terminal=False,
                score=1,
                path=p.path + common,
            )
            mid.attach_loader(p._loader)

            # Reattach existing child under mid
            nxt.label = remainder_existing
            nxt.path = mid.path + remainder_existing
            mid.children[remainder_existing[0]] = nxt

            # Attach new branch for the new word remainder
            if remainder_new:
                new_leaf = CompressedTrieNode(
                    label=remainder_new,
                    is_terminal=True,
                    score=score,
                    path=mid.path + remainder_new,
                )
                new_leaf.attach_loader(p._loader)
                mid.children[remainder_new[0]] = new_leaf
            else:
                # New word ends at the split point
                mid.is_terminal = True
                mid.score = score

            # Replace p → mid
            p.children[common[0]] = mid
            return

        # If we exit the loop, full word matched an existing path
        p.is_terminal = True
        p.score = max(p.score, score)

    # --- Queries ---

    def contains(self, word: str) -> bool:
        """Return True if the exact word exists."""
        node, matched = self._descend(word)
        return matched == len(word) and node is not None and node.is_terminal

    def autocomplete(self, prefix: str, k: int = 10) -> List[Tuple[str, int]]:
        """
        Return top-k (word, score) completions for a given prefix.
        Uses a bounded min-heap (k) and DFS; compatible with lazy loading.
        """
        if k <= 0:
            return []

        cache_key = None
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache_key = f"ac:{k}:{prefix}"
            try:
                cached = cache.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                logger.debug("Cache get failed for key=%s", cache_key)

        start_node, matched = self._descend(prefix)
        if start_node is None:
            return []

        # If prefix ends inside a label, pretend there's a virtual node here.
        # We still need to search below start_node with the remaining label suffix.
        results: List[Tuple[int, str]] = []  # min-heap of (score, word)
        self._collect_top_k(start_node, k, results)
        out = [(w, s) for s, w in sorted(results, key=lambda x: (-x[0], x[1]))]

        if cache_key and cache is not None:
            try:
                cache.set(cache_key, json.dumps(out), ex=300)  # 5 min TTL
            except Exception:
                logger.debug("Cache set failed for key=%s", cache_key)
        return out

    def fuzzy_search(
        self,
        query: str,
        max_edits: int = 1,
        k: int = 10,
        band: Optional[int] = None,
    ) -> List[Tuple[str, int, int]]:
        """
        Levenshtein-based fuzzy search within max_edits (default 1),
        returns top-k tuples: (word, score, edit_distance).
        Uses Ukkonen-style DP on a trie with pruning and optional banding.

        band: if provided, limits DP comparison to +/- band around diagonal.
        For short queries, band=1..2 is usually enough; if None, we use max_edits+1.
        """
        if k <= 0:
            return []

        cache_key = None
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache_key = f"fz:{k}:{max_edits}:{band}:{query}"
            try:
                cached = cache.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                logger.debug("Cache get failed for key=%s", cache_key)

        if band is None:
            band = max(1, max_edits + 1)

        # Initial DP row for empty prefix against query
        prev_row = list(range(len(query) + 1))

        heap: List[Tuple[int, int, str]] = []  # min-heap of (neg_score, dist, word)
        self._fuzzy_dfs(self, query, prev_row, max_edits, band, heap, k)

        out = [(w, -neg_s, d) for (neg_s, d, w) in sorted(heap)]
        if cache_key and cache is not None:
            try:
                cache.set(cache_key, json.dumps(out), ex=300)
            except Exception:
                logger.debug("Cache set failed for key=%s", cache_key)
        return out

    # --- Internal helpers ---

    def _descend(self, s: str) -> Tuple[Optional["CompressedTrieNode"], int]:
        """
        Descend the trie matching as much of s as possible.
        Returns (node_reached, matched_length_in_s). node_reached may be inside a label.
        """
        node: Optional[CompressedTrieNode] = self
        i = 0
        while node is not None and i < len(s):
            node._ensure_loaded()
            nxt = node.children.get(s[i])
            if nxt is None:
                return None, i
            lbl = nxt.label
            j = 0
            while i + j < len(s) and j < len(lbl) and s[i + j] == lbl[j]:
                j += 1
            i += j
            if j == len(lbl):
                node = nxt
            else:
                # Prefix stops inside this edge label
                # We treat nxt as the node, even though the match is partial.
                return nxt, i
        return node, i

    def _collect_top_k(
        self,
        node: "CompressedTrieNode",
        k: int,
        heap: List[Tuple[int, str]],
        carried_prefix: str = "",
    ) -> None:
        """
        DFS from node to collect top-k by score. Uses a bounded min-heap: (score, word)
        """
        node._ensure_loaded()
        # Compute the absolute word for this node
        # carried_prefix is used when starting from a node whose label was partially matched.
        curr_prefix = carried_prefix + node.label

        if node.is_terminal:
            word = node.path  # absolute path holds the full word
            self._heap_push_top_k(heap, k, (node.score, word))

        for child in node.children.values():
            self._collect_top_k(child, k, heap, carried_prefix=curr_prefix)

    @staticmethod
    def _heap_push_top_k(heap: List[Tuple[int, str]], k: int, item: Tuple[int, str]) -> None:
        """
        Maintain a bounded min-heap of size at most k by score, then lexicographically.
        """
        if len(heap) < k:
            heapq.heappush(heap, item)
        else:
            if item > heap[0]:
                heapq.heapreplace(heap, item)

    def _fuzzy_dfs(
        self,
        node: "CompressedTrieNode",
        query: str,
        prev_row: Sequence[int],
        max_edits: int,
        band: int,
        heap: List[Tuple[int, int, str]],
        k: int,
    ) -> None:
        """
        Depth-first traversal computing DP rows along compressed labels with pruning.
        """
        node._ensure_loaded()

        # Walk the DP over this node's label
        curr_rows = [prev_row]
        for ch in node.label:
            prev = curr_rows[-1]
            m = len(query)
            # Banded computation around diagonal ±band
            new = [prev[0] + 1]
            # left, diag, up as we fill
            for j in range(1, m + 1):
                if abs((len(new) - 1) - len(curr_rows)) > band:
                    # Outside band; approximate with a large value
                    new.append(max(prev[j], new[j - 1]) + 1)
                    continue
                cost = 0 if ch == query[j - 1] else 1
                insert_cost = new[j - 1] + 1
                delete_cost = prev[j] + 1
                replace_cost = prev[j - 1] + cost
                new.append(min(insert_cost, delete_cost, replace_cost))
            curr_rows.append(new)

            # Early prune: if the best value in the new row already exceeds max_edits + band buffer
            if min(new) > max_edits + band:
                return

        last_row = curr_rows[-1]

        # If terminal and within allowed edits, record candidate
        if node.is_terminal and last_row[-1] <= max_edits:
            word = node.path
            # Prefer higher score, then lower edit distance
            item = (-node.score, last_row[-1], word)
            if len(heap) < k:
                heapq.heappush(heap, item)
            else:
                if item < heap[0]:
                    heapq.heapreplace(heap, item)

        # If any cell within max_edits (plus a small buffer), continue to children
        threshold = max_edits + band
        if min(last_row) <= threshold:
            for child in node.children.values():
                self._fuzzy_dfs(child, query, last_row, max_edits, band, heap, k)


# -------------------------
# Compressed Trie wrapper
# -------------------------

class CompressedTrie:
    """
    High-level API around CompressedTrieNode with optional lazy loading and cache.
    """

    def __init__(
        self,
        loader: Optional[LoaderFn] = None,
        cache: Optional[CacheLike] = None,
    ) -> None:
        """
        loader: function(path) -> Iterable[(label, is_terminal, score)] to lazily load children
        cache: object with .get(key)->str|None and .set(key, value, ex=ttl_seconds)->None
        """
        self.root = CompressedTrieNode(path="")
        self.root.attach_loader(loader)
        # Expose cache to nodes
        setattr(self.root, "_cache", cache)

    # Delegate core methods

    def add(self, word: str, score: int = 1) -> None:
        word = normalize_token(word)
        if not word:
            return
        self.root.insert(word, score)

    def bulk_add(self, words: Iterable[Tuple[str, int]]) -> None:
        """
        Insert many words: iterable of (word, score).
        """
        for w, sc in words:
            self.add(w, sc)

    def contains(self, word: str) -> bool:
        return self.root.contains(normalize_token(word))

    def autocomplete(self, prefix: str, k: int = 10) -> List[Tuple[str, int]]:
        return self.root.autocomplete(normalize_token(prefix), k=k)

    def fuzzy_search(
        self, query: str, max_edits: int = 1, k: int = 10, band: Optional[int] = None
    ) -> List[Tuple[str, int, int]]:
        return self.root.fuzzy_search(normalize_token(query), max_edits=max_edits, k=k, band=band)


# -------------------------
# Utilities
# -------------------------

def normalize_token(s: str) -> str:
    """
    Normalize tokens to the supported alphabet: [a-z'].
    Lowercases and strips disallowed characters.
    """
    if not s:
        return ""
    s = s.strip().lower()
    out_chars: List[str] = []
    for ch in s:
        if ch == "'" or ("a" <= ch <= "z"):
            out_chars.append(ch)
    return "".join(out_chars)