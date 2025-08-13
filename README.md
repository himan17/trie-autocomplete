# Intelligent Auto-Complete (Python, Compressed Trie)

High-performance autocomplete with:
- **Compressed trie** (path compression) for reduced memory usage and better cache locality
- **Top-K** retrieval via a bounded min-heap
- **Fuzzy search** (banded Levenshtein) for typo tolerance
- Optional **Redis** cache for hot prefixes/results
- Simple **Tkinter UI** for quick testing

## Quick Start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python Controller.py -f words_598153.txt -k 10 --fuzzy --max-edits 1
```

> Use WORDS_FILE, TOP_K, FUZZY, and MAX_EDITS environment variables to set defaults.

### **File Format**

Word list files can contain either:

```
word
```

or

```
word<TAB>score
```

where score is an integer frequency or ranking weight.

## **Example Usage (Library)**

```
from Trie import CompressedTrie
t = CompressedTrie()
t.add("apple", score=42)
t.add("application", score=11)

print(t.autocomplete("app", k=5))
# [('apple', 42), ('application', 11)]

print(t.fuzzy_search("applr", max_edits=1, k=5))
# [('apple', 42, 1)]
```

## **Optional Redis Cache**

Set REDIS_HOST, REDIS_PORT (default 6379), and REDIS_DB (default 0).

If Redis is not reachable, the system runs without caching.

```
docker compose up -d redis
REDIS_HOST=localhost python Controller.py -f words_598153.txt
```

## **Project Structure**

```
.
├── Controller.py     # Wires Model + View (Tkinter UI)
├── Model.py          # High-level API around CompressedTrie
├── Trie.py           # Compressed trie with top-K, fuzzy, lazy-load hooks
├── View.py           # Tkinter UI
├── Utilities.py      # CenterWindow, LoadFile
├── words_*.txt       # Datasets
├── __init__.py
├── README.md
└── requirements.txt
```

## **Notes**

-   Fuzzy search is optimized for small edit distances (0–2).
    
-   The trie supports a **lazy loader** callback if you later shard data on disk; see Trie.py docstrings.