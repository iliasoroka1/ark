"""Text chunking and tokenization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from semantic_text_splitter import MarkdownSplitter, TextSplitter

_WORD_RE = re.compile(r"\w+")

# Common English stopwords that add noise to BM25 without improving recall
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "about", "up", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "my", "our", "your", "his", "her", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
})

_MAX_DIMS = 4096
_EMB_ONE: list[str] = [f"embedding_one_{i}" for i in range(_MAX_DIMS)]
_EMB_ZERO: list[str] = [f"embedding_zero_{i}" for i in range(_MAX_DIMS)]


def binarize_embedding(embedding: list[float]) -> list[str]:
    one, zero = _EMB_ONE, _EMB_ZERO
    return [one[i] if v > 0.0 else zero[i] for i, v in enumerate(embedding)]


def _stem(word: str) -> str | None:
    """Simple English suffix stemmer. Returns stem if different from input, else None."""
    if len(word) <= 3:
        return None
    # Order matters: try longer suffixes first
    for suffix, min_stem in [
        ("ies", 3), ("ied", 3),  # carries→carri, tried→tri
        ("ing", 3), ("tion", 3), ("ment", 3), ("ness", 3),
        ("ous", 3), ("ive", 3), ("ful", 3), ("less", 3),
        ("able", 3), ("ible", 3),
        ("es", 3), ("ed", 3), ("er", 3), ("ly", 3),
        ("s", 3),
    ]:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            return word[:-len(suffix)]
    return None


def tokenize_text(text: str, max_len: int = 64, remove_stopwords: bool = False) -> list[str]:
    tokens = []
    for m in _WORD_RE.finditer(text):
        w = m.group()
        if len(w) > max_len:
            continue
        low = w.lower()
        if remove_stopwords and low in _STOPWORDS:
            continue
        tokens.append(low)
        stem = _stem(low)
        if stem and stem != low:
            tokens.append(stem)
    return tokens


@runtime_checkable
class Chunker(Protocol):
    def chunks(self, text: str) -> list[str]: ...


class TextChunker:
    __slots__ = ("_splitter",)

    def __init__(self, capacity: int = 256, model: str = "gpt-4") -> None:
        try:
            self._splitter = TextSplitter.from_tiktoken_model(model, capacity)
        except Exception:
            self._splitter = TextSplitter(capacity * 4)

    def chunks(self, text: str) -> list[str]:
        return self._splitter.chunks(text)


class MarkdownChunker:
    __slots__ = ("_splitter",)

    def __init__(self, capacity: int = 256, model: str = "gpt-4") -> None:
        try:
            self._splitter = MarkdownSplitter.from_tiktoken_model(model, capacity)
        except Exception:
            self._splitter = MarkdownSplitter(capacity * 4)

    def chunks(self, text: str) -> list[str]:
        return self._splitter.chunks(text)


# ---------------------------------------------------------------------------
# Language detection + regex outline (vendored from tinyclaw.tools.outline)
# ---------------------------------------------------------------------------

_EXT_LANG: dict[str, str] = {
    ".py": "python", ".rs": "rust", ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "javascript", ".go": "go",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".java": "java", ".rb": "ruby",
}


def _detect_lang(path: str) -> str | None:
    return _EXT_LANG.get(Path(path).suffix.lower())


_PATTERNS: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "python": [
        (re.compile(r"^(\s*)class\s+(\w+)"), "class"),
        (re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)"), "def"),
    ],
    "rust": [
        (re.compile(r"^(\s*)pub\s+(?:async\s+)?fn\s+(\w+)"), "fn"),
        (re.compile(r"^(\s*)fn\s+(\w+)"), "fn"),
        (re.compile(r"^(\s*)(?:pub\s+)?struct\s+(\w+)"), "struct"),
        (re.compile(r"^(\s*)(?:pub\s+)?enum\s+(\w+)"), "enum"),
        (re.compile(r"^(\s*)(?:pub\s+)?trait\s+(\w+)"), "trait"),
        (re.compile(r"^(\s*)impl(?:<[^>]*>)?\s+(\w+(?:\s+for\s+\w+)?)"), "impl"),
        (re.compile(r"^(\s*)(?:pub\s+)?mod\s+(\w+)"), "mod"),
    ],
    "typescript": [
        (re.compile(r"^(\s*)(?:export\s+)?(?:default\s+)?class\s+(\w+)"), "class"),
        (re.compile(r"^(\s*)(?:export\s+)?interface\s+(\w+)"), "interface"),
        (re.compile(r"^(\s*)(?:export\s+)?type\s+(\w+)"), "type"),
        (re.compile(r"^(\s*)(?:export\s+)?(?:async\s+)?function\s+(\w+)"), "fn"),
        (re.compile(r"^(\s*)(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\("), "fn"),
    ],
    "go": [
        (re.compile(r"^(\s*)func\s+(?:\([^)]+\)\s+)?(\w+)"), "fn"),
        (re.compile(r"^(\s*)type\s+(\w+)\s+struct"), "struct"),
        (re.compile(r"^(\s*)type\s+(\w+)\s+interface"), "interface"),
        (re.compile(r"^(\s*)type\s+(\w+)"), "type"),
    ],
    "c": [
        (re.compile(r"^(\s*)(?:static\s+)?(?:inline\s+)?(?:\w+\s+)+\*?(\w+)\s*\("), "fn"),
        (re.compile(r"^(\s*)(?:typedef\s+)?struct\s+(\w+)"), "struct"),
        (re.compile(r"^(\s*)(?:typedef\s+)?enum\s+(\w+)"), "enum"),
    ],
    "java": [
        (re.compile(r"^(\s*)(?:public|private|protected)?\s*(?:static\s+)?class\s+(\w+)"), "class"),
        (re.compile(r"^(\s*)(?:public|private|protected)?\s*(?:static\s+)?interface\s+(\w+)"), "interface"),
        (re.compile(r"^(\s*)(?:public|private|protected)?\s*(?:static\s+)?(?:[\w<>\[\]]+\s+)(\w+)\s*\("), "method"),
    ],
    "ruby": [
        (re.compile(r"^(\s*)class\s+(\w+)"), "class"),
        (re.compile(r"^(\s*)module\s+(\w+)"), "module"),
        (re.compile(r"^(\s*)def\s+(\w+)"), "def"),
    ],
}
_PATTERNS["tsx"] = _PATTERNS["typescript"]
_PATTERNS["javascript"] = _PATTERNS["typescript"]
_PATTERNS["cpp"] = [
    *_PATTERNS["c"],
    (re.compile(r"^(\s*)class\s+(\w+)"), "class"),
    (re.compile(r"^(\s*)namespace\s+(\w+)"), "namespace"),
]


def _regex_outline(source: str, lang: str) -> list[tuple[int, int, str, str]]:
    patterns = _PATTERNS.get(lang)
    if not patterns:
        return []
    symbols: list[tuple[int, int, str, str]] = []
    for line_no, line in enumerate(source.splitlines(), 1):
        for pat, kind in patterns:
            m = pat.match(line)
            if m:
                indent = len(m.group(1))
                name = m.group(2)
                depth = indent // 4 if "\t" not in m.group(1) else m.group(1).count("\t")
                symbols.append((line_no, depth, kind, name))
                break
    return symbols


class SymbolChunker:
    __slots__ = ("_fallback", "_path")

    def __init__(self, path: str = "", capacity: int = 512) -> None:
        self._fallback = TextChunker(capacity=capacity)
        self._path = path

    def chunks(self, text: str) -> list[str]:
        lang = _detect_lang(self._path) if self._path else None
        if not lang:
            return self._fallback.chunks(text)
        symbols = _regex_outline(text, lang)
        if not symbols:
            return self._fallback.chunks(text)
        lines = text.splitlines()
        result: list[str] = []
        fname = Path(self._path).name if self._path else ""
        for i, (line_no, depth, kind, name) in enumerate(symbols):
            end_line = len(lines)
            for j in range(i + 1, len(symbols)):
                next_line, next_depth, _, _ = symbols[j]
                if next_depth <= depth:
                    end_line = next_line - 1
                    break
            body = "\n".join(lines[line_no - 1 : end_line])
            if not body.strip():
                continue
            header = f"# {kind} {name}"
            if fname:
                header += f" ({fname}:{line_no})"
            result.append(f"{header}\n{body}")
        return result if result else self._fallback.chunks(text)
