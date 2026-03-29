"""Temporal awareness for search.

Index time: extract dates from doc text, store in attributes.
Query time: detect date references in queries, compute temporal proximity scores.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from math import exp

# ── Date extraction (index time) ──

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_YEAR = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
_YEAR_MONTH_NUM = re.compile(r"\b(\d{4})-(\d{2})\b")  # 2026-01 without day
_Q_YEAR = re.compile(r"\bQ([1-4])\s+(\d{4})\b", re.IGNORECASE)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def extract_dates(text: str) -> list[str]:
    """Extract ISO date strings from text. Returns sorted unique dates as YYYY-MM-DD."""
    dates: set[str] = set()

    for m in _ISO_DATE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            date(y, mo, d)  # validate
            dates.add(f"{y:04d}-{mo:02d}-{d:02d}")
        except ValueError:
            pass

    for m in _MONTH_YEAR.finditer(text):
        mo = _MONTH_MAP[m.group(1).lower()]
        y = int(m.group(2))
        dates.add(f"{y:04d}-{mo:02d}-15")  # center of month

    return sorted(dates)


def dates_to_periods(dates: list[str]) -> list[str]:
    """Convert date strings to period node IDs for the hypergraph.
    Returns unique period IDs: month:YYYY-MM, quarter:YYYY-QN, year:YYYY."""
    periods: set[str] = set()
    for ds in dates:
        try:
            d = date.fromisoformat(ds)
        except ValueError:
            continue
        periods.add(f"month:{d.year:04d}-{d.month:02d}")
        q = (d.month - 1) // 3 + 1
        periods.add(f"quarter:{d.year:04d}-Q{q}")
        periods.add(f"year:{d.year:04d}")
    return sorted(periods)


# ── Query date detection (query time) ──

_Q_MONTH_YEAR = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
_Q_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_Q_QUARTER = re.compile(r"\bQ([1-4])\s+(\d{4})\b", re.IGNORECASE)
_Q_MONTH_SHORT = re.compile(
    r"\b(March|April|May|June|July|August|September|October|November|December|"
    r"January|February)\s+(\d{4})\b",
    re.IGNORECASE,
)


def detect_query_period_ids(query: str) -> list[str]:
    """Detect date references in query and return matching period node IDs."""
    ranges = detect_query_dates(query)
    if not ranges:
        return []
    period_ids: set[str] = set()
    for start, end in ranges:
        # Add month-level periods for all months in range
        d = start.replace(day=1)
        while d <= end:
            period_ids.add(f"month:{d.year:04d}-{d.month:02d}")
            q = (d.month - 1) // 3 + 1
            period_ids.add(f"quarter:{d.year:04d}-Q{q}")
            # next month
            if d.month == 12:
                d = d.replace(year=d.year + 1, month=1)
            else:
                d = d.replace(month=d.month + 1)
    return sorted(period_ids)


def detect_query_dates(query: str) -> list[tuple[date, date]] | None:
    """Detect date ranges in a query. Returns list of (start, end) date pairs, or None."""
    ranges: list[tuple[date, date]] = []

    # ISO dates: exact day
    for m in _Q_ISO.finditer(query):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = date(y, mo, d)
            ranges.append((dt, dt))
        except ValueError:
            pass

    # "January 2026" → full month range
    for m in _Q_MONTH_YEAR.finditer(query):
        mo = _MONTH_MAP[m.group(1).lower()]
        y = int(m.group(2))
        start = date(y, mo, 1)
        if mo == 12:
            end = date(y + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(y, mo + 1, 1) - timedelta(days=1)
        ranges.append((start, end))

    # "Q1 2026" → quarter range
    for m in _Q_QUARTER.finditer(query):
        q, y = int(m.group(1)), int(m.group(2))
        start_month = (q - 1) * 3 + 1
        start = date(y, start_month, 1)
        end_month = start_month + 2
        if end_month == 12:
            end = date(y + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(y, end_month + 1, 1) - timedelta(days=1)
        ranges.append((start, end))

    return ranges if ranges else None


# ── Temporal proximity scoring (RRF signal) ──

_SIGMA_DAYS = 15.0  # Gaussian std dev — 15d away ≈ 0.6, 30d ≈ 0.13


def temporal_proximity_score(
    doc_dates: list[str],
    query_ranges: list[tuple[date, date]],
) -> float:
    """Compute temporal proximity score for a doc given query date ranges.
    Returns 0.0-1.0 where 1.0 = perfect date match."""
    if not doc_dates or not query_ranges:
        return 0.0

    best = 0.0
    for ds in doc_dates:
        try:
            dd = date.fromisoformat(ds)
        except ValueError:
            continue
        for start, end in query_ranges:
            # Distance = 0 if within range, else min distance to range edges
            if start <= dd <= end:
                dist = 0.0
            else:
                dist = min(abs((dd - start).days), abs((dd - end).days))
            score = exp(-(dist ** 2) / (2 * _SIGMA_DAYS ** 2))
            best = max(best, score)

    return best
