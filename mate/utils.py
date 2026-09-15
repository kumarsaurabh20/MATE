"""Small shared helpers for sorting, column lookup, and numerical orientation."""

import re
from typing import List, Optional

import numpy as np
import pandas as pd


def natural_key(value: object) -> List[object]:
    """Natural sort key: 2 before 10, T2 before T10."""
    text = str(value)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first matching column name from a candidate list, case-insensitive."""
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def safe_output_prefix(prefix: object) -> str:
    prefix = str(prefix) if prefix else "results"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)


def orient_vector(vec: np.ndarray) -> np.ndarray:
    """
    SVD vectors have arbitrary sign. Orient the vector so that the largest
    absolute entry is positive. This makes outputs more consistent.
    """
    vec = np.asarray(vec, dtype=float)
    if vec.size == 0 or np.all(np.isnan(vec)):
        return vec
    idx = int(np.nanargmax(np.abs(vec)))
    if vec[idx] < 0:
        vec = -vec
    return vec


def parse_comma_list(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]
