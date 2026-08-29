from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config


VALID_ANALYSIS_MODES = {"strict", "expanded", "all"}


def ensure_output_dirs() -> None:
    for path in config.OUTPUT_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def critical_composition_mask(df: pd.DataFrame) -> pd.Series:
    """Agreed model exclusions; the source workbook is never edited."""
    def num(name):
        return pd.to_numeric(df[name], errors="coerce")

    return (
        num("Zn").le(0)
        | num("Al").lt(75)
        | num("Al").gt(99.9)
        | num("Cu").gt(6)
        | num("Ti").gt(1)
        | num("Zr").gt(1)
    ).fillna(False)


def read_sheet(task: str, mode: str = "expanded", exclude_critical: bool = False) -> pd.DataFrame:
    if task not in config.SHEETS:
        raise KeyError(f"Unknown task: {task}")
    if mode not in VALID_ANALYSIS_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_ANALYSIS_MODES)}")
    if not config.WORKBOOK_PATH.exists():
        raise FileNotFoundError(config.WORKBOOK_PATH)

    df = pd.read_excel(config.WORKBOOK_PATH, sheet_name=config.SHEETS[task])
    required = {"Model_Row_ID", "Source_Group", "Include_Main_Model", "Sensitivity_Group"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{task} sheet is missing required columns: {sorted(missing)}")

    if mode == "strict":
        df = df.loc[df["Sensitivity_Group"].eq("Primary")].copy()
    elif mode == "expanded":
        include = pd.to_numeric(df["Include_Main_Model"], errors="coerce").fillna(0).eq(1)
        df = df.loc[include].copy()
    else:
        df = df.copy()

    if exclude_critical:
        df = df.loc[~critical_composition_mask(df)].copy()

    df["Source_Group"] = df["Source_Group"].astype("string").str.strip()
    if df["Source_Group"].isna().any() or df["Source_Group"].eq("").any():
        bad = df.loc[df["Source_Group"].isna() | df["Source_Group"].eq(""), "Model_Row_ID"].tolist()
        raise ValueError(f"{task} has blank Source_Group values: {bad[:10]}")
    return df.reset_index(drop=True)


def expected_targets(task: str) -> list[str]:
    if task in config.TARGET_COLUMNS:
        return [config.TARGET_COLUMNS[task]]
    if task in {"MTL", "TRIPLE"}:
        return [config.TARGET_COLUMNS[t] for t in ("YS", "UTS", "EL")]
    if task == "FOUR":
        return [config.TARGET_COLUMNS[t] for t in ("YS", "UTS", "EL", "HV")]
    raise KeyError(task)


def feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    feature_sets = {
        "composition": config.COMPOSITION_FEATURES,
        "composition_derived": config.COMPOSITION_FEATURES + config.DERIVED_FEATURES,
        "composition_process": config.COMPOSITION_FEATURES + config.PROCESS_FEATURES,
        "composition_derived_process": (
            config.COMPOSITION_FEATURES + config.DERIVED_FEATURES + config.PROCESS_FEATURES
        ),
    }
    if feature_set not in feature_sets:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    cols = [c for c in feature_sets[feature_set] if c in df.columns]
    forbidden = set(config.METADATA_COLUMNS + config.AUDIT_ONLY_COLUMNS + config.ALL_TARGET_AND_MASK_COLUMNS)
    overlap = forbidden.intersection(cols)
    if overlap:
        raise AssertionError(f"Forbidden leakage columns selected: {sorted(overlap)}")
    return cols


def validate_mtl_masks(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for task in ("YS", "UTS", "EL"):
        target = config.TARGET_COLUMNS[task]
        mask = config.MASK_COLUMNS[task]
        if target not in df or mask not in df:
            raise ValueError(f"MTL is missing {target} or {mask}")
        expected = df[target].notna().astype(int)
        actual = pd.to_numeric(df[mask], errors="coerce").fillna(-1).astype(int)
        mismatch = int((expected != actual).sum())
        records.append({"target": task, "observed": int(expected.sum()), "mask_mismatch": mismatch})
    return pd.DataFrame(records)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, ensure_ascii=False, indent=2)


def numeric_feature_summary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in columns:
        if col not in df:
            continue
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        observed = int(s.notna().sum())
        nonzero = int(s.fillna(0).ne(0).sum())
        rows.append({
            "feature": col,
            "rows": n,
            "observed": observed,
            "missing_pct": 100.0 * (n - observed) / n if n else np.nan,
            "nonzero": nonzero,
            "unique_observed": int(s.nunique(dropna=True)),
            "min": s.min(),
            "median": s.median(),
            "max": s.max(),
        })
    return pd.DataFrame(rows)
