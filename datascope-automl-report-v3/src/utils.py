from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_output_dir(path: str | Path = "outputs") -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def safe_round(value: Any, digits: int = 4) -> Any:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return round(float(value), digits)
    except Exception:
        return value


def format_percent(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def is_probably_identifier(series: pd.Series) -> bool:
    if series.empty:
        return False
    name = str(series.name).lower()
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    return ("id" == name or name.endswith("_id") or name.endswith("id")) and unique_ratio > 0.8


def short_text(value: Any, max_len: int = 60) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def optional_dependency_status() -> pd.DataFrame:
    packages = [
        ("xgboost", "XGBoost model interface"),
        ("lightgbm", "LightGBM model interface"),
        ("shap", "SHAP explainability"),
        ("pydicom", "DICOM reading"),
        ("torch", "CNN/RNN/LSTM/ViT deep models"),
        ("openpyxl", "Excel .xlsx reading"),
    ]
    rows = []
    for package, purpose in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "installed")
            installed = True
        except Exception:
            version = "not installed"
            installed = False
        rows.append({"package": package, "installed": installed, "version": version, "purpose": purpose})
    return pd.DataFrame(rows)
