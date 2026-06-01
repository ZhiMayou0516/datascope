from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


@dataclass
class ColumnTypes:
    numeric: list[str]
    categorical: list[str]
    datetime: list[str]
    boolean: list[str]
    high_cardinality: list[str]


def basic_overview(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "missing_cell_ratio": float(df.isna().sum().sum() / max(df.shape[0] * df.shape[1], 1)),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "missing_ratio": [float(df[c].isna().mean()) for c in df.columns],
            "unique_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
    return summary.sort_values(["missing_ratio", "missing_count"], ascending=False).reset_index(drop=True)


def _looks_like_datetime_name(column: str) -> bool:
    pattern = r"(date|time|timestamp|datetime|created|updated|日期|时间)"
    return re.search(pattern, str(column), flags=re.IGNORECASE) is not None


def _is_parseable_datetime(series: pd.Series, min_success_ratio: float = 0.75) -> bool:
    if is_datetime64_any_dtype(series):
        return True
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.astype(str).head(min(200, len(non_null)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= min_success_ratio)


def is_high_cardinality(series: pd.Series, absolute_threshold: int = 50, ratio_threshold: float = 0.5) -> bool:
    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / max(len(series.dropna()), 1)
    return bool(unique_count >= absolute_threshold or (unique_count >= 15 and unique_ratio >= ratio_threshold))


def detect_column_types(df: pd.DataFrame) -> ColumnTypes:
    datetime_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    bool_cols: list[str] = []
    high_cardinality_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        if is_bool_dtype(series):
            bool_cols.append(col)
            categorical_cols.append(col)
        elif _is_parseable_datetime(series) and (_looks_like_datetime_name(col) or not is_numeric_dtype(series)):
            datetime_cols.append(col)
        elif is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            if is_high_cardinality(series):
                high_cardinality_cols.append(col)

    return ColumnTypes(
        numeric=numeric_cols,
        categorical=categorical_cols,
        datetime=datetime_cols,
        boolean=bool_cols,
        high_cardinality=high_cardinality_cols,
    )


def numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    return df[numeric_cols].describe().T.reset_index().rename(columns={"index": "column"})


def categorical_summary(df: pd.DataFrame, categorical_cols: list[str], top_n: int = 5) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        vc = df[col].astype("object").value_counts(dropna=True).head(top_n)
        rows.append(
            {
                "column": col,
                "unique_count": int(df[col].nunique(dropna=True)),
                "unique_ratio": float(df[col].nunique(dropna=True) / max(df[col].notna().sum(), 1)),
                "is_high_cardinality": is_high_cardinality(df[col]),
                "top_values": ", ".join([f"{idx}: {cnt}" for idx, cnt in vc.items()]),
            }
        )
    return pd.DataFrame(rows)


def infer_task_type(y: pd.Series, unique_threshold: int = 20) -> str:
    non_null = y.dropna()
    if non_null.empty:
        return "classification"
    unique_count = non_null.nunique(dropna=True)
    if is_bool_dtype(non_null) or non_null.dtype == "object" or str(non_null.dtype).startswith("category"):
        return "classification"
    if unique_count <= min(unique_threshold, max(2, int(len(non_null) * 0.1))):
        return "classification"
    return "regression"


def target_quality_check(df: pd.DataFrame, target_col: str) -> tuple[bool, list[str]]:
    messages: list[str] = []
    if target_col not in df.columns:
        return False, ["选择的 target column 不存在。"]

    missing_ratio = float(df[target_col].isna().mean())
    if missing_ratio > 0.5:
        return False, [f"目标列缺失比例过高：{missing_ratio:.1%}。建议先清洗数据。"]
    if missing_ratio > 0:
        messages.append(f"目标列有 {missing_ratio:.1%} 缺失值，训练时会自动删除这些行。")

    unique_count = df[target_col].nunique(dropna=True)
    if unique_count < 2:
        return False, ["目标列有效类别/取值少于 2 个，无法建模。"]

    if len(df.dropna(subset=[target_col])) < 10:
        return False, ["删除目标缺失值后样本数少于 10，模型评价不可靠。"]

    return True, messages


def compact_dataset_warnings(df: pd.DataFrame) -> list[str]:
    warnings_list: list[str] = []
    if len(df) < 30:
        warnings_list.append("当前样本数少于 30，模型结果仅适合流程演示，不适合作为可靠结论。")
    if df.shape[1] > max(50, len(df) * 2):
        warnings_list.append("特征数相对样本数偏多，baseline 模型可能过拟合。")
    high_missing = missing_summary(df).query("missing_ratio >= 0.5")
    if not high_missing.empty:
        warnings_list.append("存在缺失比例超过 50% 的列，建议建模前考虑删除或补充数据。")
    high_card = detect_column_types(df).high_cardinality
    if high_card:
        warnings_list.append(f"检测到高基数类别列：{', '.join(high_card[:5])}。建模时会默认合并稀有类别，避免 One-Hot 维度爆炸。")
    return warnings_list


def outlier_report_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            lower, upper = q1, q3
            mask = pd.Series(False, index=series.index)
        else:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)
        rows.append(
            {
                "column": col,
                "outlier_count": int(mask.sum()),
                "outlier_ratio": float(mask.mean()) if len(mask) else 0.0,
                "lower_bound": float(lower) if pd.notna(lower) else None,
                "upper_bound": float(upper) if pd.notna(upper) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_ratio", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def _normalized_name(text: str) -> str:
    return re.sub(r"[^a-z0-9一-龥]", "", str(text).lower())


def leakage_report(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Heuristic leakage checks. It is conservative and meant to warn, not decide."""
    if target_col not in df.columns:
        return pd.DataFrame()
    y = df[target_col]
    task = infer_task_type(y)
    target_name = _normalized_name(target_col)
    risky_keywords = ["label", "target", "outcome", "result", "diagnosis", "class", "score", "结果", "标签", "诊断", "结局"]
    rows = []
    for col in df.columns:
        if col == target_col:
            continue
        s = df[col]
        reasons = []
        risk_score = 0
        col_name = _normalized_name(col)
        if target_name and (target_name in col_name or col_name in target_name):
            reasons.append("列名与 target 高度相似")
            risk_score += 3
        if any(k in col_name for k in risky_keywords):
            reasons.append("列名包含 label/target/outcome/result 等高风险词")
            risk_score += 2
        try:
            comparable = s.astype(str).fillna("<NA>").reset_index(drop=True) == y.astype(str).fillna("<NA>").reset_index(drop=True)
            equal_ratio = float(comparable.mean())
            if equal_ratio > 0.95:
                reasons.append(f"与 target 逐行几乎相同（{equal_ratio:.1%}）")
                risk_score += 5
        except Exception:
            pass
        if task == "regression" and is_numeric_dtype(s) and is_numeric_dtype(y):
            corr = pd.concat([pd.to_numeric(s, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).corr().iloc[0, 1]
            if pd.notna(corr) and abs(corr) > 0.98:
                reasons.append(f"与 target 相关系数极高（{corr:.3f}）")
                risk_score += 4
        elif task == "classification":
            non_null = pd.DataFrame({"x": s, "y": y}).dropna()
            if len(non_null) > 0:
                unique_x = non_null["x"].nunique(dropna=True)
                if unique_x > 1:
                    purity = non_null.groupby("x")["y"].nunique(dropna=True).le(1).mean()
                    if unique_x >= 10 and purity > 0.95:
                        reasons.append("高比例特征取值可唯一映射到 target，可能是 ID/泄漏特征")
                        risk_score += 3
        if reasons:
            rows.append({"column": col, "risk_score": risk_score, "reasons": "; ".join(reasons)})
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["column", "risk_score", "reasons"])


def high_cardinality_report(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        unique_count = int(df[col].nunique(dropna=True))
        non_null = int(df[col].notna().sum())
        unique_ratio = unique_count / max(non_null, 1)
        rows.append(
            {
                "column": col,
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "recommended_action": "Rare category grouping + OneHot" if is_high_cardinality(df[col]) else "OneHot",
            }
        )
    columns = ["column", "unique_count", "unique_ratio", "recommended_action"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["unique_count", "unique_ratio"], ascending=False).reset_index(drop=True)
