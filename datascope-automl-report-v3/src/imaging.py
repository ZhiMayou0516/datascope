from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd


class ImagingLoadError(ValueError):
    pass


def load_npz_upload(uploaded_file) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    raw = uploaded_file.getvalue()
    if not raw:
        raise ImagingLoadError("NPZ 文件为空。")
    try:
        data = np.load(BytesIO(raw), allow_pickle=False)
        arrays = {key: data[key] for key in data.files}
    except Exception as exc:
        raise ImagingLoadError(f"NPZ 读取失败：{exc}") from exc
    if not arrays:
        raise ImagingLoadError("NPZ 中没有可读取数组。")
    summary = summarize_npz_arrays(arrays)
    return arrays, summary


def summarize_npz_arrays(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for key, arr in arrays.items():
        a = np.asarray(arr)
        rows.append(
            {
                "name": key,
                "shape": " × ".join(map(str, a.shape)),
                "ndim": int(a.ndim),
                "dtype": str(a.dtype),
                "min": float(np.nanmin(a)) if np.issubdtype(a.dtype, np.number) and a.size else None,
                "max": float(np.nanmax(a)) if np.issubdtype(a.dtype, np.number) and a.size else None,
                "mean": float(np.nanmean(a)) if np.issubdtype(a.dtype, np.number) and a.size else None,
            }
        )
    return pd.DataFrame(rows)


def guess_feature_and_label_arrays(arrays: dict[str, np.ndarray]) -> tuple[str | None, str | None]:
    label_candidates = ["y", "label", "labels", "target", "targets", "class", "classes"]
    y_name = None
    for name in arrays:
        if name.lower() in label_candidates:
            y_name = name
            break
    x_name = None
    if y_name:
        y_len = len(arrays[y_name]) if arrays[y_name].ndim >= 1 else None
        for name, arr in arrays.items():
            if name == y_name:
                continue
            if arr.ndim >= 2 and y_len is not None and len(arr) == y_len:
                x_name = name
                break
    if x_name is None:
        for name, arr in arrays.items():
            if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
                x_name = name
                break
    return x_name, y_name


def npz_to_feature_table(arrays: dict[str, np.ndarray], x_name: str, y_name: str | None = None) -> pd.DataFrame:
    if x_name not in arrays:
        raise ImagingLoadError("选择的 X 数组不存在。")
    X = np.asarray(arrays[x_name])
    if X.ndim < 2:
        raise ImagingLoadError("X 数组至少需要二维，例如 N×H×W 或 N×T×F。")
    n = X.shape[0]
    flat = X.reshape(n, -1).astype(float)
    features = pd.DataFrame(
        {
            "img_mean": np.nanmean(flat, axis=1),
            "img_std": np.nanstd(flat, axis=1),
            "img_min": np.nanmin(flat, axis=1),
            "img_max": np.nanmax(flat, axis=1),
            "img_p10": np.nanpercentile(flat, 10, axis=1),
            "img_p50": np.nanpercentile(flat, 50, axis=1),
            "img_p90": np.nanpercentile(flat, 90, axis=1),
            "img_energy": np.nanmean(flat**2, axis=1),
        }
    )
    features.insert(0, "sample_index", np.arange(n))
    if y_name and y_name in arrays:
        y = np.asarray(arrays[y_name]).reshape(-1)
        if len(y) != n:
            raise ImagingLoadError("X 和 y 的第一个维度不一致。")
        features["target"] = y
    return features


def read_dicom_upload(uploaded_file) -> tuple[pd.DataFrame, np.ndarray | None]:
    try:
        import pydicom
    except Exception as exc:
        raise ImagingLoadError("pydicom 未安装，无法读取 DICOM。请运行 pip install pydicom。") from exc
    raw = uploaded_file.getvalue()
    if not raw:
        raise ImagingLoadError("DICOM 文件为空。")
    try:
        ds = pydicom.dcmread(BytesIO(raw), force=True)
    except Exception as exc:
        raise ImagingLoadError(f"DICOM 读取失败：{exc}") from exc

    fields = [
        "PatientID", "PatientAge", "PatientSex", "StudyDate", "Modality", "Manufacturer",
        "Rows", "Columns", "PixelSpacing", "SliceThickness", "SeriesDescription", "StudyDescription",
    ]
    rows = []
    for field in fields:
        value = getattr(ds, field, None)
        if value is not None:
            rows.append({"tag": field, "value": str(value)})
    pixel = None
    try:
        pixel = ds.pixel_array.astype(float)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixel = pixel * slope + intercept
    except Exception:
        pixel = None
    return pd.DataFrame(rows), pixel


def array_task_type(y: np.ndarray) -> str:
    y = np.asarray(y).reshape(-1)
    if y.dtype.kind in {"U", "S", "O", "b"}:
        return "classification"
    unique = np.unique(y[~pd.isna(y)]) if y.size else []
    if len(unique) <= min(20, max(2, int(len(y) * 0.1))):
        return "classification"
    return "regression"
