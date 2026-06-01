from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

COMMON_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin1")
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}


class DataLoadError(ValueError):
    """Raised when a tabular file cannot be safely loaded."""


def _decode_bytes(raw: bytes, encodings: Iterable[str] = COMMON_ENCODINGS) -> tuple[str, str]:
    if not raw or len(raw.strip()) == 0:
        raise DataLoadError("文件为空，请上传包含表头和数据行的数据文件。")

    last_error: Exception | None = None
    for enc in encodings:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError as exc:
            last_error = exc
    raise DataLoadError(f"无法识别文件编码，请尝试另存为 UTF-8 后重新上传。原始错误：{last_error}")


def _check_duplicate_header(text: str, delimiter: str = ",") -> None:
    lines = text.splitlines()
    first_line = lines[0] if lines else ""
    if not first_line.strip():
        raise DataLoadError("文件缺少表头，请确认第一行是列名。")

    try:
        header = next(csv.reader([first_line], delimiter=delimiter))
    except csv.Error as exc:
        raise DataLoadError(f"表头解析失败：{exc}") from exc

    normalized = [h.strip() for h in header]
    duplicated = sorted({name for name in normalized if normalized.count(name) > 1 and name != ""})
    empty_count = sum(name == "" for name in normalized)

    if empty_count > 0:
        raise DataLoadError("存在空列名，请先补全列名后再上传。")
    if duplicated:
        raise DataLoadError(f"存在重复列名：{', '.join(duplicated)}。请先重命名这些列。")


def _validate_loaded_dataframe(df: pd.DataFrame, *, allow_tiny: bool = False) -> None:
    if df.empty:
        raise DataLoadError("没有有效数据行，请至少保留若干行样本。")
    if df.shape[1] < 2:
        raise DataLoadError("至少需要 2 列：一个或多个特征列，以及一个 target 列。")
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise DataLoadError(f"存在重复列名：{duplicated}")
    if len(df) < 5 and not allow_tiny:
        raise DataLoadError("数据行数少于 5 行，暂不适合做自动建模演示。")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    copied = df.copy()
    copied.columns = [str(c).strip() for c in copied.columns]
    if copied.columns.duplicated().any():
        duplicated = copied.columns[copied.columns.duplicated()].tolist()
        raise DataLoadError(f"清理列名后出现重复列：{duplicated}")
    if any(c == "" for c in copied.columns):
        raise DataLoadError("存在空列名，请先补全列名。")
    return copied


def _read_delimited(raw: bytes, filename: str, delimiter: str, sample_rows: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    text, encoding = _decode_bytes(raw)
    _check_duplicate_header(text, delimiter=delimiter)
    try:
        df = pd.read_csv(StringIO(text), sep=delimiter, nrows=sample_rows)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("文件为空或没有可读取的数据。") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"解析失败，请检查分隔符、引号或异常换行。错误：{exc}") from exc
    except Exception as exc:
        raise DataLoadError(f"读取文件时发生未知错误：{exc}") from exc
    df = clean_column_names(df)
    _validate_loaded_dataframe(df)
    return df, {"source": filename, "encoding": encoding, "format": "tsv" if delimiter == "\t" else "csv", "size_bytes": len(raw), "sample_rows": sample_rows}


def _read_excel(raw: bytes, filename: str, selected_sheet: str | int | None = None, sample_rows: int | None = None) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    try:
        excel_file = pd.ExcelFile(BytesIO(raw))
        sheet_names = list(excel_file.sheet_names)
        if not sheet_names:
            raise DataLoadError("Excel 文件没有可读取的 sheet。")
        sheet = selected_sheet if selected_sheet is not None else sheet_names[0]
        if isinstance(sheet, str) and sheet not in sheet_names:
            sheet = sheet_names[0]
        df = pd.read_excel(excel_file, sheet_name=sheet, nrows=sample_rows)
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"读取 Excel 失败：{exc}") from exc
    df = clean_column_names(df)
    _validate_loaded_dataframe(df)
    meta = {"source": filename, "encoding": "excel-binary", "format": "excel", "sheet_name": str(sheet), "size_bytes": len(raw), "sample_rows": sample_rows}
    return df, meta, sheet_names


def read_tabular_upload(uploaded_file, selected_sheet: str | int | None = None, sample_rows: int | None = None) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Read CSV/TSV/TXT/XLSX/XLS with validation. Returns df, metadata, sheet_names."""
    filename = getattr(uploaded_file, "name", "uploaded_file")
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(f"暂不支持 {ext or '未知'} 格式。当前支持 CSV、TSV、TXT、XLSX、XLS。")
    raw = uploaded_file.getvalue()
    if ext in {".xlsx", ".xls"}:
        return _read_excel(raw, filename, selected_sheet, sample_rows=sample_rows)
    delimiter = "\t" if ext in {".tsv", ".txt"} else ","
    df, meta = _read_delimited(raw, filename, delimiter, sample_rows=sample_rows)
    return df, meta, []


def _detect_encoding_from_file(path: Path, encodings: Iterable[str] = COMMON_ENCODINGS) -> str:
    raw = path.read_bytes()[: min(path.stat().st_size, 1024 * 1024)]
    for enc in encodings:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin1"


def _read_first_line(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="replace", newline="") as f:
        return f.readline()


def read_tabular_path(path: str | Path, selected_sheet: str | int | None = None, sample_rows: int | None = None) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Read a local tabular file. This avoids Streamlit's browser upload path for large files."""
    path = Path(path).expanduser()
    if not path.exists():
        raise DataLoadError(f"数据不存在：{path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(f"暂不支持 {ext or '未知'} 格式。当前支持 CSV、TSV、TXT、XLSX、XLS。")

    if ext in {".xlsx", ".xls"}:
        raw = path.read_bytes()
        df, meta, sheets = _read_excel(raw, path.name, selected_sheet=selected_sheet, sample_rows=sample_rows)
        meta["source"] = str(path)
        return df, meta, sheets

    delimiter = "\t" if ext in {".tsv", ".txt"} else ","
    encoding = _detect_encoding_from_file(path)
    _check_duplicate_header(_read_first_line(path, encoding), delimiter=delimiter)
    try:
        df = pd.read_csv(path, sep=delimiter, encoding=encoding, nrows=sample_rows)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("文件为空或没有可读取的数据。") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"解析失败，请检查分隔符、引号或异常换行。错误：{exc}") from exc
    except Exception as exc:
        raise DataLoadError(f"读取文件时发生未知错误：{exc}") from exc
    df = clean_column_names(df)
    _validate_loaded_dataframe(df)
    meta = {
        "source": str(path),
        "encoding": encoding,
        "format": "tsv" if delimiter == "\t" else "csv",
        "size_bytes": path.stat().st_size,
        "sample_rows": sample_rows,
    }
    return df, meta, []

# Backward-compatible aliases.
def read_csv_from_upload(uploaded_file) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, meta, _ = read_tabular_upload(uploaded_file)
    return df, meta


def read_csv_from_path(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, meta, _ = read_tabular_path(path)
    return df, meta
