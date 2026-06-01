from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_VERSION = "2.0"


def make_project_config(
    *,
    data_source: str | None,
    sheet_name: str | None,
    target_col: str | None,
    task_type: str | None,
    model_config: Any | None,
    notes: str = "",
) -> dict[str, Any]:
    if model_config is None:
        model_payload = None
    elif is_dataclass(model_config):
        model_payload = asdict(model_config)
    elif hasattr(model_config, "to_dict"):
        model_payload = model_config.to_dict()
    else:
        model_payload = dict(model_config)
    return {
        "config_version": DEFAULT_CONFIG_VERSION,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": data_source,
        "sheet_name": sheet_name,
        "target_col": target_col,
        "task_type": task_type,
        "model_config": model_payload,
        "notes": notes,
    }


def config_to_bytes(config: dict[str, Any]) -> bytes:
    return json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")


def load_config_from_bytes(raw: bytes) -> dict[str, Any]:
    try:
        config = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"配置文件解析失败：{exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("配置文件格式错误：根对象必须是 JSON object。")
    return config


def save_config(config: dict[str, Any], output_dir: str | Path = "outputs", filename: str = "datascope_config.json") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_bytes(config_to_bytes(config))
    return path
