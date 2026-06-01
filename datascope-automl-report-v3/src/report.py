from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template

from src.eda import basic_overview, detect_column_types, high_cardinality_report, leakage_report, missing_summary, outlier_report_iqr
from src.utils import safe_round
from src.visualization import correlation_heatmap, missing_bar, target_distribution

REPORT_TEMPLATE = """# DataScope AutoML Report

生成时间：{{ generated_at }}  
数据来源：{{ source }}

## 1. 数据概览

| 项目 | 数值 |
|---|---:|
| 样本数 | {{ overview.n_rows }} |
| 特征/列数量 | {{ overview.n_columns }} |
| 重复行数量 | {{ overview.duplicate_rows }} |
| 总缺失单元格 | {{ overview.total_missing_cells }} |
| 总缺失比例 | {{ "%.2f%%" | format(overview.missing_cell_ratio * 100) }} |
| 内存占用 MB | {{ "%.3f" | format(overview.memory_mb) }} |

## 2. 自动识别的列类型

- 数值列：{{ column_types.numeric | join(", ") if column_types.numeric else "无" }}
- 类别列：{{ column_types.categorical | join(", ") if column_types.categorical else "无" }}
- 高基数类别列：{{ column_types.high_cardinality | join(", ") if column_types.high_cardinality else "无" }}
- 可能的时间列：{{ column_types.datetime | join(", ") if column_types.datetime else "无" }}

## 3. 缺失值分析

{% if top_missing %}
| 列名 | 数据类型 | 缺失数量 | 缺失比例 | 唯一值数量 |
|---|---|---:|---:|---:|
{% for row in top_missing %}| {{ row.column }} | {{ row.dtype }} | {{ row.missing_count }} | {{ "%.2f%%" | format(row.missing_ratio * 100) }} | {{ row.unique_count }} |
{% endfor %}
{% else %}
当前数据未检测到缺失值。
{% endif %}

## 4. 数据质量检查

### 4.1 异常值检测（IQR）
{% if outliers %}
| 列名 | 异常值数量 | 异常值比例 | 下界 | 上界 |
|---|---:|---:|---:|---:|
{% for row in outliers %}| {{ row.column }} | {{ row.outlier_count }} | {{ "%.2f%%" | format(row.outlier_ratio * 100) }} | {{ row.lower_bound }} | {{ row.upper_bound }} |
{% endfor %}
{% else %}
未检测到明显异常值或没有数值列。
{% endif %}

### 4.2 数据泄漏风险
{% if leakage %}
| 列名 | 风险分 | 原因 |
|---|---:|---|
{% for row in leakage %}| {{ row.column }} | {{ row.risk_score }} | {{ row.reasons }} |
{% endfor %}
{% else %}
未检测到明显数据泄漏风险。注意：该检查为启发式规则，不能替代人工审查。
{% endif %}

### 4.3 高基数类别特征
{% if high_cardinality %}
| 列名 | 唯一值数量 | 唯一值比例 | 建议处理 |
|---|---:|---:|---|
{% for row in high_cardinality %}| {{ row.column }} | {{ row.unique_count }} | {{ "%.2f%%" | format(row.unique_ratio * 100) }} | {{ row.recommended_action }} |
{% endfor %}
{% else %}
未检测到需要特殊处理的高基数类别特征。
{% endif %}

## 5. 目标变量

- Target column：{{ target_col or "未选择" }}
- 任务类型：{{ task_type or "未建模" }}

{% if target_summary %}
{{ target_summary }}
{% endif %}

## 6. 模型设置

{% if model_config %}
- 模型：{{ model_config.model_name }}
- 测试集比例：{{ model_config.test_size }}
- 随机种子：{{ model_config.random_state }}
- 数值特征标准化：{{ "是" if model_config.scale_numeric else "否" }}
- 交叉验证：{{ "启用" if model_config.use_cv else "未启用" }}
- 超参数搜索：{{ model_config.search_type if model_config.use_search else "未启用" }}
- 稀有类别合并：min_frequency={{ model_config.rare_min_frequency }}, max_categories={{ model_config.rare_max_categories }}
- 使用特征数：{{ used_feature_count }}
- 数值特征数：{{ numeric_feature_count }}
- 类别特征数：{{ categorical_feature_count }}
{% else %}
尚未训练模型。
{% endif %}

## 7. 模型指标

{% if metrics_table %}
| 指标 | 数值 |
|---|---:|
{% for key, value in metrics_table.items() %}| {{ key }} | {{ value }} |
{% endfor %}
{% else %}
尚无模型指标。
{% endif %}

{% if cv_rows %}
### 交叉验证结果
| 指标 | 均值 | 标准差 |
|---|---:|---:|
{% for row in cv_rows %}| {{ row.metric }} | {{ row.mean }} | {{ row.std }} |
{% endfor %}
{% endif %}

{% if best_params %}
### 最优参数
```json
{{ best_params }}
```
{% endif %}

## 8. 主要结论

{% for item in conclusions %}
- {{ item }}
{% endfor %}

## 9. 后续改进方向

- 增加更严格的时间外验证、外部验证集与数据泄漏人工审查。
- 增加 SHAP waterfall/force plot、模型校准曲线、决策曲线分析。
- 对 DICOM/NPZ 数据接入更完整的医学影像预处理和深度学习训练流水线。
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataScope AutoML Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1120px; margin: 36px auto; line-height: 1.65; color: #172033; }
    h1, h2, h3 { color: #0f172a; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td { border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }
    th { background: #f8fafc; }
    .card { border: 1px solid #e5e7eb; border-radius: 16px; padding: 18px 22px; margin: 16px 0; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); }
    .muted { color: #64748b; }
    pre { background: #f8fafc; padding: 12px; border-radius: 10px; overflow-x: auto; }
  </style>
</head>
<body>
<div class="card">
{{ markdown_html }}
</div>
{% if figures %}
<h2>自动嵌入的 EDA 图表</h2>
{% for fig in figures %}
<div class="card">{{ fig | safe }}</div>
{% endfor %}
{% endif %}
</body>
</html>"""


def _target_summary(df: pd.DataFrame, target_col: str | None, task_type: str | None) -> str:
    if not target_col or target_col not in df.columns:
        return ""
    series = df[target_col]
    if task_type == "classification":
        counts = series.astype("object").fillna("<missing>").value_counts().head(20)
        lines = ["目标变量频数 Top 20：", "", "| 取值 | 数量 |", "|---|---:|"]
        lines += [f"| {idx} | {cnt} |" for idx, cnt in counts.items()]
        return "\n".join(lines)
    desc = series.describe()
    lines = ["目标变量描述统计：", "", "| 指标 | 数值 |", "|---|---:|"]
    for key, value in desc.items():
        lines.append(f"| {key} | {safe_round(value, 4)} |")
    return "\n".join(lines)


def _metrics_for_report(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    skip = {"confusion_matrix", "labels"}
    output = {}
    for key, value in metrics.items():
        if key in skip:
            continue
        if value is None:
            output[key] = "不可用"
        elif isinstance(value, float):
            output[key] = safe_round(value, 4)
        else:
            output[key] = value
    return output


def build_conclusions(
    df: pd.DataFrame,
    target_col: str | None,
    task_type: str | None,
    metrics: dict[str, Any] | None,
    feature_importance: pd.DataFrame | None,
) -> list[str]:
    overview = basic_overview(df)
    missing = missing_summary(df)
    conclusions = [
        f"数据集包含 {overview['n_rows']} 行、{overview['n_columns']} 列，重复行数量为 {overview['duplicate_rows']}。",
        f"整体缺失比例为 {overview['missing_cell_ratio']:.2%}，建模前应重点检查高缺失列。",
    ]
    if target_col:
        conclusions.append(f"本次分析选择 `{target_col}` 作为目标变量，自动识别/设置为 {task_type} 任务。")
    if not missing.empty and missing.iloc[0]["missing_ratio"] > 0:
        conclusions.append(
            f"缺失比例最高的列是 `{missing.iloc[0]['column']}`，缺失比例为 {missing.iloc[0]['missing_ratio']:.2%}。"
        )
    if metrics:
        if task_type == "classification":
            conclusions.append(f"baseline 分类模型的 weighted F1 为 {metrics.get('f1_weighted', 0):.4f}。")
            if metrics.get("roc_auc") is not None:
                conclusions.append(f"二分类 ROC-AUC 为 {metrics.get('roc_auc'):.4f}。")
        elif task_type == "regression":
            conclusions.append(f"baseline 回归模型的 R² 为 {metrics.get('r2', 0):.4f}，RMSE 为 {metrics.get('rmse', 0):.4f}。")
    if feature_importance is not None and not feature_importance.empty:
        top = feature_importance.iloc[0]
        conclusions.append(f"当前模型中贡献最高的特征是 `{top['feature']}`。")
    return conclusions


def generate_markdown_report(
    df: pd.DataFrame,
    source: str = "unknown",
    target_col: str | None = None,
    task_type: str | None = None,
    model_config: Any | None = None,
    metrics: dict[str, Any] | None = None,
    feature_importance: pd.DataFrame | None = None,
    model_result: Any | None = None,
) -> str:
    overview = basic_overview(df)
    column_types = detect_column_types(df)
    missing = missing_summary(df)
    top_missing = missing.head(10).to_dict("records")
    metrics_table = _metrics_for_report(metrics)
    conclusions = build_conclusions(df, target_col, task_type, metrics, feature_importance)
    outlier_rows = outlier_report_iqr(df, column_types.numeric).head(10).to_dict("records")
    leakage_rows = leakage_report(df, target_col).head(10).to_dict("records") if target_col else []
    high_card_rows = high_cardinality_report(df, column_types.categorical).query("recommended_action != 'OneHot'").head(10).to_dict("records")
    cv_rows = getattr(model_result, "cv_summary", None)
    cv_rows = cv_rows.to_dict("records") if cv_rows is not None and not cv_rows.empty else []

    template = Template(REPORT_TEMPLATE)
    return template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source=source,
        overview=overview,
        column_types=column_types,
        top_missing=top_missing,
        outliers=outlier_rows,
        leakage=leakage_rows,
        high_cardinality=high_card_rows,
        target_col=target_col,
        task_type=task_type,
        target_summary=_target_summary(df, target_col, task_type),
        model_config=model_config,
        metrics_table=metrics_table,
        conclusions=conclusions,
        cv_rows=cv_rows,
        best_params=getattr(model_result, "best_params", None),
        used_feature_count=len(getattr(model_result, "used_features", [])) if model_result is not None else 0,
        numeric_feature_count=len(getattr(model_result, "numeric_features", [])) if model_result is not None else 0,
        categorical_feature_count=len(getattr(model_result, "categorical_features", [])) if model_result is not None else 0,
    )


def _markdown_to_html_blocks(markdown_text: str) -> str:
    # Small dependency-free renderer good enough for generated report preview/download.
    lines = markdown_text.splitlines()
    html: list[str] = []
    in_ul = False
    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []

    def flush_ul():
        nonlocal in_ul
        if in_ul:
            html.append("</ul>")
            in_ul = False

    def flush_table():
        nonlocal table_lines
        if not table_lines:
            return
        rows = [r.strip().strip("|").split("|") for r in table_lines if r.strip()]
        if len(rows) >= 2:
            html.append("<table>")
            html.append("<thead><tr>" + "".join(f"<th>{c.strip()}</th>" for c in rows[0]) + "</tr></thead>")
            html.append("<tbody>")
            for row in rows[2:]:
                html.append("<tr>" + "".join(f"<td>{c.strip()}</td>" for c in row) + "</tr>")
            html.append("</tbody></table>")
        table_lines = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            if not in_code:
                flush_ul(); flush_table(); in_code = True; code_lines = []
            else:
                html.append("<pre>" + "\n".join(code_lines) + "</pre>"); in_code = False
            continue
        if in_code:
            code_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue
        if line.startswith("|"):
            flush_ul(); table_lines.append(line); continue
        else:
            flush_table()
        if line.startswith("# "):
            flush_ul(); html.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            flush_ul(); html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            flush_ul(); html.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- "):
            if not in_ul:
                html.append("<ul>"); in_ul = True
            html.append(f"<li>{line[2:]}</li>")
        elif line.strip():
            flush_ul(); html.append(f"<p>{line}</p>")
    flush_ul(); flush_table()
    return "\n".join(html)


def report_figures(df: pd.DataFrame, target_col: str | None = None, task_type: str | None = None) -> list[str]:
    figs = [missing_bar(df)]
    types = detect_column_types(df)
    if len(types.numeric) >= 2:
        figs.append(correlation_heatmap(df, types.numeric[:12]))
    if target_col and target_col in df.columns and task_type:
        figs.append(target_distribution(df, target_col, task_type))
    return [fig.to_html(full_html=False, include_plotlyjs="inline" if i == 0 else False) for i, fig in enumerate(figs)]


def markdown_to_simple_html(markdown_text: str, df: pd.DataFrame | None = None, target_col: str | None = None, task_type: str | None = None) -> str:
    figures = report_figures(df, target_col, task_type) if df is not None else []
    template = Template(HTML_TEMPLATE)
    return template.render(markdown_html=_markdown_to_html_blocks(markdown_text), figures=figures)


def save_report(
    markdown_text: str,
    output_dir: str | Path = "outputs",
    stem: str = "datascope_report",
    df: pd.DataFrame | None = None,
    target_col: str | None = None,
    task_type: str | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{stem}.md"
    html_path = output_dir / f"{stem}.html"
    md_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(markdown_to_simple_html(markdown_text, df=df, target_col=target_col, task_type=task_type), encoding="utf-8")
    return md_path, html_path
