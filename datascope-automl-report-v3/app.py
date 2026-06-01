from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config_manager import config_to_bytes, load_config_from_bytes, make_project_config, save_config
from src.data_loader import DataLoadError, clean_column_names, read_tabular_path, read_tabular_upload
from src.deep_models import DeepTrainConfig, model_summary, torch_available, train_deep_baseline
from src.eda import (
    basic_overview,
    categorical_summary,
    compact_dataset_warnings,
    detect_column_types,
    high_cardinality_report,
    infer_task_type,
    leakage_report,
    missing_summary,
    numeric_summary,
    outlier_report_iqr,
    target_quality_check,
)
from src.explainability import compute_shap_importance
from src.imaging import ImagingLoadError, array_task_type, guess_feature_and_label_arrays, load_npz_upload, npz_to_feature_table, read_dicom_upload
from src.modeling import ModelConfig, get_available_models, infer_task_for_target, train_model
from src.report import generate_markdown_report, markdown_to_simple_html, save_report
from src.utils import dataframe_to_csv_bytes, ensure_output_dir, optional_dependency_status, safe_round
from src.visualization import (
    categorical_frequency,
    confusion_matrix_heatmap,
    correlation_heatmap,
    feature_importance_bar,
    missing_bar,
    numeric_distribution,
    npz_array_preview,
    outlier_bar,
    regression_prediction_scatter,
    shap_importance_bar,
    target_distribution,
)

APP_TITLE = "DataScope AutoML Report"
SAMPLE_CLASSIFICATION = Path("sample_data/classification_demo.csv")
SAMPLE_REGRESSION = Path("sample_data/regression_demo.csv")
SAMPLE_IMAGING_NPZ = Path("sample_data/imaging_npz_demo.npz")

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

CUSTOM_CSS = """
<style>
.main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.metric-card {border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px 18px; background: linear-gradient(180deg, #ffffff, #f8fafc); box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);}
.small-muted {color: #64748b; font-size: 0.92rem;}
.section-title {font-size: 1.25rem; font-weight: 700; margin-top: 0.5rem;}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


class UploadedBytes:
    def __init__(self, name: str, raw: bytes):
        self.name = name
        self._raw = raw

    def getvalue(self) -> bytes:
        return self._raw


def init_state() -> None:
    defaults = {
        "df": None,
        "data_meta": {},
        "sheet_names": [],
        "uploaded_name": None,
        "uploaded_raw": None,
        "sample_rows": None,
        "target_col": None,
        "task_type": None,
        "model_config": None,
        "model_result": None,
        "shap_df": None,
        "report_markdown": None,
        "npz_arrays": None,
        "npz_summary": None,
        "dicom_meta": None,
        "dicom_pixel": None,
        "deep_metrics": None,
        "deep_predictions": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_model_state() -> None:
    st.session_state.model_config = None
    st.session_state.model_result = None
    st.session_state.shap_df = None
    st.session_state.report_markdown = None


def reset_data_state() -> None:
    st.session_state.target_col = None
    st.session_state.task_type = None
    reset_model_state()


def load_sample(path: Path) -> None:
    try:
        df, meta, sheets = read_tabular_path(path)
        st.session_state.df = clean_column_names(df)
        st.session_state.data_meta = meta
        st.session_state.sheet_names = sheets
        st.session_state.uploaded_name = None
        st.session_state.uploaded_raw = None
        reset_data_state()
        st.success(f"已加载示例数据：{path.name}")
    except DataLoadError as exc:
        st.error(str(exc))


def load_current_uploaded_sheet(sheet_name: str | None = None, sample_rows: int | None = None) -> None:
    raw = st.session_state.uploaded_raw
    name = st.session_state.uploaded_name
    if not raw or not name:
        return
    try:
        df, meta, sheets = read_tabular_upload(UploadedBytes(name, raw), selected_sheet=sheet_name, sample_rows=sample_rows or st.session_state.get("sample_rows"))
        st.session_state.df = clean_column_names(df)
        st.session_state.data_meta = meta
        st.session_state.sheet_names = sheets
        reset_data_state()
        st.success(f"已加载：{name}" + (f" / sheet={meta.get('sheet_name')}" if meta.get("sheet_name") else ""))
    except DataLoadError as exc:
        st.error(str(exc))


def require_data() -> pd.DataFrame | None:
    df = st.session_state.df
    if df is None:
        st.info("请先在 Upload Data 页面上传 CSV/TSV/Excel，或加载内置 demo 数据。")
        return None
    return df


def render_header() -> None:
    st.title("📊 DataScope AutoML Report")
    st.caption("科研表格数据 + 影像/数组数据的自动分析、建模、解释与报告生成工具。")


def render_overview_cards(df: pd.DataFrame) -> None:
    overview = basic_overview(df)
    cols = st.columns(5)
    items = [
        ("Rows", overview["n_rows"]),
        ("Columns", overview["n_columns"]),
        ("Duplicate Rows", overview["duplicate_rows"]),
        ("Missing Cells", overview["total_missing_cells"]),
        ("Missing Ratio", f"{overview['missing_cell_ratio']:.2%}"),
    ]
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def page_upload() -> None:
    st.subheader("Upload Data")
    st.write("支持 CSV、TSV、TXT、Excel 单/多 sheet。没有数据时可以直接加载内置 demo。")

    left, right = st.columns([1.25, 1])
    with left:
        with st.expander("大文件模式 / 本地路径读取", expanded=False):
            st.caption("Streamlit 默认上传上限已在 .streamlit/config.toml 中提高到 2048MB。若文件仍然很大，建议直接填本机路径读取，避免浏览器上传占内存。")
            sample_rows = st.number_input("预览/采样读取行数，0 表示读取全部", min_value=0, max_value=2_000_000, value=0, step=1000)
            local_path = st.text_input("本机数据路径（可选，例如 D:/data/table.csv）", value="")
            if st.button("从本机路径读取", use_container_width=True) and local_path.strip():
                try:
                    df, meta, sheets = read_tabular_path(local_path.strip(), sample_rows=int(sample_rows) or None)
                    st.session_state.df = clean_column_names(df)
                    st.session_state.data_meta = meta
                    st.session_state.sheet_names = sheets
                    st.session_state.uploaded_name = None
                    st.session_state.uploaded_raw = None
                    reset_data_state()
                    st.success(f"已从路径读取：{local_path.strip()}")
                except Exception as exc:
                    st.error(f"读取失败：{exc}")

        uploaded_file = st.file_uploader("选择表格数据文件", type=["csv", "tsv", "txt", "xlsx", "xls"])
        if uploaded_file is not None:
            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.uploaded_raw = uploaded_file.getvalue()
            st.session_state.sample_rows = int(sample_rows) or None
            load_current_uploaded_sheet(sample_rows=st.session_state.sample_rows)

        if st.session_state.sheet_names:
            selected_sheet = st.selectbox("Excel sheet", st.session_state.sheet_names)
            if st.button("切换到该 sheet", use_container_width=True):
                load_current_uploaded_sheet(selected_sheet, sample_rows=st.session_state.get("sample_rows"))

    with right:
        st.markdown("#### Demo 数据")
        c1, c2 = st.columns(2)
        if c1.button("加载分类 demo", use_container_width=True):
            load_sample(SAMPLE_CLASSIFICATION)
        if c2.button("加载回归 demo", use_container_width=True):
            load_sample(SAMPLE_REGRESSION)
        st.caption("demo 数据位于 `sample_data/`，适合 README 截图和现场演示。")

    df = st.session_state.df
    if df is not None:
        st.divider()
        st.markdown("### 数据概览")
        render_overview_cards(df)
        meta = st.session_state.data_meta or {}
        st.caption(
            f"数据来源：{meta.get('source', 'unknown')} ｜ 格式：{meta.get('format', '-')} ｜ 编码/引擎：{meta.get('encoding', '-')}"
        )

        for msg in compact_dataset_warnings(df):
            st.warning(msg)

        types = detect_column_types(df)
        col1, col2, col3, col4 = st.columns(4)
        col1.write("**数值列**")
        col1.write(types.numeric or "无")
        col2.write("**类别列**")
        col2.write(types.categorical or "无")
        col3.write("**高基数类别列**")
        col3.write(types.high_cardinality or "无")
        col4.write("**可能的时间列**")
        col4.write(types.datetime or "无")

        st.markdown("#### 前几行数据")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("#### 缺失值与类型")
        st.dataframe(missing_summary(df), use_container_width=True)


def page_eda() -> None:
    st.subheader("EDA")
    df = require_data()
    if df is None:
        return

    types = detect_column_types(df)
    render_overview_cards(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Missing", "Numeric", "Categorical", "Correlation", "Target"])

    with tab1:
        st.plotly_chart(missing_bar(df), use_container_width=True)
        st.dataframe(missing_summary(df), use_container_width=True)

    with tab2:
        if not types.numeric:
            st.info("未识别到数值列。")
        else:
            selected_num = st.selectbox("选择数值变量", types.numeric, key="eda_numeric_col")
            st.plotly_chart(numeric_distribution(df, selected_num), use_container_width=True)
            st.dataframe(numeric_summary(df, types.numeric), use_container_width=True)

    with tab3:
        if not types.categorical:
            st.info("未识别到类别列。")
        else:
            selected_cat = st.selectbox("选择类别变量", types.categorical, key="eda_cat_col")
            st.plotly_chart(categorical_frequency(df, selected_cat), use_container_width=True)
            st.dataframe(categorical_summary(df, types.categorical), use_container_width=True)

    with tab4:
        if len(types.numeric) < 2:
            st.info("相关性热图至少需要 2 个数值列。")
        else:
            max_cols = st.slider("最多纳入相关性热图的数值列数", 3, min(30, len(types.numeric)), min(12, len(types.numeric)))
            st.plotly_chart(correlation_heatmap(df, types.numeric[:max_cols]), use_container_width=True)

    with tab5:
        target = st.selectbox(
            "选择 target column",
            df.columns,
            index=0 if st.session_state.target_col is None else list(df.columns).index(st.session_state.target_col),
        )
        task_type = infer_task_type(df[target])
        st.session_state.target_col = target
        st.session_state.task_type = task_type
        st.info(f"自动识别任务类型：**{task_type}**")
        ok, messages = target_quality_check(df, target)
        for msg in messages:
            st.warning(msg)
        if ok:
            st.plotly_chart(target_distribution(df, target, task_type), use_container_width=True)
        else:
            st.error("目标列暂不适合建模，请更换 target 或清洗数据。")


def page_quality() -> None:
    st.subheader("Data Quality")
    df = require_data()
    if df is None:
        return
    types = detect_column_types(df)

    st.markdown("### 异常值检测")
    outlier_df = outlier_report_iqr(df, types.numeric)
    if outlier_df.empty:
        st.info("没有数值列或未检测到明显 IQR 异常值。")
    else:
        st.plotly_chart(outlier_bar(outlier_df), use_container_width=True)
        st.dataframe(outlier_df, use_container_width=True)

    st.markdown("### 高基数类别特征")
    high_df = high_cardinality_report(df, types.categorical)
    if high_df.empty:
        st.info("没有类别列。")
    else:
        st.dataframe(high_df, use_container_width=True)
        if "recommended_action" in high_df.columns and not high_df.query("recommended_action != 'OneHot'").empty:
            st.warning("建模时会默认先合并长尾类别，再 OneHot，避免类别维度爆炸。")

    st.markdown("### 数据泄漏检查")
    if st.session_state.target_col is None:
        default_idx = 0
    else:
        default_idx = list(df.columns).index(st.session_state.target_col)
    target = st.selectbox("用于泄漏检查的 target", df.columns, index=default_idx)
    leak_df = leakage_report(df, target)
    if leak_df.empty:
        st.success("未检测到明显泄漏风险。这个检查是启发式规则，仍建议人工确认。")
    else:
        st.warning("检测到潜在泄漏风险，请人工判断这些列是否在预测时真实可用。")
        st.dataframe(leak_df, use_container_width=True)


def page_modeling() -> None:
    st.subheader("Modeling")
    df = require_data()
    if df is None:
        return

    with st.form("modeling_form"):
        target_col = st.selectbox(
            "Target column",
            df.columns,
            index=0 if st.session_state.target_col is None else list(df.columns).index(st.session_state.target_col),
        )
        detected_task = infer_task_for_target(df, target_col)
        task_type = st.radio(
            "Task type",
            ["classification", "regression"],
            index=0 if detected_task == "classification" else 1,
            horizontal=True,
        )
        model_name = st.selectbox("Model", get_available_models(task_type))

        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = c2.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
        scale_numeric = c3.checkbox("Scale numeric features", value=True)

        st.markdown("#### 高基数类别处理")
        h1, h2 = st.columns(2)
        rare_min_frequency = h1.slider("Rare category min frequency", 0.001, 0.10, 0.01, step=0.001, format="%.3f")
        rare_max_categories = h2.slider("Max categories kept per categorical column", 5, 100, 30, step=5)

        st.markdown("#### 交叉验证与超参数搜索")
        c4, c5, c6, c7 = st.columns(4)
        use_cv = c4.checkbox("启用交叉验证", value=False)
        cv_folds = c5.slider("CV folds", 2, 10, 5)
        use_search = c6.checkbox("启用超参数搜索", value=False)
        search_type = c7.selectbox("Search type", ["random", "grid"])
        n_iter = st.slider("Random search iterations", 4, 40, 12, disabled=not use_search or search_type == "grid")

        submitted = st.form_submit_button("训练模型", use_container_width=True)

    ok, messages = target_quality_check(df, target_col)
    for msg in messages:
        st.warning(msg)
    if not ok:
        st.error("当前 target 不满足建模条件。")
        return

    if submitted:
        config = ModelConfig(
            target_col=target_col,
            task_type=task_type,
            model_name=model_name,
            test_size=float(test_size),
            random_state=int(random_state),
            scale_numeric=bool(scale_numeric),
            rare_min_frequency=float(rare_min_frequency),
            rare_max_categories=int(rare_max_categories),
            use_cv=bool(use_cv),
            cv_folds=int(cv_folds),
            use_search=bool(use_search),
            search_type=search_type,
            n_iter=int(n_iter),
        )
        try:
            with st.spinner("正在训练模型；如果启用搜索，会比 baseline 稍慢..."):
                result = train_model(df, config)
            st.session_state.target_col = target_col
            st.session_state.task_type = task_type
            st.session_state.model_config = config
            st.session_state.model_result = result
            st.session_state.shap_df = None
            st.session_state.report_markdown = None
            st.success("模型训练完成。")
        except Exception as exc:
            st.error(f"模型训练失败：{exc}")

    result = st.session_state.model_result
    if result is None:
        st.info("设置参数后点击训练，即可查看指标、交叉验证/搜索结果、特征重要性和预测 CSV。")
        return

    st.divider()
    st.markdown("### 模型结果")
    st.caption(
        f"模型：{result.model_name} ｜ 任务：{result.task_type} ｜ 使用特征数：{len(result.used_features)} ｜ 删除 target 缺失/无效行：{result.dropped_rows}"
    )

    if result.task_type == "classification":
        metrics = result.metrics
        cols = st.columns(5)
        cols[0].metric("Accuracy", safe_round(metrics.get("accuracy"), 4))
        cols[1].metric("Precision", safe_round(metrics.get("precision_weighted"), 4))
        cols[2].metric("Recall", safe_round(metrics.get("recall_weighted"), 4))
        cols[3].metric("F1", safe_round(metrics.get("f1_weighted"), 4))
        cols[4].metric("ROC-AUC", safe_round(metrics.get("roc_auc"), 4) if metrics.get("roc_auc") is not None else "N/A")
        st.plotly_chart(confusion_matrix_heatmap(metrics["confusion_matrix"]), use_container_width=True)
    else:
        metrics = result.metrics
        cols = st.columns(3)
        cols[0].metric("R²", safe_round(metrics.get("r2"), 4))
        cols[1].metric("MAE", safe_round(metrics.get("mae"), 4))
        cols[2].metric("RMSE", safe_round(metrics.get("rmse"), 4))
        st.plotly_chart(regression_prediction_scatter(result.predictions), use_container_width=True)

    if result.cv_summary is not None and not result.cv_summary.empty:
        st.markdown("### 交叉验证结果")
        st.dataframe(result.cv_summary, use_container_width=True)

    if result.search_summary is not None and not result.search_summary.empty:
        st.markdown("### 超参数搜索结果")
        st.json(result.best_params or {})
        st.dataframe(result.search_summary, use_container_width=True)

    st.markdown("### 预测结果")
    st.dataframe(result.predictions.head(200), use_container_width=True)
    st.download_button(
        "下载预测结果 CSV",
        data=dataframe_to_csv_bytes(result.predictions),
        file_name="datascope_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### Feature Importance")
    if result.feature_importance is None or result.feature_importance.empty:
        st.info("当前模型不支持 feature importance，或无法从 Pipeline 中提取特征名。")
    else:
        st.plotly_chart(feature_importance_bar(result.feature_importance), use_container_width=True)
        st.dataframe(result.feature_importance.head(50), use_container_width=True)


def page_explainability() -> None:
    st.subheader("Explainability")
    result = st.session_state.model_result
    if result is None:
        st.info("请先在 Modeling 页面训练模型。")
        return

    st.write("这里计算全局 SHAP 重要性。为了不卡死，默认只抽取测试集的一部分样本。")
    max_samples = st.slider("Max SHAP samples", 20, 300, 120, step=20)
    if st.button("计算 SHAP", use_container_width=True):
        with st.spinner("正在计算 SHAP，全局重要性通常几秒到几十秒不等..."):
            shap_df, err = compute_shap_importance(result, max_samples=max_samples)
        if err:
            st.warning(err)
        else:
            st.session_state.shap_df = shap_df
            st.success("SHAP 计算完成。")

    shap_df = st.session_state.shap_df
    if shap_df is not None and not shap_df.empty:
        st.plotly_chart(shap_importance_bar(shap_df), use_container_width=True)
        st.dataframe(shap_df.head(60), use_container_width=True)


def page_imaging() -> None:
    st.subheader("Imaging / NPZ / DICOM")
    st.write("这个页面用于展示你常用的影像数据能力：DICOM 元数据/像素预览、NPZ 数组读取、特征表转换，以及轻量 CNN/RNN/LSTM/ViT 原型训练。")

    tab_npz, tab_dicom, tab_deep = st.tabs(["NPZ arrays", "DICOM", "Deep models"])

    with tab_npz:
        uploaded_npz = st.file_uploader("上传 NPZ 文件", type=["npz"], key="npz_upload")
        c1, c2 = st.columns([1, 1])
        if c1.button("加载内置 NPZ demo", use_container_width=True):
            try:
                with open(SAMPLE_IMAGING_NPZ, "rb") as f:
                    arrays, summary = load_npz_upload(UploadedBytes(SAMPLE_IMAGING_NPZ.name, f.read()))
                st.session_state.npz_arrays = arrays
                st.session_state.npz_summary = summary
                st.success("已加载 imaging_npz_demo.npz")
            except Exception as exc:
                st.error(str(exc))
        if uploaded_npz is not None:
            try:
                arrays, summary = load_npz_upload(uploaded_npz)
                st.session_state.npz_arrays = arrays
                st.session_state.npz_summary = summary
                st.success("NPZ 读取成功。")
            except ImagingLoadError as exc:
                st.error(str(exc))

        arrays = st.session_state.npz_arrays
        if arrays:
            st.dataframe(st.session_state.npz_summary, use_container_width=True)
            names = list(arrays.keys())
            arr_name = st.selectbox("选择数组预览", names)
            arr = arrays[arr_name]
            idx = st.slider("sample index", 0, max(0, arr.shape[0] - 1), 0) if arr.ndim >= 3 else 0
            st.plotly_chart(npz_array_preview(arr, title=arr_name, index=idx), use_container_width=True)

            guess_x, guess_y = guess_feature_and_label_arrays(arrays)
            st.markdown("#### 将 NPZ 转成统计特征表")
            x_name = st.selectbox("X array", names, index=names.index(guess_x) if guess_x in names else 0)
            y_options = ["<none>"] + names
            y_default = y_options.index(guess_y) if guess_y in names else 0
            y_name = st.selectbox("y/target array", y_options, index=y_default)
            if st.button("生成影像统计特征表并载入表格工作流", use_container_width=True):
                try:
                    feature_df = npz_to_feature_table(arrays, x_name, None if y_name == "<none>" else y_name)
                    st.session_state.df = feature_df
                    st.session_state.data_meta = {"source": f"NPZ:{x_name}", "format": "npz-derived-table", "encoding": "binary"}
                    st.session_state.target_col = "target" if "target" in feature_df.columns else None
                    st.session_state.task_type = infer_task_type(feature_df["target"]) if "target" in feature_df.columns else None
                    reset_model_state()
                    st.success("已生成表格特征并载入 Upload/EDA/Modeling 工作流。")
                    st.dataframe(feature_df.head(), use_container_width=True)
                except Exception as exc:
                    st.error(f"转换失败：{exc}")

    with tab_dicom:
        uploaded_dicom = st.file_uploader("上传 DICOM 文件", type=["dcm", "dicom", "ima"], key="dicom_upload")
        if uploaded_dicom is not None:
            try:
                meta, pixel = read_dicom_upload(uploaded_dicom)
                st.session_state.dicom_meta = meta
                st.session_state.dicom_pixel = pixel
                st.success("DICOM 读取成功。")
            except ImagingLoadError as exc:
                st.error(str(exc))
        if st.session_state.dicom_meta is not None:
            st.dataframe(st.session_state.dicom_meta, use_container_width=True)
            pixel = st.session_state.dicom_pixel
            if pixel is not None:
                st.plotly_chart(npz_array_preview(pixel, title="DICOM pixel preview"), use_container_width=True)
            else:
                st.info("该 DICOM 未能读取 pixel_array，可能是压缩格式或缺少像素数据。")

    with tab_deep:
        arrays = st.session_state.npz_arrays
        if not torch_available():
            st.warning("当前环境未安装 torch。页面仍可展示接口，但训练 CNN/RNN/LSTM/ViT 需要安装 torch。")
        if not arrays:
            st.info("请先在 NPZ arrays 标签页上传或加载 NPZ 数据。")
            return
        names = list(arrays.keys())
        guess_x, guess_y = guess_feature_and_label_arrays(arrays)
        x_name = st.selectbox("Deep model X array", names, index=names.index(guess_x) if guess_x in names else 0, key="deep_x")
        y_name = st.selectbox("Deep model y array", names, index=names.index(guess_y) if guess_y in names else 0, key="deep_y")
        X = arrays[x_name]
        y = arrays[y_name]
        inferred_task = array_task_type(y)
        d1, d2, d3 = st.columns(3)
        model_type = d1.selectbox("Architecture", ["CNN", "RNN", "LSTM", "ViT"])
        task_type = d2.radio("Task", ["classification", "regression"], index=0 if inferred_task == "classification" else 1, horizontal=True)
        epochs = d3.slider("Epochs", 1, 20, 3)
        st.markdown("#### Architecture summary")
        output_dim = int(pd.Series(y.reshape(-1)).nunique()) if task_type == "classification" else 1
        st.dataframe(model_summary(model_type, tuple(X.shape[1:]), task_type, output_dim), use_container_width=True)

        b1, b2, b3 = st.columns(3)
        batch_size = b1.slider("Batch size", 4, 64, 16, step=4)
        lr = b2.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)
        max_samples = b3.slider("Max samples", 32, 2048, 512, step=32)
        if st.button("训练轻量深度学习 baseline", use_container_width=True):
            cfg = DeepTrainConfig(model_type=model_type, task_type=task_type, epochs=epochs, batch_size=batch_size, learning_rate=lr, max_samples=max_samples)
            try:
                with st.spinner("正在 CPU 上训练轻量深度学习 demo，默认只跑少量 epoch..."):
                    metrics_df, pred_df = train_deep_baseline(X, y, cfg)
                st.session_state.deep_metrics = metrics_df
                st.session_state.deep_predictions = pred_df
                st.success("深度学习 baseline 训练完成。")
            except Exception as exc:
                st.error(f"训练失败：{exc}")
        if st.session_state.deep_metrics is not None:
            st.dataframe(st.session_state.deep_metrics, use_container_width=True)
            st.dataframe(st.session_state.deep_predictions.head(100), use_container_width=True)


def page_report() -> None:
    st.subheader("Report")
    df = require_data()
    if df is None:
        return

    target_col = st.session_state.target_col
    task_type = st.session_state.task_type
    config = st.session_state.model_config
    result = st.session_state.model_result
    metrics = result.metrics if result else None
    feature_importance = result.feature_importance if result else None
    source = (st.session_state.data_meta or {}).get("source", "unknown")

    if st.button("生成 Markdown / HTML 报告", use_container_width=True):
        report_md = generate_markdown_report(
            df=df,
            source=source,
            target_col=target_col,
            task_type=task_type,
            model_config=config,
            metrics=metrics,
            feature_importance=feature_importance,
            model_result=result,
        )
        save_report(report_md, ensure_output_dir(), stem="datascope_report", df=df, target_col=target_col, task_type=task_type)
        st.session_state.report_markdown = report_md
        st.success("报告已生成，可在下方预览和下载。HTML 报告会自动嵌入 EDA 图表。")

    report_md = st.session_state.report_markdown
    if report_md is None:
        st.info("建议先完成 Modeling，再生成包含模型指标、CV、搜索结果和数据质量检查的报告。也可以只基于数据概览生成报告。")
        return

    st.markdown("### 报告预览")
    st.markdown(report_md)

    html = markdown_to_simple_html(report_md, df=df, target_col=target_col, task_type=task_type)
    c1, c2 = st.columns(2)
    c1.download_button(
        "下载 Markdown 报告",
        data=report_md.encode("utf-8"),
        file_name="datascope_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    c2.download_button(
        "下载 HTML 报告（含 EDA 图表）",
        data=html.encode("utf-8"),
        file_name="datascope_report.html",
        mime="text/html",
        use_container_width=True,
    )


def page_config() -> None:
    st.subheader("Config / Reproducibility")
    st.write("保存当前 target、任务类型、模型参数、数据来源信息，方便复现实验流程。")
    notes = st.text_area("Notes", value="", placeholder="例如：classification_demo.csv, target=outcome, RF + CV")
    config_payload = make_project_config(
        data_source=(st.session_state.data_meta or {}).get("source"),
        sheet_name=(st.session_state.data_meta or {}).get("sheet_name"),
        target_col=st.session_state.target_col,
        task_type=st.session_state.task_type,
        model_config=st.session_state.model_config,
        notes=notes,
    )
    st.json(config_payload)
    c1, c2 = st.columns(2)
    if c1.button("保存到 outputs/datascope_config.json", use_container_width=True):
        path = save_config(config_payload, ensure_output_dir())
        st.success(f"已保存：{path}")
    c2.download_button(
        "下载配置 JSON",
        data=config_to_bytes(config_payload),
        file_name="datascope_config.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("### 载入配置")
    uploaded_config = st.file_uploader("上传 datascope_config.json", type=["json"], key="config_upload")
    if uploaded_config is not None:
        try:
            cfg = load_config_from_bytes(uploaded_config.getvalue())
            st.session_state.target_col = cfg.get("target_col") if cfg.get("target_col") in (list(st.session_state.df.columns) if st.session_state.df is not None else []) else st.session_state.target_col
            st.session_state.task_type = cfg.get("task_type") or st.session_state.task_type
            st.success("配置已读取。当前页面不会自动训练模型，请到 Modeling 页面确认后重新训练。")
            st.json(cfg)
        except Exception as exc:
            st.error(str(exc))


def page_about() -> None:
    st.subheader("About")
    st.markdown(
        """
        **DataScope AutoML Report** 是一个面向科研数据的小型数据科学工作台。  
        现在它同时覆盖：表格数据 AutoML、数据质量检查、交叉验证/超参数搜索、XGBoost/LightGBM 接口、SHAP 解释、报告生成、DICOM/NPZ 影像数据预览，以及 CNN/RNN/LSTM/ViT 原型模型。
        """
    )
    st.markdown("#### 依赖状态")
    st.dataframe(optional_dependency_status(), use_container_width=True)
    st.markdown("#### 推荐展示流程")
    st.write("1. Upload Data 加载 demo 或上传 CSV/TSV/Excel。")
    st.write("2. EDA + Data Quality 展示自动分析、异常值、泄漏、高基数特征处理。")
    st.write("3. Modeling 训练 baseline，可开启 CV/搜索和 XGBoost/LightGBM。")
    st.write("4. Explainability 计算 SHAP。")
    st.write("5. Imaging 页面展示 DICOM/NPZ 与 CNN/RNN/LSTM/ViT 原型。")
    st.write("6. Report 生成含图表的 HTML 报告，Config 保存复现实验配置。")


def main() -> None:
    init_state()
    render_header()

    page = st.sidebar.radio(
        "Navigation",
        ["Upload Data", "EDA", "Data Quality", "Modeling", "Explainability", "Imaging", "Report", "Config", "About"],
    )
    st.sidebar.divider()
    st.sidebar.caption("Local-first · No external API · Python 3.10+")

    if page == "Upload Data":
        page_upload()
    elif page == "EDA":
        page_eda()
    elif page == "Data Quality":
        page_quality()
    elif page == "Modeling":
        page_modeling()
    elif page == "Explainability":
        page_explainability()
    elif page == "Imaging":
        page_imaging()
    elif page == "Report":
        page_report()
    elif page == "Config":
        page_config()
    else:
        page_about()


if __name__ == "__main__":
    main()
