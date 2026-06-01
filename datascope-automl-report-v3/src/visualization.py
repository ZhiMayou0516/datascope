from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def missing_bar(df: pd.DataFrame) -> go.Figure:
    missing_ratio = df.isna().mean().sort_values(ascending=False).reset_index()
    missing_ratio.columns = ["column", "missing_ratio"]
    fig = px.bar(
        missing_ratio,
        x="column",
        y="missing_ratio",
        title="Missing Value Ratio by Column",
        labels={"missing_ratio": "Missing Ratio", "column": "Column"},
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-35, height=420, margin=dict(l=20, r=20, t=60, b=100))
    fig.update_yaxes(tickformat=".0%", range=[0, max(0.05, min(1.0, missing_ratio["missing_ratio"].max() * 1.15))])
    return fig


def numeric_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        marginal="box",
        title=f"Numeric Distribution: {column}",
        labels={column: column},
    )
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=20, r=20, t=60, b=40))
    return fig


def categorical_frequency(df: pd.DataFrame, column: str, top_n: int = 20) -> go.Figure:
    counts = df[column].astype("object").fillna("<missing>").value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(
        counts,
        x=column,
        y="count",
        title=f"Categorical Frequency: {column}",
        labels={column: column, "count": "Count"},
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-35, height=430, margin=dict(l=20, r=20, t=60, b=110))
    return fig


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Correlation Heatmap requires at least 2 numeric columns")
        return fig
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Numeric Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(template="plotly_white", height=max(420, min(800, 70 * len(numeric_cols))))
    return fig


def target_distribution(df: pd.DataFrame, target_col: str, task_type: str) -> go.Figure:
    if task_type == "classification":
        return categorical_frequency(df, target_col, top_n=30)
    return numeric_distribution(df, target_col)


def confusion_matrix_heatmap(cm_df: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=20, r=20, t=60, b=40))
    return fig


def feature_importance_bar(importance_df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    data = importance_df.head(top_n).sort_values("importance", ascending=True)
    fig = px.bar(
        data,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {min(top_n, len(data))} Feature Importance",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(template="plotly_white", height=max(420, 22 * len(data) + 120), margin=dict(l=20, r=20, t=60, b=40))
    return fig


def shap_importance_bar(shap_df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    data = shap_df.head(top_n).sort_values("mean_abs_shap", ascending=True)
    fig = px.bar(
        data,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title=f"Top {min(top_n, len(data))} Mean |SHAP|",
        labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
    )
    fig.update_layout(template="plotly_white", height=max(420, 22 * len(data) + 120), margin=dict(l=20, r=20, t=60, b=40))
    return fig


def regression_prediction_scatter(results_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        results_df,
        x="y_true",
        y="y_pred",
        title="Regression: Predicted vs Actual",
        labels={"y_true": "Actual", "y_pred": "Predicted"},
    )
    min_v = min(results_df["y_true"].min(), results_df["y_pred"].min())
    max_v = max(results_df["y_true"].max(), results_df["y_pred"].max())
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Ideal"))
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=20, r=20, t=60, b=40))
    return fig


def outlier_bar(outlier_df: pd.DataFrame) -> go.Figure:
    if outlier_df is None or outlier_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No numeric outliers detected")
        return fig
    data = outlier_df.sort_values("outlier_ratio", ascending=True)
    fig = px.bar(
        data,
        x="outlier_ratio",
        y="column",
        orientation="h",
        title="IQR Outlier Ratio by Numeric Column",
        labels={"outlier_ratio": "Outlier Ratio", "column": "Column"},
    )
    fig.update_layout(template="plotly_white", height=max(420, 25 * len(data) + 100))
    fig.update_xaxes(tickformat=".0%")
    return fig


def npz_array_preview(array, title: str = "Array Preview", index: int = 0, slice_index: int | None = None) -> go.Figure:
    import numpy as np

    arr = np.asarray(array)
    if arr.ndim == 1:
        fig = px.line(y=arr, title=title, labels={"index": "Index", "value": "Value"})
    elif arr.ndim == 2:
        fig = px.imshow(arr, title=title, color_continuous_scale="gray")
    elif arr.ndim == 3:
        idx = min(max(index, 0), arr.shape[0] - 1)
        fig = px.imshow(arr[idx], title=f"{title} | item {idx}", color_continuous_scale="gray")
    elif arr.ndim >= 4:
        idx = min(max(index, 0), arr.shape[0] - 1)
        sl = 0 if slice_index is None else min(max(slice_index, 0), arr.shape[1] - 1)
        img = arr[idx, sl]
        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
            img = img.squeeze()
        fig = px.imshow(img, title=f"{title} | item {idx}, slice {sl}", color_continuous_scale="gray")
    else:
        fig = go.Figure()
        fig.update_layout(title="Unsupported array shape")
    fig.update_layout(template="plotly_white", height=460, margin=dict(l=20, r=20, t=60, b=40))
    return fig
