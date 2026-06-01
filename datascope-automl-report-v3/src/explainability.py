from __future__ import annotations

import numpy as np
import pandas as pd


def _feature_names_from_result(model_result) -> list[str]:
    preprocessor = model_result.pipeline.named_steps["preprocessor"]
    names: list[str] = []
    numeric = getattr(model_result, "numeric_features", []) or []
    categorical = getattr(model_result, "categorical_features", []) or []
    if numeric and "num" in preprocessor.named_transformers_:
        names.extend([str(c) for c in numeric])
    if categorical and "cat" in preprocessor.named_transformers_:
        cat_pipeline = preprocessor.named_transformers_.get("cat")
        encoder = cat_pipeline.named_steps.get("onehot") if cat_pipeline is not None else None
        if encoder is not None:
            try:
                names.extend([str(n) for n in encoder.get_feature_names_out(categorical)])
            except Exception:
                try:
                    names.extend([str(n) for n in encoder.get_feature_names_out()])
                except Exception:
                    names.extend([str(c) for c in categorical])
    if names:
        return names
    try:
        return [str(x) for x in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def compute_shap_importance(model_result, max_samples: int = 120) -> tuple[pd.DataFrame | None, str | None]:
    """Return mean absolute SHAP importance for the trained sklearn pipeline."""
    try:
        import shap
    except Exception as exc:
        return None, f"SHAP 未安装或无法导入：{exc}。请运行 pip install shap。"

    if model_result is None or model_result.X_test_raw is None:
        return None, "没有可解释的模型结果。请先完成建模。"

    pipeline = model_result.pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_raw = model_result.X_test_raw.head(max_samples)

    try:
        X_trans = preprocessor.transform(X_raw)
        feature_names = _feature_names_from_result(model_result)
        if len(feature_names) != X_trans.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]
        X_frame = pd.DataFrame(X_trans, columns=feature_names)
    except Exception as exc:
        return None, f"预处理后的特征矩阵无法转换为 SHAP 输入：{exc}"

    try:
        explainer = shap.Explainer(model, X_frame)
        shap_values = explainer(X_frame)
        values = shap_values.values
        if values.ndim == 3:
            mean_abs = np.mean(np.abs(values), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(values), axis=0)
    except Exception:
        try:
            explainer = shap.Explainer(pipeline.predict, X_raw)
            shap_values = explainer(X_raw)
            values = shap_values.values
            if values.ndim == 3:
                mean_abs = np.mean(np.abs(values), axis=(0, 2))
            else:
                mean_abs = np.mean(np.abs(values), axis=0)
            X_frame = X_raw.reset_index(drop=True)
        except Exception as exc:
            return None, f"SHAP 计算失败：{exc}"

    shap_df = pd.DataFrame({"feature": list(X_frame.columns), "mean_abs_shap": mean_abs})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return shap_df, None
