from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.eda import detect_column_types, infer_task_type, target_quality_check


@dataclass
class ModelConfig:
    target_col: str
    task_type: str
    model_name: str
    test_size: float = 0.2
    random_state: int = 42
    scale_numeric: bool = True
    exclude_datetime: bool = True
    rare_min_frequency: float = 0.01
    rare_max_categories: int = 30
    use_cv: bool = False
    cv_folds: int = 5
    use_search: bool = False
    search_type: str = "random"
    n_iter: int = 12

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelResult:
    pipeline: Pipeline
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame | None
    task_type: str
    model_name: str
    used_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_rows: int
    cv_summary: pd.DataFrame | None = None
    search_summary: pd.DataFrame | None = None
    best_params: dict[str, Any] | None = None
    X_test_raw: pd.DataFrame | None = None
    y_test: pd.Series | None = None
    label_classes: list[str] | None = None


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Collapse rare/high-cardinality categories before OneHotEncoder.

    Works on pandas DataFrames or numpy arrays. For each column, it keeps the most
    frequent categories while replacing long-tail values with __OTHER__.
    """

    def __init__(self, min_frequency: float = 0.01, max_categories: int = 30, other_label: str = "__OTHER__"):
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.other_label = other_label
        self.keep_values_: dict[str, set[str]] = {}
        self.feature_names_in_: list[str] = []

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        self.keep_values_ = {}
        n = max(len(X_df), 1)
        for col in X_df.columns:
            counts = X_df[col].astype("object").fillna("__MISSING__").astype(str).value_counts()
            min_count = max(1, int(np.ceil(self.min_frequency * n)))
            keep = counts[counts >= min_count].head(self.max_categories).index.astype(str).tolist()
            if not keep:
                keep = counts.head(min(self.max_categories, len(counts))).index.astype(str).tolist()
            self.keep_values_[col] = set(keep)
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        out = pd.DataFrame(index=X_df.index)
        for col in X_df.columns:
            values = X_df[col].astype("object").fillna("__MISSING__").astype(str)
            keep = self.keep_values_.get(col, set())
            out[col] = values.where(values.isin(keep), self.other_label)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)

    def _to_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        names = self.feature_names_in_ if self.feature_names_in_ else [f"cat_{i}" for i in range(np.asarray(X).shape[1])]
        return pd.DataFrame(X, columns=names)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _optional_xgb_classifier(random_state: int):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ImportError("XGBoost 未安装。请运行：pip install xgboost") from exc
    return XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def _optional_xgb_regressor(random_state: int):
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise ImportError("XGBoost 未安装。请运行：pip install xgboost") from exc
    return XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )


def _optional_lgbm_classifier(random_state: int):
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise ImportError("LightGBM 未安装。请运行：pip install lightgbm") from exc
    return LGBMClassifier(n_estimators=140, learning_rate=0.06, random_state=random_state, n_jobs=-1, verbose=-1)


def _optional_lgbm_regressor(random_state: int):
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        raise ImportError("LightGBM 未安装。请运行：pip install lightgbm") from exc
    return LGBMRegressor(n_estimators=140, learning_rate=0.06, random_state=random_state, n_jobs=-1, verbose=-1)


def get_available_models(task_type: str) -> list[str]:
    if task_type == "classification":
        return ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]
    return ["Linear Regression", "Ridge Regression", "Random Forest Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor"]


def get_model(task_type: str, model_name: str, random_state: int):
    if task_type == "classification":
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000)
        if model_name == "Random Forest":
            return RandomForestClassifier(n_estimators=80, random_state=random_state, n_jobs=-1, class_weight="balanced")
        if model_name == "Gradient Boosting":
            return GradientBoostingClassifier(random_state=random_state)
        if model_name == "XGBoost":
            return _optional_xgb_classifier(random_state)
        if model_name == "LightGBM":
            return _optional_lgbm_classifier(random_state)
    else:
        if model_name == "Linear Regression":
            return LinearRegression()
        if model_name == "Ridge Regression":
            return Ridge(random_state=random_state)
        if model_name == "Random Forest Regressor":
            return RandomForestRegressor(n_estimators=80, random_state=random_state, n_jobs=-1)
        if model_name == "Gradient Boosting Regressor":
            return GradientBoostingRegressor(random_state=random_state)
        if model_name == "XGBoost Regressor":
            return _optional_xgb_regressor(random_state)
        if model_name == "LightGBM Regressor":
            return _optional_lgbm_regressor(random_state)
    raise ValueError(f"未知模型：{model_name}")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
    rare_min_frequency: float = 0.01,
    rare_max_categories: int = 30,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rare", RareCategoryGrouper(min_frequency=rare_min_frequency, max_categories=rare_max_categories)),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("没有可用于建模的特征列。请至少保留一个数值或类别特征。")

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def _prepare_features(df: pd.DataFrame, config: ModelConfig) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], int]:
    ok, messages = target_quality_check(df, config.target_col)
    if not ok:
        raise ValueError("；".join(messages))

    before = len(df)
    clean_df = df.dropna(subset=[config.target_col]).copy()
    dropped_rows = before - len(clean_df)

    column_types = detect_column_types(clean_df.drop(columns=[config.target_col]))
    numeric_features = [c for c in column_types.numeric if c != config.target_col]
    categorical_features = [c for c in column_types.categorical if c != config.target_col]

    if config.exclude_datetime:
        excluded = set(column_types.datetime)
        numeric_features = [c for c in numeric_features if c not in excluded]
        categorical_features = [c for c in categorical_features if c not in excluded]

    X = clean_df[numeric_features + categorical_features].copy()
    y = clean_df[config.target_col].copy()

    if config.task_type == "regression":
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid]
        dropped_rows += int((~valid).sum())
        if len(y) < 10:
            raise ValueError("回归任务中 target 转为数值后有效样本少于 10，无法可靠建模。")

    return X, y, numeric_features, categorical_features, dropped_rows


def _classification_metrics(y_true, y_pred, y_proba=None, labels: list[str] | None = None) -> dict[str, Any]:
    labels = labels or sorted(pd.Series(y_true).dropna().unique().tolist(), key=lambda x: str(x))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["labels"] = labels
    metrics["confusion_matrix"] = pd.DataFrame(cm, index=labels, columns=labels)

    if y_proba is not None and len(labels) == 2:
        try:
            positive_label = labels[1]
            y_binary = (pd.Series(y_true).astype(str).values == str(positive_label)).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(y_binary, y_proba[:, 1]))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _can_stratify(y: pd.Series | np.ndarray) -> bool:
    counts = pd.Series(y).value_counts(dropna=False)
    return bool(len(counts) >= 2 and counts.min() >= 2)


def _safe_cv_folds(y, requested: int, task_type: str) -> int:
    requested = int(max(2, min(requested, 10)))
    if task_type == "classification":
        min_class = int(pd.Series(y).value_counts().min())
        return max(2, min(requested, min_class))
    return max(2, min(requested, len(y) // 5 if len(y) >= 10 else 2))


def _get_feature_names(pipeline: Pipeline, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    names: list[str] = []
    if numeric_features and "num" in preprocessor.named_transformers_:
        names.extend([str(c) for c in numeric_features])
    if categorical_features and "cat" in preprocessor.named_transformers_:
        cat_pipeline = preprocessor.named_transformers_.get("cat")
        encoder = cat_pipeline.named_steps.get("onehot") if cat_pipeline is not None else None
        if encoder is not None:
            try:
                names.extend([str(n) for n in encoder.get_feature_names_out(categorical_features)])
            except Exception:
                try:
                    names.extend([str(n) for n in encoder.get_feature_names_out()])
                except Exception:
                    names.extend([str(c) for c in categorical_features])
    if names:
        return names
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def extract_feature_importance(
    pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    feature_names = _get_feature_names(pipeline, numeric_features, categorical_features)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None

    if len(values) != len(feature_names):
        return None
    importance_df = pd.DataFrame({"feature": feature_names, "importance": values})
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def get_param_grid(model_name: str, task_type: str) -> dict[str, list[Any]]:
    if model_name in {"Random Forest", "Random Forest Regressor"}:
        return {
            "model__n_estimators": [80, 120, 180],
            "model__max_depth": [None, 3, 5, 8],
            "model__min_samples_leaf": [1, 2, 4],
        }
    if model_name in {"Gradient Boosting", "Gradient Boosting Regressor"}:
        return {
            "model__n_estimators": [60, 100, 140],
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__max_depth": [2, 3, 4],
        }
    if model_name == "Logistic Regression":
        return {"model__C": [0.1, 0.5, 1.0, 2.0, 5.0]}
    if model_name == "Ridge Regression":
        return {"model__alpha": [0.1, 1.0, 5.0, 10.0]}
    if "XGBoost" in model_name:
        return {
            "model__n_estimators": [80, 120, 160],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__subsample": [0.8, 1.0],
        }
    if "LightGBM" in model_name:
        return {
            "model__n_estimators": [80, 140, 200],
            "model__num_leaves": [15, 31, 63],
            "model__learning_rate": [0.03, 0.06, 0.1],
        }
    return {}


def _cv_summary(pipeline: Pipeline, X: pd.DataFrame, y, task_type: str, folds: int) -> pd.DataFrame | None:
    scoring = ["accuracy", "f1_weighted"] if task_type == "classification" else ["r2", "neg_mean_absolute_error"]
    try:
        result = cross_validate(pipeline, X, y, cv=folds, scoring=scoring, n_jobs=1, error_score="raise")
    except Exception:
        return None
    rows = []
    for key, values in result.items():
        if key.startswith("test_"):
            name = key.replace("test_", "")
            vals = -values if name.startswith("neg_") else values
            name = name.replace("neg_", "")
            rows.append({"metric": name, "mean": float(np.mean(vals)), "std": float(np.std(vals))})
    return pd.DataFrame(rows)


def _search_fit(pipeline: Pipeline, X_train, y_train, config: ModelConfig, cv_folds: int):
    grid = get_param_grid(config.model_name, config.task_type)
    if not grid:
        return pipeline.fit(X_train, y_train), None, None
    scoring = "f1_weighted" if config.task_type == "classification" else "r2"
    if config.search_type == "grid":
        search = GridSearchCV(pipeline, grid, scoring=scoring, cv=cv_folds, n_jobs=1, error_score="raise")
    else:
        search = RandomizedSearchCV(
            pipeline,
            grid,
            n_iter=min(config.n_iter, np.prod([len(v) for v in grid.values()])),
            scoring=scoring,
            cv=cv_folds,
            random_state=config.random_state,
            n_jobs=1,
            error_score="raise",
        )
    search.fit(X_train, y_train)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    keep_cols = [c for c in ["rank_test_score", "mean_test_score", "std_test_score", "params"] if c in cv_results.columns]
    return search.best_estimator_, cv_results[keep_cols].head(20).reset_index(drop=True), search.best_params_


def train_model(df: pd.DataFrame, config: ModelConfig) -> ModelResult:
    X, y_raw, numeric_features, categorical_features, dropped_rows = _prepare_features(df, config)

    if len(X) < 10:
        raise ValueError("有效样本少于 10，无法进行 train/test split。")

    label_encoder: LabelEncoder | None = None
    label_classes: list[str] | None = None
    if config.task_type == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        label_classes = [str(c) for c in label_encoder.classes_.tolist()]
    else:
        y = y_raw

    test_size = min(max(config.test_size, 0.1), 0.5)
    stratify = y if config.task_type == "classification" and _can_stratify(y) else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=config.random_state)

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        config.scale_numeric,
        rare_min_frequency=config.rare_min_frequency,
        rare_max_categories=config.rare_max_categories,
    )
    model = get_model(config.task_type, config.model_name, config.random_state)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    cv_folds = _safe_cv_folds(y_train, config.cv_folds, config.task_type)

    search_summary = None
    best_params = None
    if config.use_search:
        pipeline, search_summary, best_params = _search_fit(pipeline, X_train, y_train, config, cv_folds)
    else:
        pipeline.fit(X_train, y_train)

    cv_summary = _cv_summary(pipeline, X, y, config.task_type, cv_folds) if config.use_cv else None

    y_pred_model = pipeline.predict(X_test)
    if config.task_type == "classification" and label_encoder is not None:
        y_test_labels = pd.Series(label_encoder.inverse_transform(np.asarray(y_test).astype(int)), name=config.target_col)
        y_pred_labels = label_encoder.inverse_transform(np.asarray(y_pred_model).astype(int))
        predictions = X_test.copy()
        predictions.insert(0, "y_true", y_test_labels.values)
        predictions.insert(1, "y_pred", y_pred_labels)
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    predictions.insert(2, "positive_class_probability", y_proba[:, 1])
            except Exception:
                y_proba = None
        metrics = _classification_metrics(y_test_labels, y_pred_labels, y_proba, labels=label_classes)
    else:
        y_test_series = pd.Series(y_test, name=config.target_col)
        predictions = X_test.copy()
        predictions.insert(0, "y_true", y_test_series.values)
        predictions.insert(1, "y_pred", y_pred_model)
        metrics = _regression_metrics(y_test_series, y_pred_model)

    feature_importance = extract_feature_importance(pipeline, numeric_features, categorical_features)
    return ModelResult(
        pipeline=pipeline,
        metrics=metrics,
        predictions=predictions.reset_index(drop=True),
        feature_importance=feature_importance,
        task_type=config.task_type,
        model_name=config.model_name,
        used_features=numeric_features + categorical_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dropped_rows=dropped_rows,
        cv_summary=cv_summary,
        search_summary=search_summary,
        best_params=best_params,
        X_test_raw=X_test.reset_index(drop=True),
        y_test=pd.Series(y_test).reset_index(drop=True),
        label_classes=label_classes,
    )


def infer_task_for_target(df: pd.DataFrame, target_col: str) -> str:
    if target_col not in df.columns:
        raise ValueError("目标列不存在。")
    return infer_task_type(df[target_col])
