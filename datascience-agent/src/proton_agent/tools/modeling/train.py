"""
Train Model Tool: automated feature preprocessing, train/test splitting, model fitting, evaluation,
and Optuna Bayesian hyperparameter optimization with RandomizedSearchCV fallback.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult
from proton_agent.tools.modeling.detect import infer_problem_type, infer_target_column


class LabelEncodedClassifierWrapper:
    """Wrapper that ensures classifiers (like XGBoost) that require 0..K-1 integers handle arbitrary classes."""
    def __init__(self, base_estimator: Any):
        self.base_estimator = base_estimator
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X: Any, y: Any, **kwargs: Any) -> LabelEncodedClassifierWrapper:
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.base_estimator.fit(X, y_enc, **kwargs)
        return self

    def predict(self, X: Any) -> np.ndarray:
        preds = self.base_estimator.predict(X)
        return self.label_encoder.inverse_transform(np.asarray(preds, dtype=int))

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(self.base_estimator, "predict_proba"):
            return self.base_estimator.predict_proba(X)
        raise AttributeError("Underlying estimator does not support predict_proba")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self.base_estimator.get_params(deep=deep)

    def set_params(self, **params: Any) -> LabelEncodedClassifierWrapper:
        self.base_estimator.set_params(**params)
        return self

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return getattr(self.base_estimator, "feature_importances_", None)


class TrainModelTool(BaseTool):
    """Trains a machine learning model (classification or regression) with automated preprocessing and Optuna Bayesian tuning."""

    name: str = "train_model"
    description: str = (
        "Train machine learning models (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, "
        "Extra Trees, AdaBoost, Hist Gradient Boosting, Decision Tree, Logistic/Linear Regression, Ridge, Lasso, KNN, SVM, Naive Bayes, MLP Neural Net)."
    )

    def execute(
        self,
        session: SessionState,
        target_column: str | None = None,
        model_type: str = "random_forest",
        tune: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df.copy()

        # Deduplicate dataset before train-test split to prevent train-test data leakage
        dup_count = int(df.duplicated().sum())
        if dup_count > 0 and (len(df) - dup_count) >= 15:
            df = df.drop_duplicates().reset_index(drop=True)

        if target_column and target_column in df.columns:
            target = target_column

        elif session.target_column and session.target_column in df.columns:
            target = session.target_column
        else:
            target = infer_target_column(df)

        if not target or target not in df.columns:
            return ToolResult(
                success=False,
                data={},
                error="No valid target column found in dataset.",
            )

        target_column = target
        session.target_column = target_column

        # Separate features and target
        y_raw = df[target_column]
        X_raw = df.drop(columns=[target_column])

        # Exclude ID/index/URL columns from features to prevent data leakage and noise
        id_cols = [
            c for c in X_raw.columns
            if c.lower() in ("id", "index", "unnamed: 0", "row_id", "rowid", "patient_id", "customer_id", "user_id", "url", "link", "guid", "href")
            or any(c.lower().endswith(sfx) for sfx in ("_id", "_url", "_link", "_guid", "_href"))
            or (X_raw[c].dtype in ("int64", "float64") and X_raw[c].nunique() == len(X_raw) and (X_raw[c] == np.arange(1, len(X_raw) + 1)).all())
        ]
        if id_cols and len(X_raw.columns) > len(id_cols):
            X_raw = X_raw.drop(columns=id_cols)

        # Identify and exclude raw free-form text paragraphs and high-cardinality non-categorical text
        text_cols_to_drop = []
        cat_cols_to_encode = []
        for col in X_raw.select_dtypes(include=["object", "category", "string"]).columns:
            avg_len = X_raw[col].dropna().astype(str).str.len().mean() if not X_raw[col].dropna().empty else 0
            nunique = X_raw[col].nunique()
            col_l = col.lower()
            if avg_len > 40 or nunique > 50 or "excerpt" in col_l or "cleaned_text" in col_l or "normalized_text" in col_l or "source_url" in col_l:
                text_cols_to_drop.append(col)
            else:
                cat_cols_to_encode.append(col)

        if text_cols_to_drop:
            X_raw = X_raw.drop(columns=[c for c in text_cols_to_drop if c in X_raw.columns])

        if X_raw.empty:
            return ToolResult(
                success=False,
                data={},
                error="No numerical or structured categorical feature columns available to train model.",
            )

        # Preprocessing: Impute missing values
        for col in X_raw.columns:
            if X_raw[col].dtype in ("int64", "float64", "number") or pd.api.types.is_numeric_dtype(X_raw[col]):
                X_raw[col] = X_raw[col].fillna(X_raw[col].median() if not X_raw[col].isnull().all() else 0.0)
            else:
                mode_val = X_raw[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Missing"
                X_raw[col] = X_raw[col].fillna(fill_val)

        # Encode categorical features
        cols_to_dummy = [c for c in cat_cols_to_encode if c in X_raw.columns]
        if cols_to_dummy:
            X_encoded = pd.get_dummies(X_raw, columns=cols_to_dummy, drop_first=True, dtype=float)
        else:
            X_encoded = X_raw.copy()

        # Handle target missing values
        valid_mask = y_raw.notnull()
        X_encoded = X_encoded[valid_mask]
        y_raw = y_raw[valid_mask]

        # Determine problem type accurately
        problem_type, _ = infer_problem_type(y_raw)
        is_classification = problem_type == "classification"

        if is_classification:
            y = y_raw.astype(str)
        else:
            y_num = pd.to_numeric(y_raw, errors="coerce")
            if y_num.isnull().all():
                problem_type = "classification"
                is_classification = True
                y = y_raw.astype(str)
            else:
                valid_num_mask = y_num.notnull()
                y = y_num[valid_num_mask]
                X_encoded = X_encoded[valid_num_mask]

        # Train / Test split
        try:
            strat = y if is_classification and len(y.unique()) > 1 and (y.value_counts() >= 2).all() else None
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42, stratify=strat
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )

        # ── Comprehensive Model Instantiation ────────────────────────────────
        model_key = model_type.lower().replace(" ", "_").replace("-", "_")

        if is_classification:
            if "xgboost" in model_key or "xgb" in model_key:
                try:
                    import xgboost as xgb
                    base = xgb.XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)
                    model = LabelEncodedClassifierWrapper(base)
                    model_name = "XGBoost Classifier"
                except ImportError:
                    model = GradientBoostingClassifier(random_state=42)
                    model_name = "Gradient Boosting Classifier (XGBoost fallback)"
            elif "lightgbm" in model_key or "lgbm" in model_key or "light_gbm" in model_key:
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                    model_name = "LightGBM Classifier"
                except ImportError:
                    model = HistGradientBoostingClassifier(random_state=42)
                    model_name = "Hist Gradient Boosting Classifier (LightGBM fallback)"
            elif "catboost" in model_key or "cat_boost" in model_key or "cb" in model_key:
                try:
                    import catboost as cb
                    model = cb.CatBoostClassifier(random_state=42, verbose=0)
                    model_name = "CatBoost Classifier"
                except ImportError:
                    model = GradientBoostingClassifier(random_state=42)
                    model_name = "Gradient Boosting Classifier (CatBoost fallback)"
            elif "extra_tree" in model_key:
                model = ExtraTreesClassifier(n_estimators=100, random_state=42)
                model_name = "Extra Trees Classifier"
            elif "adaboost" in model_key:
                model = AdaBoostClassifier(random_state=42)
                model_name = "AdaBoost Classifier"
            elif "hist_gradient" in model_key or "hist" in model_key:
                model = HistGradientBoostingClassifier(random_state=42)
                model_name = "Hist Gradient Boosting Classifier"
            elif "gradient" in model_key or "boosting" in model_key:
                model = GradientBoostingClassifier(random_state=42)
                model_name = "Gradient Boosting Classifier"
            elif "decision_tree" in model_key or ("tree" in model_key and "forest" not in model_key):
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree Classifier"
            elif "ridge" in model_key:
                model = RidgeClassifier(random_state=42)
                model_name = "Ridge Classifier"
            elif "lasso" in model_key:
                model = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
                model_name = "Lasso Logistic Classifier"
            elif "knn" in model_key or "neighbor" in model_key:
                model = KNeighborsClassifier(n_neighbors=5)
                model_name = "K-Nearest Neighbors (KNN)"
            elif "svm" in model_key or "svc" in model_key:
                model = SVC(probability=True, random_state=42)
                model_name = "Support Vector Machine (SVC)"
            elif "naive" in model_key or "bayes" in model_key:
                model = GaussianNB()
                model_name = "Gaussian Naive Bayes"
            elif "mlp" in model_key or "neural" in model_key:
                model = MLPClassifier(max_iter=500, random_state=42)
                model_name = "MLP Neural Network"
            elif "logistic" in model_key or "linear" in model_key:
                model = LogisticRegression(max_iter=1000, random_state=42)
                model_name = "Logistic Regression"
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model_name = "Random Forest Classifier"
        else:
            if "xgboost" in model_key or "xgb" in model_key:
                try:
                    import xgboost as xgb
                    model = xgb.XGBRegressor(random_state=42, verbosity=0)
                    model_name = "XGBoost Regressor"
                except ImportError:
                    model = GradientBoostingRegressor(random_state=42)
                    model_name = "Gradient Boosting Regressor (XGBoost fallback)"
            elif "lightgbm" in model_key or "lgbm" in model_key or "light_gbm" in model_key:
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMRegressor(random_state=42, verbose=-1)
                    model_name = "LightGBM Regressor"
                except ImportError:
                    model = HistGradientBoostingRegressor(random_state=42)
                    model_name = "Hist Gradient Boosting Regressor (LightGBM fallback)"
            elif "catboost" in model_key or "cat_boost" in model_key or "cb" in model_key:
                try:
                    import catboost as cb
                    model = cb.CatBoostRegressor(random_state=42, verbose=0)
                    model_name = "CatBoost Regressor"
                except ImportError:
                    model = GradientBoostingRegressor(random_state=42)
                    model_name = "Gradient Boosting Regressor (CatBoost fallback)"
            elif "extra_tree" in model_key:
                model = ExtraTreesRegressor(n_estimators=100, random_state=42)
                model_name = "Extra Trees Regressor"
            elif "adaboost" in model_key:
                model = AdaBoostRegressor(random_state=42)
                model_name = "AdaBoost Regressor"
            elif "hist_gradient" in model_key or "hist" in model_key:
                model = HistGradientBoostingRegressor(random_state=42)
                model_name = "Hist Gradient Boosting Regressor"
            elif "gradient" in model_key or "boosting" in model_key:
                model = GradientBoostingRegressor(random_state=42)
                model_name = "Gradient Boosting Regressor"
            elif "decision_tree" in model_key or ("tree" in model_key and "forest" not in model_key):
                model = DecisionTreeRegressor(random_state=42)
                model_name = "Decision Tree Regressor"
            elif "ridge" in model_key:
                model = Ridge(random_state=42)
                model_name = "Ridge Regressor"
            elif "lasso" in model_key:
                model = Lasso(random_state=42)
                model_name = "Lasso Regressor"
            elif "knn" in model_key or "neighbor" in model_key:
                model = KNeighborsRegressor(n_neighbors=5)
                model_name = "K-Nearest Neighbors Regressor (KNN)"
            elif "svm" in model_key or "svr" in model_key:
                model = SVR()
                model_name = "Support Vector Regressor (SVR)"
            elif "naive" in model_key or "bayes" in model_key:
                model = BayesianRidge()
                model_name = "Bayesian Ridge Regressor"
            elif "mlp" in model_key or "neural" in model_key:
                model = MLPRegressor(max_iter=500, random_state=42)
                model_name = "MLP Neural Network Regressor"
            elif "linear" in model_key or ("regression" in model_key and "forest" not in model_key and "tree" not in model_key):
                model = LinearRegression()
                model_name = "Linear Regression"
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model_name = "Random Forest Regressor"

        # ── Model Training & Optuna Tuning ───────────────────────────────────
        best_params: dict[str, Any] = {}
        tuning_engine: Optional[str] = None
        trials_run: int = 0
        tuning_rationale: Optional[str] = None
        baseline_metric_val: Optional[float] = None

        if tune:
            # First calculate baseline score before tuning
            try:
                base_model = clone(model) if hasattr(model, "get_params") else model
                base_model.fit(X_train, y_train)
                base_pred = base_model.predict(X_test)
                if is_classification:
                    baseline_metric_val = float(f1_score(y_test, base_pred, average="weighted", zero_division=0))
                else:
                    baseline_metric_val = float(r2_score(y_test, base_pred))
            except Exception:
                baseline_metric_val = None

            optuna_succeeded = False
            try:
                import optuna  # type: ignore
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                underlying_obj = getattr(model, "base_estimator", model)
                cls_name = type(underlying_obj).__name__

                def objective(trial: optuna.Trial) -> float:
                    params: dict[str, Any] = {}
                    if "RandomForest" in cls_name or "ExtraTrees" in cls_name:
                        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200, step=25)
                        params["max_depth"] = trial.suggest_int("max_depth", 3, 20)
                        params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 8)
                    elif "GradientBoosting" in cls_name:
                        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 150, step=25)
                        params["max_depth"] = trial.suggest_int("max_depth", 3, 6)
                        params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
                    elif "XGB" in cls_name:
                        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 150, step=25)
                        params["max_depth"] = trial.suggest_int("max_depth", 3, 6)
                        params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
                    elif "LGBM" in cls_name:
                        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 150, step=25)
                        params["max_depth"] = trial.suggest_int("max_depth", 3, 6)
                        params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
                        params["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
                    elif "CatBoost" in cls_name:
                        params["iterations"] = trial.suggest_int("iterations", 50, 150, step=25)
                        params["depth"] = trial.suggest_int("depth", 3, 6)
                        params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
                    elif "DecisionTree" in cls_name:
                        params["max_depth"] = trial.suggest_int("max_depth", 3, 15)
                        params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 8)
                    elif "LogisticRegression" in cls_name:
                        params["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)
                    elif "Ridge" in cls_name or "Lasso" in cls_name:
                        params["alpha"] = trial.suggest_float("alpha", 0.01, 10.0, log=True)
                    elif "SVC" in cls_name or "SVR" in cls_name:
                        params["C"] = trial.suggest_float("C", 0.1, 10.0, log=True)
                        params["kernel"] = trial.suggest_categorical("kernel", ["linear", "rbf"])
                    elif "KNeighbors" in cls_name:
                        params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 9, step=2)

                    try:
                        t_model = clone(model)
                        if hasattr(t_model, "base_estimator"):
                            t_model.base_estimator.set_params(**params)
                        else:
                            t_model.set_params(**params)

                        # CV evaluation
                        cv_folds = 3 if len(X_train) >= 15 else 2
                        cv_scoring = "f1_weighted" if is_classification else "r2"
                        scores = cross_val_score(t_model, X_train, y_train, cv=cv_folds, scoring=cv_scoring)
                        return float(scores.mean())
                    except Exception:
                        return -999.0

                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=42),
                )
                study.optimize(objective, n_trials=15, timeout=25)

                if len(study.trials) > 0 and study.best_value > -900:
                    best_params = study.best_params
                    trials_run = len(study.trials)
                    tuning_engine = "Optuna (TPE/Bayesian Optimization)"

                    if hasattr(model, "base_estimator"):
                        model.base_estimator.set_params(**best_params)
                    elif hasattr(model, "set_params"):
                        model.set_params(**best_params)

                    model.fit(X_train, y_train)
                    optuna_succeeded = True
            except Exception:
                optuna_succeeded = False

            if not optuna_succeeded:
                # Fallback to RandomizedSearchCV if Optuna is unavailable or failed
                param_grids: dict[str, dict[str, list[Any]]] = {
                    "RandomForestClassifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5]},
                    "RandomForestRegressor": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5]},
                    "ExtraTreesClassifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
                    "ExtraTreesRegressor": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
                    "GradientBoostingClassifier": {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1, 0.2]},
                    "GradientBoostingRegressor": {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1, 0.2]},
                    "HistGradientBoostingClassifier": {"max_iter": [50, 100], "learning_rate": [0.05, 0.1, 0.2]},
                    "HistGradientBoostingRegressor": {"max_iter": [50, 100], "learning_rate": [0.05, 0.1, 0.2]},
                    "DecisionTreeClassifier": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
                    "DecisionTreeRegressor": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
                    "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
                    "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                    "Lasso": {"alpha": [0.01, 0.1, 1.0]},
                }
                underlying_obj = getattr(model, "base_estimator", model)
                cls_name = type(underlying_obj).__name__
                p_grid = param_grids.get(cls_name, {})
                if p_grid:
                    try:
                        search = RandomizedSearchCV(
                            model, p_grid, n_iter=6, cv=3, random_state=42, n_jobs=-1,
                            scoring="f1_weighted" if is_classification else "r2",
                            error_score="raise",
                        )
                        search.fit(X_train, y_train)
                        model = search.best_estimator_
                        best_params = search.best_params_
                        trials_run = 6
                        tuning_engine = "RandomizedSearchCV (Fallback)"
                    except Exception:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # ── Metrics Computation ──────────────────────────────────────────────
        metrics = {}
        if is_classification:
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics = {
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }
            tuned_score = f1
            metric_label = "F1 Score"
        else:
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))
            metrics = {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "r2_score": round(r2, 4),
            }
            tuned_score = r2
            metric_label = "R² Score"

        # Construct plain-language tuning rationale referencing actual before/after metrics
        if tune:
            engine_name = tuning_engine or "Hyperparameter optimization"
            if baseline_metric_val is not None:
                delta = tuned_score - baseline_metric_val
                if delta > 0.0001:
                    tuning_rationale = (
                        f"{engine_name} explored {trials_run} trials on {model_name}. "
                        f"Tuning improved {metric_label} from {baseline_metric_val:.4f} to {tuned_score:.4f} (+{delta:.4f}) "
                        f"with best parameters: {best_params}."
                    )
                else:
                    tuning_rationale = (
                        f"{engine_name} explored {trials_run} trials on {model_name}. "
                        f"Model achieved {metric_label}: {tuned_score:.4f} (baseline: {baseline_metric_val:.4f}) "
                        f"with best parameters: {best_params}."
                    )
            else:
                tuning_rationale = (
                    f"{engine_name} explored {trials_run} trials on {model_name}. "
                    f"Achieved {metric_label}: {tuned_score:.4f} with best parameters: {best_params}."
                )

        # Feature importances if available
        feature_importances = {}
        underlying = getattr(model, "base_estimator", model)
        if hasattr(underlying, "feature_importances_") and underlying.feature_importances_ is not None:
            imps = underlying.feature_importances_
            feat_imp_tuples = sorted(zip(X_encoded.columns, imps), key=lambda x: x[1], reverse=True)
            feature_importances = {k: round(float(v), 4) for k, v in feat_imp_tuples[:10]}

        model_record = {
            "name": model_name,
            "problem_type": problem_type,
            "target_column": target_column,
            "model": model,
            "feature_names": list(X_encoded.columns),
            "metrics": metrics,
            "feature_importances": feature_importances,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        session.current_model = model_record
        session.models[model_name] = model_record

        # Log experiment run
        session.experiments.log_run(
            dataset_name=dataset.name,
            dataset_rows=dataset.rows,
            dataset_cols=dataset.columns_count,
            target_column=target_column,
            feature_names=list(X_encoded.columns),
            model_name=model_name,
            problem_type=problem_type,
            metrics=metrics,
        )

        return ToolResult(
            success=True,
            data={
                "model_name": model_name,
                "problem_type": problem_type,
                "target_column": target_column,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "metrics": metrics,
                "feature_importances": feature_importances,
                "tuned": tune,
                "tuning_engine": tuning_engine,
                "trials_run": trials_run,
                "best_params": best_params,
                "tuning_rationale": tuning_rationale,
                "message": (
                    f"Successfully trained {model_name} on target '{target_column}'."
                    + (f" {tuning_rationale}" if tune and tuning_rationale else "")
                ),
            },
        )
