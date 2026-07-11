import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ─── Metrics ─────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Directional accuracy
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 2 else 0.0
    return {
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "Acurácia Direcional": round(dir_acc, 4),
        "Correlação IC": round(corr, 4),
    }


# ─── Time-Series Cross Validation ─────────────────────────────────────────────

def ts_cross_validate(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        m = _metrics(y_val.values, preds)
        m["Fold"] = fold
        rows.append(m)
    return pd.DataFrame(rows).set_index("Fold")


# ─── Random Forest ────────────────────────────────────────────────────────────

def run_random_forest(X_train, y_train, X_test, y_test,
                      n_estimators: int = 200, max_depth: int = 5,
                      n_cv_splits: int = 5) -> dict:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )

    # CV
    cv_results = ts_cross_validate(model, X_train, y_train, n_splits=n_cv_splits)

    # Final fit
    model.fit(X_train, y_train)
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    feature_imp = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    return {
        "model": model,
        "preds_test": preds_test,
        "preds_train": preds_train,
        "test_metrics": _metrics(y_test.values, preds_test),
        "train_metrics": _metrics(y_train.values, preds_train),
        "cv_results": cv_results,
        "feature_importance": feature_imp,
        "y_test": y_test,
        "y_train": y_train,
        "X_test": X_test,
    }


# ─── XGBoost ──────────────────────────────────────────────────────────────────

def run_xgboost(X_train, y_train, X_test, y_test,
                n_estimators: int = 200, max_depth: int = 4,
                learning_rate: float = 0.05, n_cv_splits: int = 5) -> dict | None:
    if not HAS_XGB:
        return None

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    cv_results = ts_cross_validate(model, X_train, y_train, n_splits=n_cv_splits)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    feature_imp = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    return {
        "model": model,
        "preds_test": preds_test,
        "preds_train": preds_train,
        "test_metrics": _metrics(y_test.values, preds_test),
        "train_metrics": _metrics(y_train.values, preds_train),
        "cv_results": cv_results,
        "feature_importance": feature_imp,
        "y_test": y_test,
        "y_train": y_train,
        "X_test": X_test,
    }


# ─── LightGBM ────────────────────────────────────────────────────────────────

def run_lightgbm(X_train, y_train, X_test, y_test,
                 n_estimators: int = 200, max_depth: int = 4,
                 learning_rate: float = 0.05, n_cv_splits: int = 5) -> dict | None:
    if not HAS_LGB:
        return None

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    cv_results = ts_cross_validate(model, X_train, y_train, n_splits=n_cv_splits)
    model.fit(X_train, y_train)
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    feature_imp = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    return {
        "model": model,
        "preds_test": preds_test,
        "preds_train": preds_train,
        "test_metrics": _metrics(y_test.values, preds_test),
        "train_metrics": _metrics(y_train.values, preds_train),
        "cv_results": cv_results,
        "feature_importance": feature_imp,
        "y_test": y_test,
        "y_train": y_train,
        "X_test": X_test,
    }
