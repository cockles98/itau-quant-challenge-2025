from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, dump, load

from metrics import sharpe

set_config(assume_finite=True)

@dataclass(frozen=True)
class MetaBlendConfig:
    """Configuration for the meta-blend regularisation model."""

    enabled: bool = False
    model_type: str = "ridge"
    horizon: int = 21
    standardize: bool = True
    use_regime_feature: bool = True
    use_interactions: bool = False
    alpha_grid: Tuple[float, ...] = (0.1, 1.0)
    l1_ratio_grid: Tuple[float, ...] = (0.5,)
    cv_n_splits: int = 5
    save_artifacts: bool = False
    cv_embargo_days: int = 5
    min_train_dates: int = 252
    rolling_window: Optional[int] = None
    cv_metric: str = "ic"
    random_state: Optional[int] = 42
    n_jobs: int = -1

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, object]]) -> "MetaBlendConfig":
        params = dict(raw or {})
        alpha_grid = params.get("alpha_grid")
        if alpha_grid is not None:
            params["alpha_grid"] = tuple(float(x) for x in alpha_grid if float(x) > 0)
        l1_ratio_grid = params.get("l1_ratio_grid")
        if l1_ratio_grid is not None:
            params["l1_ratio_grid"] = tuple(float(x) for x in l1_ratio_grid if 0 <= float(x) <= 1)
        metric = str(params.get("cv_metric", "ic")).lower()
        if metric not in {"ic", "sharpe_pred"}:
            params["cv_metric"] = "ic"
        params["save_artifacts"] = bool(params.get("save_artifacts", False))
        if "n_jobs" in params:
            try:
                params["n_jobs"] = int(params["n_jobs"])
            except (TypeError, ValueError):
                params["n_jobs"] = -1
        return cls(**params)


class MetaBlender:
    """Regularised meta-learner to combine factor signals."""

    def __init__(self, config: MetaBlendConfig):
        self.config = config

    def generate_scores(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        assets: Sequence[str],
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        if features.empty or target.empty:
            return pd.DataFrame(columns=assets), {"error": "empty_features"}

        idx_dates = features.index.get_level_values(0)
        unique_dates = idx_dates.unique().sort_values()
        predictions: Dict[pd.Timestamp, pd.Series] = {}
        meta_history: Dict[str, Dict[str, object]] = {}

        for current_date in unique_dates:
            history_mask = idx_dates < current_date
            if not history_mask.any():
                continue
            hist_features = features[history_mask]
            hist_target = target[history_mask]
            if hist_features.empty:
                continue
            hist_dates = hist_features.index.get_level_values(0).unique().sort_values()
            if len(hist_dates) < self.config.cv_n_splits:
                continue
            if len(hist_dates) < self.config.min_train_dates:
                continue
            if self.config.rolling_window:
                window_dates = hist_dates[-self.config.rolling_window :]
                mask = hist_features.index.get_level_values(0).isin(window_dates)
                hist_features = hist_features[mask]
                hist_target = hist_target[mask]
                hist_dates = window_dates
            selection = self._select_params(hist_features, hist_target, hist_dates)
            if selection is None:
                continue
            best_params, avg_score, fold_scores = selection
            model, scaler = self._fit_model(hist_features, hist_target, best_params)

            current_mask = idx_dates == current_date
            current_features = features[current_mask]
            if current_features.empty:
                continue
            preds = self._predict(model, scaler, current_features)
            predictions[current_date] = pd.Series(
                preds,
                index=current_features.index.get_level_values(1),
                name="meta_score",
            )
            meta_history[current_date.isoformat()] = {
                "alpha": float(best_params["alpha"]),
                "l1_ratio": (
                    float(best_params["l1_ratio"])
                    if "l1_ratio" in best_params
                    else None
                ),
                "avg_cv_score": float(avg_score) if np.isfinite(avg_score) else None,
                "fold_scores": [float(s) for s in fold_scores],
                "train_rows": int(hist_features.shape[0]),
            }

        if not predictions:
            return pd.DataFrame(columns=assets), {"warning": "no_predictions"}

        scores_df = pd.DataFrame.from_dict(predictions, orient="index")
        scores_df = scores_df.sort_index().reindex(columns=list(assets), fill_value=0.0)
        meta = {
            "config": asdict(self.config),
            "per_date": meta_history,
        }
        return scores_df, meta

    def _select_params(
        self,
        hist_features: pd.DataFrame,
        hist_target: pd.Series,
        hist_dates: pd.Index,
    ) -> Optional[Tuple[Dict[str, float], float, List[float]]]:
        cfg = self.config
        from validation.purged_cv import purged_kfold_split

        splits = list(
            purged_kfold_split(
                hist_dates, n_splits=cfg.cv_n_splits, embargo_days=cfg.cv_embargo_days
            )
        )
        param_grid = self._build_param_grid()
        best_score = -np.inf
        best_params: Optional[Dict[str, float]] = None
        best_fold_scores: List[float] = []

        def _score_params(params: Dict[str, float]) -> Tuple[Dict[str, float], Optional[float], List[float]]:
            fold_scores: List[float] = []
            for train_idx, test_idx in splits:
                train_dates = hist_dates[train_idx]
                test_dates = hist_dates[test_idx]
                train_mask = hist_features.index.get_level_values(0).isin(train_dates)
                test_mask = hist_features.index.get_level_values(0).isin(test_dates)
                if not train_mask.any() or not test_mask.any():
                    continue
                X_train = hist_features[train_mask]
                y_train = hist_target[train_mask]
                X_test = hist_features[test_mask]
                y_test = hist_target[test_mask]
                if X_train.empty or X_test.empty:
                    continue
                model, scaler = self._fit_model(X_train, y_train, params)
                preds = self._predict(model, scaler, X_test)
                score = self._evaluate_metric(preds, y_test, X_test.index)
                if np.isnan(score):
                    continue
                fold_scores.append(score)
            if not fold_scores:
                return params, None, []
            avg_score = float(np.mean(fold_scores))
            return params, avg_score, fold_scores

        if cfg.n_jobs and cfg.n_jobs != 1:
            evaluated = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
                delayed(_score_params)(params) for params in param_grid
            )
        else:
            evaluated = [_score_params(params) for params in param_grid]

        for params, avg_score, fold_scores in evaluated:
            if avg_score is None:
                continue
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                best_fold_scores = fold_scores

        if best_params is None:
            return None
        return best_params, best_score, best_fold_scores

    def _fit_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        params: Dict[str, float],
    ) -> Tuple[object, Optional[StandardScaler]]:
        X = features.to_numpy(dtype=float, copy=False)
        y = target.to_numpy(dtype=float, copy=False)
        scaler: Optional[StandardScaler] = None
        if self.config.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        model = self._make_model(params)
        model.fit(X, y)
        return model, scaler

    def _predict(
        self,
        model: object,
        scaler: Optional[StandardScaler],
        features: pd.DataFrame,
    ) -> np.ndarray:
        X = features.to_numpy(dtype=float, copy=False)
        if scaler is not None:
            X = scaler.transform(X)
        return model.predict(X)

    def _make_model(self, params: Dict[str, float]) -> object:
        model_type = self.config.model_type.lower()
        if model_type == "ridge":
            return Ridge(alpha=float(params["alpha"]))
        if model_type == "elasticnet":
            return ElasticNet(
                alpha=float(params["alpha"]),
                l1_ratio=float(params.get("l1_ratio", 0.5)),
                random_state=self.config.random_state,
                max_iter=10000,
            )
        raise ValueError(f"Unsupported meta-blend model_type '{self.config.model_type}'")

    def _build_param_grid(self) -> List[Dict[str, float]]:
        cfg = self.config
        alphas = [float(a) for a in cfg.alpha_grid if a > 0]
        if not alphas:
            alphas = [0.1]
        if cfg.model_type.lower() == "elasticnet":
            ratios = [float(r) for r in cfg.l1_ratio_grid if 0 <= r <= 1]
            if not ratios:
                ratios = [0.5]
            return [{"alpha": a, "l1_ratio": r} for a in alphas for r in ratios]
        return [{"alpha": a} for a in alphas]

    def _evaluate_metric(
        self,
        predictions: np.ndarray,
        actual: pd.Series,
        index: pd.Index,
    ) -> float:
        metric = self.config.cv_metric.lower()
        pred_series = pd.Series(predictions, index=index)
        df = pd.DataFrame({"pred": pred_series, "actual": actual})
        if metric == "sharpe_pred":
            returns = self._compute_weighted_returns(df)
            if returns.empty:
                return np.nan
            return float(sharpe(returns))
        return float(self._mean_ic(df))

    @staticmethod
    def _mean_ic(df: pd.DataFrame) -> float:
        grouped = df.groupby(level=0)
        ics: List[float] = []
        for _, group in grouped:
            if group["pred"].nunique() < 2 or group["actual"].nunique() < 2:
                continue
            corr = group["pred"].corr(group["actual"])
            if np.isnan(corr):
                continue
            ics.append(float(corr))
        if not ics:
            return np.nan
        return float(np.mean(ics))

    @staticmethod
    def _compute_weighted_returns(df: pd.DataFrame) -> pd.Series:
        grouped = df.groupby(level=0)
        rets: List[Tuple[pd.Timestamp, float]] = []
        for date, group in grouped:
            weights = group["pred"].replace([np.inf, -np.inf], np.nan).dropna()
            actual = group.loc[weights.index, "actual"]
            if weights.empty or actual.empty:
                continue
            weights = weights - weights.mean()
            norm = weights.abs().sum()
            if norm == 0:
                continue
            weights = weights / norm
            ret = float((weights * actual).sum())
            rets.append((date, ret))
        if not rets:
            return pd.Series(dtype=float)
        dates, values = zip(*rets)
        return pd.Series(values, index=pd.Index(dates, name="date"))


def build_feature_frame(
    momentum: pd.DataFrame,
    quality: pd.DataFrame,
    regime: Optional[pd.Series],
    config: MetaBlendConfig,
) -> pd.DataFrame:
    def _stack(df: pd.DataFrame, name: str) -> pd.Series:
        try:
            return df.stack(future_stack=True).rename(name)
        except TypeError:
            return df.stack(dropna=False).rename(name)

    momentum = momentum.sort_index()
    quality = quality.reindex(momentum.index)
    common_assets = momentum.columns.intersection(quality.columns)
    momentum = momentum.reindex(columns=common_assets)
    quality = quality.reindex(columns=common_assets)

    data = pd.concat([
        _stack(momentum, "momentum"),
        _stack(quality, "quality"),
    ], axis=1)

    # Sanitize infinities that may arise from upstream calculations
    data = data.replace([np.inf, -np.inf], np.nan)

    if config.use_regime_feature and regime is not None:
        regime_aligned = regime.reindex(momentum.index).ffill().bfill().fillna(0.0)
        regime_vals = regime_aligned.reindex(data.index.get_level_values(0)).to_numpy()
        data["regime"] = regime_vals
    elif "regime" in data.columns:
        data = data.drop(columns=["regime"])

    if config.use_interactions and "regime" in data.columns:
        data["regime_momentum"] = data["regime"] * data["momentum"]
        data["regime_quality"] = data["regime"] * data["quality"]

    mandatory = ["momentum", "quality"]
    if config.use_regime_feature and "regime" in data.columns:
        mandatory.append("regime")
    data = data.dropna(subset=mandatory)
    return data


def _maybe_load_scores_cache(
    cache_dir: Optional[Path],
    cache_id: Optional[str],
) -> Optional[Tuple[pd.DataFrame, Dict[str, object]]]:
    if not cache_dir or not cache_id:
        return None
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{cache_id}.joblib"
    if not cache_path.exists():
        return None
    try:
        payload = load(cache_path)
    except (OSError, ValueError, EOFError):
        return None
    if not isinstance(payload, dict):
        return None
    mix_df = payload.get("mix_df")
    meta = payload.get("meta", {})
    if not isinstance(mix_df, pd.DataFrame):
        return None
    if not isinstance(meta, dict):
        meta = {}
    meta = dict(meta)
    meta["cache_hit"] = True
    return mix_df, meta


def _store_scores_cache(
    cache_dir: Optional[Path],
    cache_id: Optional[str],
    mix_df: pd.DataFrame,
    meta: Dict[str, object],
) -> None:
    if not cache_dir or not cache_id:
        return
    cache_dir = Path(cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    cache_path = cache_dir / f"{cache_id}.joblib"
    payload = {
        "mix_df": mix_df,
        "meta": dict(meta),
    }
    try:
        dump(payload, cache_path)
    except (OSError, ValueError):
        return


def run_meta_blend(
    momentum: pd.DataFrame,
    quality: pd.DataFrame,
    regime: Optional[pd.Series],
    forward_returns: pd.DataFrame,
    assets: Sequence[str],
    config_dict: Optional[Dict[str, object]],
    *,
    cache_dir: Optional[Path] = None,
    cache_id: Optional[str] = None,
    scores_cache_dir: Optional[Path] = None,
    scores_cache_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cached_scores = _maybe_load_scores_cache(scores_cache_dir, scores_cache_id)
    if cached_scores is not None:
        return cached_scores

    config = MetaBlendConfig.from_dict(config_dict)
    features = build_feature_frame(momentum, quality, regime, config)
    forward = forward_returns.reindex(momentum.index).reindex(columns=assets)
    try:
        target = forward.sort_index().stack(future_stack=True).rename("target")
    except TypeError:
        target = forward.sort_index().stack(dropna=False).rename("target")
    # Ensure target has only finite values
    target = target.replace([np.inf, -np.inf], np.nan)
    dataset_cache_path: Optional[Path] = None
    if cache_dir is not None and cache_id:
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset_cache_path = cache_dir / f"{cache_id}.parquet"
        if dataset_cache_path.exists():
            try:
                dataset = pd.read_parquet(dataset_cache_path)
            except (OSError, ValueError):
                dataset = None
        else:
            dataset = None
    else:
        dataset = None

    if dataset is None:
        dataset = features.join(target, how="inner")
        dataset = dataset.dropna(subset=["target"])
        if dataset_cache_path is not None:
            try:
                dataset.to_parquet(dataset_cache_path, compression="snappy")
            except (OSError, ValueError, ImportError):
                pass
    else:
        dataset = dataset.dropna(subset=["target"])

    if dataset.empty:
        return pd.DataFrame(columns=assets), {"warning": "empty_dataset"}
    features_df = dataset.drop(columns=["target"])
    target_series = dataset["target"]
    blender = MetaBlender(config)
    scores, meta = blender.generate_scores(features_df, target_series, assets=assets)
    _store_scores_cache(scores_cache_dir, scores_cache_id, scores, meta)
    return scores, meta
