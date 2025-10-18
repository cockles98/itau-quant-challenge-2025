from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Generator, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from dataio.config import load_config

__all__ = [
    "PurgedKFoldTimeSeries",
    "purged_kfold_split",
    "run_purged_tuning",
]


def _ensure_index(obj: Sequence) -> pd.Index:
    if isinstance(obj, pd.Index):
        return obj
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return pd.Index(obj.index)
    return pd.Index(obj)


class PurgedKFoldTimeSeries:
    """Purged K-Fold cross-validation with temporal embargo.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    embargo : int, default 0
        Number of samples to exclude on each side of the test fold.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if embargo < 0:
            raise ValueError("embargo must be non-negative.")
        self.n_splits = int(n_splits)
        self.embargo = int(embargo)

    def split(
        self,
        X: Sequence,
        y: Sequence | None = None,
        groups: Sequence | None = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices."""

        if groups is not None:
            times = _ensure_index(groups)
        else:
            times = _ensure_index(X)

        order = np.argsort(times.to_numpy())
        sorted_indices = np.asarray(order)
        n_samples = len(times)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        start_pos = 0
        for fold_size in fold_sizes:
            stop_pos = start_pos + fold_size

            test_positions = np.arange(start_pos, stop_pos)
            test_indices = sorted_indices[test_positions]

            embargo_start = max(0, start_pos - self.embargo)
            embargo_stop = min(n_samples, stop_pos + self.embargo)

            keep_before = sorted_indices[:embargo_start]
            keep_after = sorted_indices[embargo_stop:]
            train_indices = np.concatenate((keep_before, keep_after))

            yield train_indices, test_indices
            start_pos = stop_pos

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def purged_kfold_split(
    dates: Sequence[pd.Timestamp] | pd.Index,
    n_splits: int = 5,
    embargo_days: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Backward-compatible helper returning list of train/test index arrays."""

    cv = PurgedKFoldTimeSeries(n_splits=n_splits, embargo=embargo_days)
    index = _ensure_index(dates)
    return list(cv.split(index))


def _sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std <= 1e-12:
        return float("nan")
    return float(np.sqrt(252.0) * mean / std)


def _calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    if max_drawdown >= 0:
        return float("nan")
    denom = abs(max_drawdown)
    return annual_return / denom if denom > 1e-9 else float("nan")


def _evaluate_config(
    cfg: dict,
    panel: pd.DataFrame,
    cv: PurgedKFoldTimeSeries,
    dates: pd.Index,
) -> dict:
    result = run_backtest(cfg, panel=panel)
    equity_curve = result["equity_curve"]
    equity_curve = equity_curve.reindex(dates).ffill().dropna()
    returns = equity_curve.pct_change().dropna()

    calmar = _calmar_ratio(
        result["kpis"].get("annual_return", float("nan")),
        result["kpis"].get("max_drawdown", float("nan")),
    )

    sharpe_scores: list[float] = []
    for _, test_idx in cv.split(dates):
        if len(test_idx) == 0:
            continue
        fold_dates = dates[test_idx]
        local_returns = returns.loc[fold_dates.min() : fold_dates.max()]
        sharpe_scores.append(_sharpe_ratio(local_returns))

    metrics = {
        "annual_return": float(result["kpis"].get("annual_return", float("nan"))),
        "annual_vol": float(result["kpis"].get("annual_vol", float("nan"))),
        "sharpe": float(np.nanmean(sharpe_scores)),
        "max_drawdown": float(result["kpis"].get("max_drawdown", float("nan"))),
        "calmar": float(calmar),
        "final_equity": float(result["kpis"].get("final_equity", float("nan"))),
    }
    return metrics


def _apply_tda_ph(cfg: dict, window: int, norm: str) -> None:
    section = cfg.setdefault("tda_ph", {})
    section["window"] = int(window)
    section["norm"] = str(norm)


def _apply_mapper_params(cfg: dict, n_cubes: int, overlap: float, eps_quantile: float) -> None:
    section = cfg.setdefault("mapper", {})
    section["n_cubes"] = int(n_cubes)
    section["overlap"] = float(overlap)
    section["eps_quantile"] = float(eps_quantile)


def _apply_factor_weights(cfg: dict, alpha: float, beta: float, gamma: float, delta: float) -> None:
    factors = cfg.setdefault("factors", {})
    factors["alpha"] = float(alpha)
    factors["beta"] = float(beta)
    factors["gamma"] = float(gamma)
    mapper_cfg = cfg.setdefault("mapper", {})
    periph = mapper_cfg.setdefault("peripherality", {})
    periph["enabled"] = True
    periph["delta"] = float(delta)


def _heatmap(data: pd.DataFrame, x: str, y: str, value: str, title: str, path: Path) -> None:
    pivot = data.pivot(index=y, columns=x, values=value)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(pivot.values):
        label = "nan" if np.isnan(val) else f"{val:.2f}"
        ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_purged_tuning(
    cfg: dict,
    panel: pd.DataFrame,
    *,
    n_splits: int = 5,
    embargo: int = 5,
    output_dir: Path | str = Path("reports"),
) -> dict:
    """Run Purged K-Fold tuning for TDA parameters and factor weights."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = PurgedKFoldTimeSeries(n_splits=n_splits, embargo=embargo)

    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    unique_dates = panel.index.get_level_values(date_level).unique().sort_values()

    summary_rows: list[dict] = []
    heatmap_paths: list[Path] = []

    # Stage 1: tda_ph window x norm
    tda_windows = cfg.get("validation", {}).get("tda_ph_windows", [30, 50, 70])
    tda_norms = cfg.get("validation", {}).get("tda_ph_norms", ["l1", "l2"])
    tda_results = []
    for window, norm in itertools.product(tda_windows, tda_norms):
        cfg_local = json.loads(json.dumps(cfg))
        _apply_tda_ph(cfg_local, window, norm)
        metrics = _evaluate_config(cfg_local, panel, cv, unique_dates)
        record = {"stage": "tda_ph", "window": window, "norm": norm, **metrics}
        tda_results.append(record)
        summary_rows.append(record)
    tda_df = pd.DataFrame(tda_results)
    if not tda_df.empty:
        heatmap_path = output_dir / "heatmap_tda_ph.png"
        _heatmap(tda_df, "norm", "window", "sharpe", "tda_ph Sharpe", heatmap_path)
        heatmap_paths.append(heatmap_path)

    # Stage 2: Mapper grid
    mapper_cfg = cfg.get("validation", {})
    n_cubes_grid = mapper_cfg.get("mapper_n_cubes", [6, 8, 10])
    overlap_grid = mapper_cfg.get("mapper_overlaps", [0.3, 0.5, 0.7])
    eps_grid = mapper_cfg.get("mapper_eps_quantiles", [0.15, 0.25, 0.35])
    mapper_results = []
    for n_cubes, overlap, eps in itertools.product(n_cubes_grid, overlap_grid, eps_grid):
        cfg_local = json.loads(json.dumps(cfg))
        _apply_mapper_params(cfg_local, n_cubes, overlap, eps)
        metrics = _evaluate_config(cfg_local, panel, cv, unique_dates)
        record = {
            "stage": "mapper",
            "n_cubes": n_cubes,
            "overlap": overlap,
            "eps_quantile": eps,
            **metrics,
        }
        mapper_results.append(record)
        summary_rows.append(record)
    mapper_df = pd.DataFrame(mapper_results)
    if not mapper_df.empty:
        for eps_value in mapper_df["eps_quantile"].unique():
            subset = mapper_df[mapper_df["eps_quantile"] == eps_value]
            if subset.empty:
                continue
            heatmap_path = output_dir / f"heatmap_mapper_eps{eps_value:.2f}.png"
            _heatmap(
                subset,
                "overlap",
                "n_cubes",
                "sharpe",
                f"Mapper Sharpe (eps={eps_value:.2f})",
                heatmap_path,
            )
            heatmap_paths.append(heatmap_path)

    # Stage 3: Factor weights
    weights_cfg = cfg.get("validation", {})
    alpha_grid = weights_cfg.get("factor_alpha", [0.4, 0.6, 0.8])
    beta_grid = weights_cfg.get("factor_beta", [0.2, 0.3, 0.4])
    gamma_grid = weights_cfg.get("factor_gamma", [0.05, 0.1, 0.15])
    delta_grid = weights_cfg.get("factor_delta", [0.1, 0.2, 0.3])

    factor_results = []
    for alpha, beta, gamma, delta in itertools.product(alpha_grid, beta_grid, gamma_grid, delta_grid):
        cfg_local = json.loads(json.dumps(cfg))
        _apply_factor_weights(cfg_local, alpha, beta, gamma, delta)
        metrics = _evaluate_config(cfg_local, panel, cv, unique_dates)
        record = {
            "stage": "factor",
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            **metrics,
        }
        factor_results.append(record)
        summary_rows.append(record)
    factor_df = pd.DataFrame(factor_results)
    if not factor_df.empty:
        # Heatmap alpha vs beta (best gamma/delta)
        pivot_df = (
            factor_df.sort_values("sharpe", ascending=False)
            .drop_duplicates(subset=["alpha", "beta"])
        )
        heatmap_path = output_dir / "heatmap_factors_alpha_beta.png"
        _heatmap(pivot_df, "alpha", "beta", "sharpe", "Sharpe (alpha/beta)", heatmap_path)
        heatmap_paths.append(heatmap_path)

        pivot_gamma_delta = (
            factor_df.sort_values("sharpe", ascending=False)
            .drop_duplicates(subset=["gamma", "delta"])
        )
        heatmap_path2 = output_dir / "heatmap_factors_gamma_delta.png"
        _heatmap(
            pivot_gamma_delta,
            "gamma",
            "delta",
            "sharpe",
            "Sharpe (gamma/delta)",
            heatmap_path2,
        )
        heatmap_paths.append(heatmap_path2)

    summary_df = pd.DataFrame(summary_rows)
    results_path = output_dir / "tuning_results.csv"
    summary_df.to_csv(results_path, index=False)

    best_by_stage = (
        summary_df.sort_values("sharpe", ascending=False)
        .groupby("stage")
        .head(1)
        .reset_index(drop=True)
    )

    return {
        "results_csv": results_path,
        "heatmaps": heatmap_paths,
        "best": best_by_stage,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run purged CV tuning pipeline.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Configuration YAML (default: %(default)s)",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument(
        "--output",
        default="reports",
        help="Output directory (default: %(default)s)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dates_cfg = cfg.get("dates", {})
    start = pd.Timestamp(dates_cfg.get("start"))
    end = pd.Timestamp(dates_cfg.get("end"))
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Configuration must include dates.start and dates.end")
    from dataio.loaders import get_panel

    panel = get_panel(start, end)
    result = run_purged_tuning(
        cfg,
        panel,
        n_splits=args.n_splits,
        embargo=args.embargo,
        output_dir=args.output,
    )
    print(f"Tuning results saved to {result['results_csv']}")
    print("Heatmaps:")
    for path in result["heatmaps"]:
        print("  ", path)
    print("Best parameter sets:")
    print(result["best"])
