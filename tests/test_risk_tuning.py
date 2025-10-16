from pathlib import Path

import pandas as pd

from dataio.config import load_config
from main import _load_panel
from validation import risk_heatmap, tune_risk_parameters


def _load_base_cfg():
    return load_config(Path("configs") / "base.yaml")


def _load_panel_sample(cfg):
    panel = _load_panel(cfg)
    # Use a small slice to keep the test fast.
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    subset_dates = panel.index.get_level_values(date_level).unique()[:40]
    return panel.loc[(subset_dates, slice(None)), :]


def test_tune_risk_parameters_returns_metrics(tmp_path):
    cfg = _load_base_cfg()
    grid = {"risk.target_vol": [0.07, 0.08], "risk.mdd_thres": [-0.12]}

    cfg["paths"]["artifacts"] = str(tmp_path / "artifacts")
    cfg["paths"]["reports"] = str(tmp_path / "reports")

    panel = _load_panel_sample(cfg)

    results, csv_path = tune_risk_parameters(
        cfg,
        panel,
        grid,
        metrics=["sharpe", "max_drawdown"],
        output_dir=tmp_path,
        isolate_artifacts=True,
    )

    assert isinstance(results, pd.DataFrame)
    assert set(["risk.target_vol", "risk.mdd_thres", "sharpe", "max_drawdown"]).issubset(
        results.columns
    )
    assert len(results) == 2
    assert csv_path is not None and csv_path.exists()


def test_risk_heatmap_creates_image(tmp_path):
    cfg = _load_base_cfg()
    grid = {
        "risk.target_vol": [0.07, 0.08],
        "risk.turnover_cap": [0.25, 0.35],
    }

    cfg["paths"]["artifacts"] = str(tmp_path / "artifacts")
    cfg["paths"]["reports"] = str(tmp_path / "reports")

    panel = _load_panel_sample(cfg)

    df, heatmap_path = risk_heatmap(
        cfg,
        panel,
        grid,
        output_dir=tmp_path,
        isolate_artifacts=True,
    )

    assert not df.empty
    assert heatmap_path.exists()
