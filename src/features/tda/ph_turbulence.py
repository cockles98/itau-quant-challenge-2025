from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape, PairwiseDistance


def _persistence_lengths(
    diagram: np.ndarray,
    infinity_value: float,
) -> np.ndarray:
    """Return valid persistence lengths for a single diagram."""
    if diagram.size == 0:
        return np.empty(0, dtype=float)
    birth = diagram[:, 0]
    death = diagram[:, 1]
    dims = diagram[:, 2]
    valid = np.isfinite(birth) & (dims >= 0)
    effective_death = np.where(np.isfinite(death), death, infinity_value)
    lengths = np.maximum(effective_death - birth, 0.0)
    valid &= lengths > 0
    return lengths[valid]


def _diagram_counts(
    diagram: np.ndarray,
    infinity_value: float,
    dimensions: Sequence[int],
) -> Dict[int, int]:
    """Count non-degenerate features per homology dimension."""
    if diagram.size == 0:
        return {dim: 0 for dim in dimensions}
    birth = diagram[:, 0]
    death = diagram[:, 1]
    dims = diagram[:, 2].astype(int)
    effective_death = np.where(np.isfinite(death), death, infinity_value)
    valid = np.isfinite(birth) & np.isfinite(effective_death)
    valid &= effective_death > birth
    counts: Dict[int, int] = {}
    for dim in dimensions:
        counts[dim] = int(np.count_nonzero(valid & (dims == dim)))
    return counts


class PHTurbulenceTransformer(BaseEstimator, TransformerMixin):
    """Rolling persistent homology turbulence indicator.

    Parameters
    ----------
    window : int, default 50
        Size of the sliding window applied over the rows (dates).
    homology_dimensions : sequence of int, default (0, 1)
        Homology dimensions to compute in Vietoris-Rips persistence.
    metric : str, default "euclidean"
        Distance metric used inside ``VietorisRipsPersistence``.
    norm : {"l1", "l2"}, default "l2"
        Norm applied to the persistence landscapes when building the primary
        turbulence score.
    n_layers : int, default 3
        Number of layers used in the persistence landscape representation.
    landscape_bins : int, default 100
        Number of bins for the landscape sampling grid.
    collapse_edges : bool, default True
        Whether to use edge collapse acceleration inside Vietoris-Rips.
    infinity_values : float | None, default None
        Value assigned to infinite deaths in persistence diagrams. ``None``
        delegates to the internal choice of Giotto-TDA.
    n_jobs : int | None, default None
        Parallel jobs forwarded to Giotto-TDA estimators.
    compute_entropy : bool, default True
        If True, compute and expose persistence entropy as an extra feature.
    compute_amplitude : bool, default True
        If True, compute and expose the sum of persistence lengths.
    compute_wasserstein : bool, default True
        If True, compute a one-step Wasserstein shift between consecutive
        diagrams (can be costly for long series).
    wasserstein_p : float, default 2.0
        Order parameter for the Wasserstein distance when enabled.
    output_name : str, default "ph_turbulence"
        Name assigned to the returned Series.

    Notes
    -----
    The transformer expects a ``pandas.DataFrame`` with a Datetime-like index
    and wide format (columns as tickers). The ``transform`` method returns a
    Series indexed exactly as the input; entries preceding the first full
    window are filled with ``NaN``.
    Extra features and metadata are made available after calling ``transform``
    via :attr:`extra_features_` and :attr:`meta_`, respectively.
    """

    def __init__(
        self,
        *,
        window: int = 50,
        homology_dimensions: Sequence[int] = (0, 1),
        metric: str = "euclidean",
        norm: str = "l2",
        n_layers: int = 3,
        landscape_bins: int = 100,
        collapse_edges: bool = True,
        infinity_values: float | None = None,
        n_jobs: int | None = None,
        compute_entropy: bool = True,
        compute_amplitude: bool = True,
        compute_wasserstein: bool = True,
        wasserstein_p: float = 2.0,
        output_name: str = "ph_turbulence",
    ) -> None:
        self.window = window
        self.homology_dimensions = tuple(homology_dimensions)
        self.metric = metric
        self.norm = norm
        self.n_layers = n_layers
        self.landscape_bins = landscape_bins
        self.collapse_edges = collapse_edges
        self.infinity_values = infinity_values
        self.n_jobs = n_jobs
        self.compute_entropy = compute_entropy
        self.compute_amplitude = compute_amplitude
        self.compute_wasserstein = compute_wasserstein
        self.wasserstein_p = wasserstein_p
        self.output_name = output_name

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PHTurbulenceTransformer":
        X = self._validate_input(X)
        if X.shape[0] < self.window:
            msg = (
                f"Need at least {self.window} rows to compute persistence "
                f"features; received {X.shape[0]}."
            )
            raise ValueError(msg)

        self._n_features_in_ = X.shape[1]
        self._columns = tuple(X.columns)
        self._vr_ = VietorisRipsPersistence(
            metric=self.metric,
            homology_dimensions=self.homology_dimensions,
            collapse_edges=self.collapse_edges,
            infinity_values=self.infinity_values,
            n_jobs=self.n_jobs,
        )
        self._landscape_ = PersistenceLandscape(
            n_layers=self.n_layers,
            n_bins=self.landscape_bins,
            n_jobs=self.n_jobs,
        )
        if self.compute_wasserstein:
            self._pairwise_ = PairwiseDistance(
                metric="wasserstein",
                order=self.wasserstein_p,
                metric_params={"p": self.wasserstein_p},
                n_jobs=self.n_jobs,
            )
        else:
            self._pairwise_ = None

        # Reset state containers
        self.extra_features_: pd.DataFrame | None = None
        self.meta_: Dict[str, pd.Series] | None = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        check_is_fitted(self, "_n_features_in_")
        X = self._validate_input(X)
        if X.shape[1] != self._n_features_in_:
            msg = (
                "Input columns do not match fitted data. "
                f"Expected {self._columns}, received {tuple(X.columns)}."
            )
            raise ValueError(msg)
        if X.shape[0] < self.window:
            msg = (
                f"Need at least {self.window} rows to compute persistence "
                f"features; received {X.shape[0]}."
            )
            raise ValueError(msg)

        window_data, window_index = self._build_windows(X)
        diagrams = self._vr_.fit_transform(window_data)
        landscapes = self._landscape_.fit_transform(diagrams)
        flattened = landscapes.reshape(landscapes.shape[0], -1)
        turbulence = self._aggregate_norm(flattened)

        series = pd.Series(
            np.nan,
            index=X.index,
            dtype=float,
            name=self.output_name,
        )
        series.loc[window_index] = turbulence

        infinity_value = (
            self._vr_.infinity_values_
            if self._vr_.infinity_values_ is not None
            else np.max(np.where(np.isfinite(diagrams[..., 1]), diagrams[..., 1], 0.0))
        )

        extra_features = pd.DataFrame(index=X.index)
        if self.compute_entropy:
            entropy = np.array(
                [
                    self._persistence_entropy(diagram, infinity_value)
                    for diagram in diagrams
                ]
            )
            extra_features.loc[window_index, "persistence_entropy"] = entropy

        if self.compute_amplitude:
            amplitude = np.array(
                [
                    self._persistence_amplitude(diagram, infinity_value)
                    for diagram in diagrams
                ]
            )
            extra_features.loc[window_index, "amplitude_sum"] = amplitude

        if self.compute_wasserstein and self._pairwise_ is not None:
            matrix = self._pairwise_.fit_transform(diagrams)
            wasserstein = np.full(diagrams.shape[0], np.nan, dtype=float)
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                idx = np.arange(1, matrix.shape[0])
                wasserstein[idx] = matrix[idx, idx - 1]
            extra_features.loc[
                window_index,
                "wasserstein_shift",
            ] = wasserstein

        self.extra_features_ = extra_features if not extra_features.empty else None

        counts_h0, counts_h1 = self._collect_counts(
            diagrams,
            window_index,
            X.index,
            infinity_value,
        )
        self.meta_ = {
            "diag_count_H0": counts_h0,
            "diag_count_H1": counts_h1,
        }
        return series

    # --------------------------------------------------------------------- #
    # Helpers
    def _aggregate_norm(self, landscapes: np.ndarray) -> np.ndarray:
        norm = self.norm.lower()
        if norm not in {"l1", "l2"}:
            raise ValueError("norm must be either 'l1' or 'l2'.")
        ord_ = 1 if norm == "l1" else 2
        return np.linalg.norm(landscapes, ord=ord_, axis=1)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas.DataFrame.")
        if X.isna().to_numpy().any():
            raise ValueError("Input contains NaNs; please impute before calling.")
        return X

    def _build_windows(self, X: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
        values = X.to_numpy(dtype=float)
        n_samples = values.shape[0]
        windows: List[np.ndarray] = []
        index: List[pd.Timestamp] = []
        for end in range(self.window - 1, n_samples):
            start = end - self.window + 1
            windows.append(values[start : end + 1])
            index.append(X.index[end])
        return np.stack(windows, axis=0), pd.Index(index, name=X.index.name)

    def _persistence_entropy(
        self,
        diagram: np.ndarray,
        infinity_value: float,
    ) -> float:
        lengths = _persistence_lengths(diagram, infinity_value)
        total = float(np.sum(lengths))
        if total == 0.0:
            return 0.0
        prob = lengths / total
        return float(-np.sum(prob * np.log(prob)))

    def _persistence_amplitude(
        self,
        diagram: np.ndarray,
        infinity_value: float,
    ) -> float:
        lengths = _persistence_lengths(diagram, infinity_value)
        return float(np.sum(lengths))

    def _collect_counts(
        self,
        diagrams: np.ndarray,
        window_index: pd.Index,
        full_index: pd.Index,
        infinity_value: float,
    ) -> tuple[pd.Series, pd.Series]:
        counts_0 = pd.Series(np.nan, index=full_index, dtype=float)
        counts_1 = pd.Series(np.nan, index=full_index, dtype=float)
        for diagram, ts in zip(diagrams, window_index):
            counts = _diagram_counts(
                diagram,
                infinity_value,
                dimensions=self.homology_dimensions,
            )
            counts_0.loc[ts] = counts.get(0, 0)
            counts_1.loc[ts] = counts.get(1, 0)
        counts_0.name = "diag_count_H0"
        counts_1.name = "diag_count_H1"
        return counts_0, counts_1
