"""Temporal Cardinality Shape Similarity (Seq2Synth paper, Section Appendix B.6, Eq. 37).

For a Multi-Child relational schema, measures whether the temporal evolution of
parent-to-child cardinality is preserved between real and synthetic data.

For each parent ``p`` and each time window ``w`` on a uniform grid, we count
the number of children of ``p`` whose timestamp falls in ``w`` to obtain a
chronological sequence ``S_p = (count_{p,1}, ..., count_{p,W})``. With
:math:`F_{\\text{real}}` and :math:`F_{\\text{synth}}` the empirical CDFs on
the pooled set of ``count_{p,w}`` values,

.. math::

    D_{KS} = \\sup_S \\lvert F_{\\text{real}}(S) - F_{\\text{synth}}(S) \\rvert,
    \\qquad
    \\text{TemporalCardinalityShapeSim} = 1 - D_{KS}.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from seq2synth.metrics.structural.base import BaseStructuralMetric

logger = logging.getLogger(__name__)


class TemporalCardinalityShapeSimilarity(BaseStructuralMetric):
    """Temporal Cardinality Shape Similarity (Eq. 51-52).

    Attributes:
        parent_table: Name of the parent table.
        child_table: Name of the child table (holds events with a timestamp).
        foreign_key: Column in ``child_table`` referencing ``parent_table``'s
            primary key.
        time_col: Timestamp column in ``child_table``.
        window: Pandas offset alias for the window width (e.g. ``"1D"``,
            ``"1W"``, ``"1M"``).
        align_windows_to_real: When ``True`` (default), build the window grid
            from the real data time range and reuse it for synthetic; when
            ``False``, compute the grid from the union of the two time ranges
            so the same grid is still applied consistently to both sides.
    """

    def __init__(
        self,
        parent_table: str,
        child_table: str,
        foreign_key: str,
        time_col: str,
        window: str = "1D",
        align_windows_to_real: bool = True,
    ) -> None:
        self.parent_table = parent_table
        self.child_table = child_table
        self.foreign_key = foreign_key
        self.time_col = time_col
        self.window = window
        self.align_windows_to_real = align_windows_to_real

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _has_relationship(
        schema: dict[str, Any], parent: str, child: str, fk: str
    ) -> bool:
        for rel in schema.get("relationships", []) or []:
            if (
                rel.get("parent") == parent
                and rel.get("child") == child
                and rel.get("fk") == fk
            ):
                return True
        return False

    def _to_datetime(self, series: pd.Series) -> pd.Series:
        try:
            out = pd.to_datetime(series, errors="raise")
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"Column '{self.time_col}' in table '{self.child_table}' is "
                f"not datetime-castable: {exc}"
            ) from exc
        return out

    def _parent_ids(
        self,
        data: dict[str, pd.DataFrame],
        child: pd.DataFrame,
    ) -> pd.Index:
        """Union of parent ids seen either in the parent table (if provided)
        or referenced from the child FK column."""
        ids: set = set()
        if self.parent_table in data and data[self.parent_table] is not None:
            p_df = data[self.parent_table]
            pk_col = self._parent_pk_col(p_df)
            if pk_col is not None:
                ids.update(pd.unique(p_df[pk_col]))
        if self.foreign_key in child.columns:
            ids.update(pd.unique(child[self.foreign_key].dropna()))
        return pd.Index(sorted(ids, key=lambda x: (str(type(x)), x)))

    def _parent_pk_col(self, parent_df: pd.DataFrame) -> str | None:
        # Prefer a column named like the FK; otherwise fall back to nothing.
        # Schema-based lookup is handled by the caller when available.
        if self.foreign_key in parent_df.columns:
            return self.foreign_key
        return None

    def _build_grid(
        self,
        real_child: pd.DataFrame,
        synth_child: pd.DataFrame,
    ) -> pd.DatetimeIndex:
        real_times = self._to_datetime(real_child[self.time_col])
        if self.align_windows_to_real:
            if len(real_times) == 0:
                return pd.DatetimeIndex([])
            start, end = real_times.min(), real_times.max()
        else:
            synth_times = self._to_datetime(synth_child[self.time_col])
            times = pd.concat([real_times, synth_times], ignore_index=True)
            if len(times) == 0:
                return pd.DatetimeIndex([])
            start, end = times.min(), times.max()
        # Floor/ceil so every observed timestamp falls inside [start, end].
        try:
            start = start.floor(self.window)
            end = end.ceil(self.window)
        except (ValueError, AttributeError):
            # Some offsets (e.g. month starts) are not floor/ceil-able.
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
        grid = pd.date_range(start=start, end=end, freq=self.window)
        if len(grid) < 2:
            # Ensure at least one full window.
            grid = pd.date_range(
                start=start, periods=2, freq=self.window
            )
        return grid

    def _per_parent_counts(
        self,
        child: pd.DataFrame,
        grid: pd.DatetimeIndex,
        parent_ids: pd.Index,
    ) -> np.ndarray:
        """Returns an array of shape (n_parents, n_windows) of counts."""
        n_parents = len(parent_ids)
        n_windows = max(len(grid) - 1, 0)
        counts = np.zeros((n_parents, n_windows), dtype=np.int64)
        if n_windows == 0 or len(child) == 0:
            return counts

        parent_to_row = {pid: i for i, pid in enumerate(parent_ids)}

        times = self._to_datetime(child[self.time_col])
        bins = pd.cut(
            times,
            bins=grid,
            right=False,
            include_lowest=True,
            labels=False,
        )
        fk = child[self.foreign_key]
        mask = bins.notna() & fk.notna()
        bins_arr = bins[mask].to_numpy().astype(np.int64)
        fk_arr = fk[mask].to_numpy()

        for row_idx, w_idx in zip(
            (parent_to_row.get(pid, -1) for pid in fk_arr), bins_arr
        ):
            if row_idx >= 0 and 0 <= w_idx < n_windows:
                counts[row_idx, w_idx] += 1
        return counts

    # --------------------------------------------------------------- compute

    def compute(
        self,
        real_data: dict[str, pd.DataFrame],
        synth_data: dict[str, pd.DataFrame],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute Temporal Cardinality Shape Similarity.

        Returns:
            ``{"score": pooled_score, "details": {...}}`` or
            ``{"score": np.nan, "details": {"reason": ...}}`` when not
            applicable.
        """
        if not self._has_relationship(
            schema, self.parent_table, self.child_table, self.foreign_key
        ):
            return {
                "score": np.nan,
                "details": {
                    "reason": (
                        f"relationship parent={self.parent_table} child="
                        f"{self.child_table} fk={self.foreign_key} not in "
                        "schema['relationships']"
                    )
                },
            }

        if self.child_table not in real_data or self.child_table not in synth_data:
            return {
                "score": np.nan,
                "details": {"reason": f"table '{self.child_table}' missing"},
            }

        real_child = real_data[self.child_table]
        synth_child = synth_data[self.child_table]

        for col in (self.foreign_key, self.time_col):
            if col not in real_child.columns or col not in synth_child.columns:
                return {
                    "score": np.nan,
                    "details": {
                        "reason": (
                            f"column '{col}' missing from child table "
                            f"'{self.child_table}'"
                        )
                    },
                }

        real_parents = self._parent_ids(real_data, real_child)
        synth_parents = self._parent_ids(synth_data, synth_child)

        if len(real_parents) == 0 or len(synth_parents) == 0:
            return {
                "score": np.nan,
                "details": {
                    "reason": "zero parents on one side",
                    "n_real_parents": int(len(real_parents)),
                    "n_synth_parents": int(len(synth_parents)),
                },
            }

        grid = self._build_grid(real_child, synth_child)
        n_windows = max(len(grid) - 1, 0)
        if n_windows == 0:
            return {
                "score": np.nan,
                "details": {"reason": "empty time range"},
            }

        real_counts = self._per_parent_counts(real_child, grid, real_parents)
        synth_counts = self._per_parent_counts(synth_child, grid, synth_parents)

        # Pooled KS over all (parent, window) cells.
        pooled_real = real_counts.ravel()
        pooled_synth = synth_counts.ravel()
        pooled_score = self._one_minus_ks(pooled_real, pooled_synth)

        # Per-window cross-sectional similarity, averaged over windows.
        per_window_scores: list[float] = []
        for w in range(n_windows):
            col_real = real_counts[:, w]
            col_synth = synth_counts[:, w]
            per_window_scores.append(
                self._one_minus_ks(col_real, col_synth)
            )
        per_window_mean = (
            float(np.mean(per_window_scores)) if per_window_scores else float("nan")
        )

        details: dict[str, Any] = {
            "pooled_score": pooled_score,
            "per_window_mean_score": per_window_mean,
            "n_windows": int(n_windows),
            "n_real_parents": int(len(real_parents)),
            "n_synth_parents": int(len(synth_parents)),
            "window_freq": self.window,
        }
        logger.debug(
            "TemporalCardinalityShapeSim: pooled=%.4f per_window_mean=%.4f "
            "n_windows=%d",
            pooled_score,
            per_window_mean,
            n_windows,
        )
        return {"score": pooled_score, "details": details}

    @staticmethod
    def _one_minus_ks(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return float("nan")
        var_a = float(np.var(a))
        var_b = float(np.var(b))
        if var_a == 0.0 and var_b == 0.0:
            return 1.0 if a[0] == b[0] else 0.0
        ks_stat = float(stats.ks_2samp(a, b).statistic)
        return max(0.0, min(1.0, 1.0 - ks_stat))
