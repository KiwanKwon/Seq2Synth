"""Multi-Parent Temporal Conditional Distribution Similarity.

Seq2Synth paper, Section Appendix B.6, Eq. 39.

For a Multi-Child + Multi-Parent schema, measures whether the conditional
temporal distribution of a child feature :math:`X_t`, given joint conditions
``(a, b, ...)`` on multiple parents, is preserved:

.. math::

    D_{\\text{cond}} = \\sum_{a, b, t} w_{a,b,t} \\,
        W_1\\!\\left( F_{\\text{real}}(X_t \\mid a, b),
                     F_{\\text{synth}}(X_t \\mid a, b) \\right),
    \\qquad
    \\text{score} = 1 - D_{\\text{cond}}.

Because :math:`W_1` is unbounded, this implementation normalizes each cell's
Wasserstein distance by ``scale_X = max(IQR(X_real), eps)`` and clips to
``[0, 1]``:

.. math::

    W_1^{\\text{norm}} = \\min(1, W_1 / \\text{scale}_X).

The paper leaves the normalization unspecified; this choice keeps the final
score in ``[0, 1]`` and is reported in ``details["scale_X"]``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from seq2synth.metrics.structural.base import BaseStructuralMetric

logger = logging.getLogger(__name__)

_EPS = 1e-12


class MultiParentConditionalSimilarity(BaseStructuralMetric):
    """Multi-Parent Temporal Conditional Distribution Similarity (Eq. 55-56).

    Attributes:
        child_table: Name of the child table holding the target and time.
        parent_tables: List of parent table names (must contain at least two).
        target_col: Numerical column in ``child_table`` to evaluate.
        condition_cols: Mapping ``{parent_table: condition_column}``.
            Each column is used as a categorical condition (raw values) or
            binned into ``n_numeric_bins`` quantile buckets if numeric.
        time_col: Timestamp column in ``child_table``.
        time_bin: Pandas offset alias for the temporal grouping.
        n_numeric_bins: Number of quantile bins for numeric condition
            columns.
        min_group_size: Minimum number of real (and synthetic) samples in a
            ``(conditions, time_bin)`` cell required for evaluation.
    """

    def __init__(
        self,
        child_table: str,
        parent_tables: list[str],
        target_col: str,
        condition_cols: dict[str, str],
        time_col: str,
        time_bin: str = "1D",
        n_numeric_bins: int = 4,
        min_group_size: int = 10,
    ) -> None:
        if len(parent_tables) < 2:
            raise ValueError(
                "MultiParentConditionalSimilarity requires at least two "
                "parent tables (multi-parent setting)."
            )
        missing_conditions = [p for p in parent_tables if p not in condition_cols]
        if missing_conditions:
            raise ValueError(
                f"condition_cols missing entries for parents: {missing_conditions}"
            )
        self.child_table = child_table
        self.parent_tables = list(parent_tables)
        self.target_col = target_col
        self.condition_cols = dict(condition_cols)
        self.time_col = time_col
        self.time_bin = time_bin
        self.n_numeric_bins = int(n_numeric_bins)
        self.min_group_size = int(min_group_size)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _primary_key(schema: dict[str, Any], table: str) -> str | None:
        return schema.get("tables", {}).get(table, {}).get("primary_key")

    def _fk_for_parent(
        self, schema: dict[str, Any], parent: str
    ) -> str:
        """Return the FK column in ``child_table`` that references ``parent``."""
        for rel in schema.get("relationships", []) or []:
            if rel.get("parent") == parent and rel.get("child") == self.child_table:
                return rel["fk"]
        raise ValueError(
            f"no relationship from child='{self.child_table}' to parent="
            f"'{parent}' in schema['relationships']"
        )

    def _is_numerical(
        self, schema: dict[str, Any], table: str, col: str
    ) -> bool:
        t = schema.get("tables", {}).get(table, {})
        return col in (t.get("numerical_cols") or [])

    # ---------------------------------------------------------------- joining

    def _assemble(
        self,
        data: dict[str, pd.DataFrame],
        schema: dict[str, Any],
    ) -> pd.DataFrame:
        """Join child with each parent on its FK and return a frame with
        columns ``_T``, ``_X``, and one ``_C_<parent>`` per parent."""
        child = data[self.child_table]
        frame = child[
            [self.time_col, self.target_col]
            + [
                self._fk_for_parent(schema, p)
                for p in self.parent_tables
            ]
        ].copy()

        rename_map = {self.time_col: "_T", self.target_col: "_X"}
        frame = frame.rename(columns=rename_map)
        frame["_T"] = pd.to_datetime(frame["_T"], errors="raise")
        frame["_X"] = pd.to_numeric(frame["_X"], errors="coerce")

        for parent in self.parent_tables:
            fk = self._fk_for_parent(schema, parent)
            pk = self._primary_key(schema, parent)
            if pk is None:
                raise ValueError(
                    f"primary_key missing for parent table '{parent}'"
                )
            p_df = data[parent][[pk, self.condition_cols[parent]]].copy()
            cond_out = f"_C_{parent}"
            p_df = p_df.rename(
                columns={pk: fk, self.condition_cols[parent]: cond_out}
            )
            frame = frame.merge(p_df, on=fk, how="inner")
        return frame

    # ---------------------------------------------------------- discretization

    def _discretize_conditions(
        self,
        real_frame: pd.DataFrame,
        synth_frame: pd.DataFrame,
        schema: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """For each parent, replace the raw condition column with a discrete
        label, using quantile bin edges from real data when numeric."""
        info: dict[str, Any] = {}
        real_out = real_frame.copy()
        synth_out = synth_frame.copy()
        for parent in self.parent_tables:
            col = f"_C_{parent}"
            cond_col = self.condition_cols[parent]
            if self._is_numerical(schema, parent, cond_col):
                values = pd.to_numeric(real_out[col], errors="coerce").dropna()
                if values.nunique() <= 1:
                    logger.warning(
                        "Condition column %s.%s has a single unique value; "
                        "falling back to a single bin.",
                        parent,
                        cond_col,
                    )
                    real_out[col] = "__singleton__"
                    synth_out[col] = "__singleton__"
                    info[parent] = {"kind": "numeric_singleton"}
                    continue
                try:
                    _, bins = pd.qcut(
                        values,
                        q=self.n_numeric_bins,
                        retbins=True,
                        duplicates="drop",
                    )
                except ValueError:
                    bins = np.array(
                        [values.min(), values.max()], dtype=float
                    )
                if len(bins) < 2:
                    bins = np.array([values.min(), values.max() + _EPS])
                # Expand outermost edges so synthetic out-of-range values still
                # fall into the first/last bin.
                bins = bins.astype(float)
                bins[0] = -np.inf
                bins[-1] = np.inf
                real_out[col] = pd.cut(
                    pd.to_numeric(real_out[col], errors="coerce"),
                    bins=bins,
                    include_lowest=True,
                ).astype(object)
                synth_out[col] = pd.cut(
                    pd.to_numeric(synth_out[col], errors="coerce"),
                    bins=bins,
                    include_lowest=True,
                ).astype(object)
                info[parent] = {"kind": "numeric", "bin_edges": bins.tolist()}
            else:
                # Categorical: keep raw values as labels (stringified to keep
                # hashable/consistent).
                real_out[col] = real_out[col].astype(object)
                synth_out[col] = synth_out[col].astype(object)
                info[parent] = {"kind": "categorical"}
        return real_out, synth_out, info

    # --------------------------------------------------------------- compute

    def compute(
        self,
        real_data: dict[str, pd.DataFrame],
        synth_data: dict[str, pd.DataFrame],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute Multi-Parent Temporal Conditional Distribution Similarity."""
        if not self._is_numerical(schema, self.child_table, self.target_col):
            raise TypeError(
                f"target_col '{self.target_col}' is not listed in "
                f"schema['tables']['{self.child_table}']['numerical_cols']"
            )
        for parent in self.parent_tables:
            # Validates that a FK exists; raises ValueError otherwise.
            self._fk_for_parent(schema, parent)

        needed = [self.child_table, *self.parent_tables]
        for name, d in (("real", real_data), ("synth", synth_data)):
            for t in needed:
                if t not in d:
                    return {
                        "score": np.nan,
                        "details": {
                            "reason": f"table '{t}' missing from {name}_data"
                        },
                    }

        real_frame = self._assemble(real_data, schema)
        synth_frame = self._assemble(synth_data, schema)

        if len(real_frame) == 0 or len(synth_frame) == 0:
            return {
                "score": np.nan,
                "details": {"reason": "empty joined frame"},
            }

        real_frame, synth_frame, disc_info = self._discretize_conditions(
            real_frame, synth_frame, schema
        )

        real_frame["_Tbin"] = real_frame["_T"].dt.floor(self.time_bin)
        synth_frame["_Tbin"] = synth_frame["_T"].dt.floor(self.time_bin)

        cond_cols = [f"_C_{p}" for p in self.parent_tables]
        group_cols = cond_cols + ["_Tbin"]

        # scale_X from real data (IQR with a floor).
        real_x = real_frame["_X"].dropna().to_numpy()
        if real_x.size == 0:
            return {
                "score": np.nan,
                "details": {"reason": "no valid target values in real data"},
            }
        q75, q25 = np.percentile(real_x, [75, 25])
        scale_x = float(max(q75 - q25, _EPS))

        real_groups = {
            key: g["_X"].dropna().to_numpy()
            for key, g in real_frame.groupby(group_cols, dropna=False)
        }
        synth_groups = {
            key: g["_X"].dropna().to_numpy()
            for key, g in synth_frame.groupby(group_cols, dropna=False)
        }

        total_real = sum(len(v) for v in real_groups.values())
        if total_real == 0:
            return {
                "score": np.nan,
                "details": {"reason": "no samples after discretization"},
            }

        per_cell: list[dict[str, Any]] = []
        weighted_w1 = 0.0
        weighted_w1_norm = 0.0
        total_weight = 0.0
        n_evaluated = 0
        n_skipped = 0

        for key, real_vals in real_groups.items():
            synth_vals = synth_groups.get(key, np.array([], dtype=float))
            if (
                real_vals.size < self.min_group_size
                or synth_vals.size < self.min_group_size
            ):
                n_skipped += 1
                continue
            w1 = float(stats.wasserstein_distance(real_vals, synth_vals))
            w1_norm = float(min(1.0, w1 / scale_x))
            weight = real_vals.size / total_real
            weighted_w1 += weight * w1
            weighted_w1_norm += weight * w1_norm
            total_weight += weight
            n_evaluated += 1
            per_cell.append(
                {
                    "conditions": _stringify_key(key, self.parent_tables),
                    "n_real": int(real_vals.size),
                    "n_synth": int(synth_vals.size),
                    "w1": w1,
                    "w1_norm": w1_norm,
                    "weight": float(weight),
                }
            )

        if n_evaluated == 0:
            return {
                "score": np.nan,
                "details": {
                    "reason": "no cells with sufficient samples",
                    "n_cells_skipped": int(n_skipped),
                    "scale_X": scale_x,
                },
            }

        # Renormalize by total evaluated weight (cells below min_group_size
        # are excluded, so remaining weights may sum to < 1).
        if total_weight > 0:
            weighted_w1 /= total_weight
            weighted_w1_norm /= total_weight

        score = float(max(0.0, min(1.0, 1.0 - weighted_w1_norm)))

        top_cells = sorted(per_cell, key=lambda r: -r["w1_norm"])[:10]
        details: dict[str, Any] = {
            "n_cells_evaluated": int(n_evaluated),
            "n_cells_skipped": int(n_skipped),
            "weighted_wasserstein_distance": float(weighted_w1),
            "weighted_wasserstein_distance_normalized": float(weighted_w1_norm),
            "scale_X": scale_x,
            "discretization": disc_info,
            "per_cell": top_cells,
        }
        logger.debug(
            "MultiParentConditionalSim: cells=%d skipped=%d w1_norm=%.4f "
            "score=%.4f",
            n_evaluated,
            n_skipped,
            weighted_w1_norm,
            score,
        )
        return {"score": score, "details": details}


def _stringify_key(
    key: Any, parent_tables: Iterable[str]
) -> dict[str, str]:
    """Turn a groupby key (possibly a tuple) into a readable mapping."""
    if not isinstance(key, tuple):
        key = (key,)
    out: dict[str, str] = {}
    parents = list(parent_tables)
    for i, p in enumerate(parents):
        out[p] = str(key[i]) if i < len(key) else ""
    if len(key) > len(parents):
        out["_Tbin"] = str(key[len(parents)])
    return out
