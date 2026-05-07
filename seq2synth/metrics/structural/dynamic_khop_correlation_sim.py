"""Compute Dynamic K-Hop Correlation Similarity.

This metric compares the temporal evolution of k-hop correlation
structures between real and synthetic data.

It is defined as:

.. math::

    \\text{MAE} = \\frac{1}{|T_{\\text{eval}}|}
    \\sum_{t \\in T_{\\text{eval}}}
    \\left| \\rho^{(k)}_{\\text{real}}(t) - \\rho^{(k)}_{\\text{synth}}(t) \\right|

    \\quad

    \\text{DynamicKHopCorrelationSim} = 1 - \\text{MAE}

where:
- :math:`\\rho^{(k)}(t)` is the k-hop correlation at time t
- :math:`T_{\\text{eval}}` is the set of evaluation time points

A higher score indicates that the synthetic data more closely
matches the temporal correlation dynamics of the real data.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from seq2synth.metrics.structural.base import BaseStructuralMetric

logger = logging.getLogger(__name__)


class DynamicKHopCorrelationSimilarity(BaseStructuralMetric):
    """Dynamic k-hop Correlation Similarity (Eq. 38).

    Attributes:
        variable_a: ``(table_name, column_name)`` for the first numerical
            variable.
        variable_b: ``(table_name, column_name)`` for the second numerical
            variable.
        time_col: ``(table_name, column_name)`` providing the evaluation
            timestamp, typically the time column of the leaf-most table on
            the join path.
        time_bin: Pandas offset alias defining the :math:`T_{\\text{eval}}`
            granularity (default ``"1D"``).
        min_samples_per_bin: Minimum number of paired observations per time
            bin required to compute the correlation; bins below this are
            skipped.
    """

    def __init__(
        self,
        variable_a: tuple[str, str],
        variable_b: tuple[str, str],
        time_col: tuple[str, str],
        time_bin: str = "1D",
        min_samples_per_bin: int = 5,
    ) -> None:
        self.variable_a = variable_a
        self.variable_b = variable_b
        self.time_col = time_col
        self.time_bin = time_bin
        self.min_samples_per_bin = int(min_samples_per_bin)

    # ------------------------------------------------------------------ path

    @staticmethod
    def _build_graph(
        schema: dict[str, Any]
    ) -> dict[str, list[dict[str, str]]]:
        """Build an undirected adjacency list keyed by table name.

        Each edge carries the relationship direction so we know which side
        holds the foreign key when assembling joins.
        """
        graph: dict[str, list[dict[str, str]]] = {}
        for rel in schema.get("relationships", []) or []:
            parent = rel["parent"]
            child = rel["child"]
            fk = rel["fk"]
            graph.setdefault(parent, []).append(
                {"neighbor": child, "direction": "to_child", "fk": fk}
            )
            graph.setdefault(child, []).append(
                {"neighbor": parent, "direction": "to_parent", "fk": fk}
            )
        return graph

    @classmethod
    def _bfs_path(
        cls,
        schema: dict[str, Any],
        source: str,
        target: str,
    ) -> list[dict[str, str]] | None:
        """Return the shortest FK path from ``source`` to ``target`` as a list
        of edge dicts; ``None`` if no path exists."""
        if source == target:
            return []
        graph = cls._build_graph(schema)
        if source not in graph or target not in graph:
            return None
        visited = {source}
        queue: deque[tuple[str, list[dict[str, str]]]] = deque([(source, [])])
        while queue:
            node, path = queue.popleft()
            for edge in graph.get(node, []):
                nb = edge["neighbor"]
                if nb in visited:
                    continue
                step = {
                    "from": node,
                    "to": nb,
                    "direction": edge["direction"],
                    "fk": edge["fk"],
                }
                new_path = path + [step]
                if nb == target:
                    return new_path
                visited.add(nb)
                queue.append((nb, new_path))
        return None

    # ----------------------------------------------------------------- utils

    @staticmethod
    def _is_numerical(schema: dict[str, Any], table: str, col: str) -> bool:
        t = schema.get("tables", {}).get(table, {})
        return col in (t.get("numerical_cols") or [])

    @staticmethod
    def _primary_key(schema: dict[str, Any], table: str) -> str | None:
        return schema.get("tables", {}).get(table, {}).get("primary_key")

    def _join_along_path(
        self,
        data: dict[str, pd.DataFrame],
        schema: dict[str, Any],
        path: list[dict[str, str]],
    ) -> pd.DataFrame:
        """Join tables along ``path`` starting from ``variable_a``'s table.

        Returns a DataFrame containing the columns needed to evaluate the
        metric: ``_A``, ``_B`` and ``_T``.
        """
        table_a, col_a = self.variable_a
        table_b, col_b = self.variable_b
        table_t, col_t = self.time_col

        # Track per-side column namespace so we don't collide when two tables
        # share column names.
        def _prefix(df: pd.DataFrame, table: str) -> pd.DataFrame:
            return df.rename(columns={c: f"{table}.{c}" for c in df.columns})

        current_table = table_a
        frame = _prefix(data[table_a], table_a)

        for step in path:
            nxt = step["to"]
            direction = step["direction"]
            fk = step["fk"]
            nxt_df = _prefix(data[nxt], nxt)
            if direction == "to_child":
                # current = parent, next = child; join on next.fk == current.PK
                parent_pk = self._primary_key(schema, current_table)
                if parent_pk is None:
                    raise ValueError(
                        f"primary_key missing for table '{current_table}'"
                    )
                left_on = f"{current_table}.{parent_pk}"
                right_on = f"{nxt}.{fk}"
            else:  # to_parent
                # current = child, next = parent; join on current.fk == next.PK
                parent_pk = self._primary_key(schema, nxt)
                if parent_pk is None:
                    raise ValueError(f"primary_key missing for table '{nxt}'")
                left_on = f"{current_table}.{fk}"
                right_on = f"{nxt}.{parent_pk}"
            if left_on not in frame.columns:
                raise KeyError(
                    f"join column '{left_on}' not found in intermediate frame"
                )
            if right_on not in nxt_df.columns:
                raise KeyError(
                    f"join column '{right_on}' not found in table '{nxt}'"
                )
            frame = frame.merge(
                nxt_df, left_on=left_on, right_on=right_on, how="inner"
            )
            current_table = nxt

        a_full = f"{table_a}.{col_a}"
        b_full = f"{table_b}.{col_b}"
        t_full = f"{table_t}.{col_t}"
        for needed in (a_full, b_full, t_full):
            if needed not in frame.columns:
                raise KeyError(
                    f"required column '{needed}' missing after join"
                )
        out = frame[[a_full, b_full, t_full]].rename(
            columns={a_full: "_A", b_full: "_B", t_full: "_T"}
        )
        out["_T"] = pd.to_datetime(out["_T"], errors="raise")
        out["_A"] = pd.to_numeric(out["_A"], errors="coerce")
        out["_B"] = pd.to_numeric(out["_B"], errors="coerce")
        out = out.dropna(subset=["_A", "_B", "_T"])
        return out

    def _bin_correlations(
        self, joined: pd.DataFrame
    ) -> dict[pd.Timestamp, float]:
        """Compute Pearson rho for each time bin with enough samples."""
        if len(joined) == 0:
            return {}
        bins = joined["_T"].dt.floor(self.time_bin)
        grouped = joined.assign(_bin=bins).groupby("_bin", sort=True)
        out: dict[pd.Timestamp, float] = {}
        for bin_ts, g in grouped:
            if len(g) < self.min_samples_per_bin:
                continue
            a = g["_A"].to_numpy()
            b = g["_B"].to_numpy()
            if np.std(a) == 0.0 or np.std(b) == 0.0:
                continue
            # Pearson correlation via numpy for speed and NaN clarity.
            rho = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(rho):
                continue
            out[bin_ts] = rho
        return out

    # --------------------------------------------------------------- compute

    def compute(
        self,
        real_data: dict[str, pd.DataFrame],
        synth_data: dict[str, pd.DataFrame],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute Dynamic k-hop Correlation Similarity."""
        table_a, col_a = self.variable_a
        table_b, col_b = self.variable_b
        table_t, col_t = self.time_col

        # --- Type / schema validation.
        if not self._is_numerical(schema, table_a, col_a):
            raise TypeError(
                f"variable_a {self.variable_a} is not listed in "
                f"schema['tables']['{table_a}']['numerical_cols']"
            )
        if not self._is_numerical(schema, table_b, col_b):
            raise TypeError(
                f"variable_b {self.variable_b} is not listed in "
                f"schema['tables']['{table_b}']['numerical_cols']"
            )

        # --- Find FK path A -> B.
        path = self._bfs_path(schema, table_a, table_b)
        if path is None:
            return {
                "score": np.nan,
                "details": {
                    "reason": (
                        f"no FK path between '{table_a}' and '{table_b}'"
                    )
                },
            }

        # The time-providing table must be on the join path (A, then each
        # step's `to`) so the evaluation timestamp is available after the
        # join.
        path_tables = [table_a] + [step["to"] for step in path]
        if table_t not in path_tables:
            return {
                "score": np.nan,
                "details": {
                    "reason": (
                        f"time_col table '{table_t}' is not on the FK path "
                        f"{path_tables}"
                    )
                },
            }

        for name, d in (("real", real_data), ("synth", synth_data)):
            for t in path_tables:
                if t not in d:
                    return {
                        "score": np.nan,
                        "details": {
                            "reason": f"table '{t}' missing from {name}_data"
                        },
                    }

        try:
            real_joined = self._join_along_path(real_data, schema, path)
            synth_joined = self._join_along_path(synth_data, schema, path)
        except KeyError as exc:
            return {
                "score": np.nan,
                "details": {"reason": f"join failed: {exc}"},
            }

        real_corrs = self._bin_correlations(real_joined)
        synth_corrs = self._bin_correlations(synth_joined)

        common = sorted(set(real_corrs) & set(synth_corrs))
        skipped = (
            len(set(real_corrs) ^ set(synth_corrs))
            + max(
                0,
                len(pd.unique(real_joined["_T"].dt.floor(self.time_bin)))
                - len(real_corrs),
            )
        )

        if len(common) < 2:
            return {
                "score": np.nan,
                "details": {
                    "reason": "insufficient temporal bins",
                    "hop_count": len(path),
                    "fk_path": path,
                    "n_bins_evaluated": len(common),
                    "n_bins_skipped": int(skipped),
                },
            }

        diffs = np.array(
            [abs(real_corrs[t] - synth_corrs[t]) for t in common], dtype=float
        )
        mae = float(np.mean(diffs))
        score = max(0.0, min(1.0, 1.0 - mae))

        per_bin = [
            {
                "bin": pd.Timestamp(t).isoformat(),
                "rho_real": float(real_corrs[t]),
                "rho_synth": float(synth_corrs[t]),
            }
            for t in common
        ]

        details: dict[str, Any] = {
            "hop_count": len(path),
            "fk_path": path,
            "n_bins_evaluated": len(common),
            "n_bins_skipped": int(skipped),
            "mae": mae,
            "per_bin": per_bin,
        }
        logger.debug(
            "DynamicKHopCorrelationSim: k=%d bins=%d mae=%.4f score=%.4f",
            len(path),
            len(common),
            mae,
            score,
        )
        return {"score": score, "details": details}
