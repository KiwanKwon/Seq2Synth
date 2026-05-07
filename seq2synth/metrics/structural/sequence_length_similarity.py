"""SequenceLength Similarity metric (Seq2Synth paper, Section Appendix B.6, Eq. 36).

Assesses whether the distribution of flow lengths :math:`L_k = |T(k)|` in
synthetic data matches that of real data, via a two-sample Kolmogorov-Smirnov
test.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from seq2synth.metrics.structural.base import BaseStructuralMetric

logger = logging.getLogger(__name__)


class SequenceLengthSimilarity(BaseStructuralMetric):
    """SequenceLength Similarity (Eq. 49-50 of the Seq2Synth paper).

    For each flow ``k`` defined by :attr:`flow_key_col`, let
    :math:`L_k = |T(k)|` be the flow length. With :math:`F_{\\text{real}}` and
    :math:`F_{\\text{synth}}` the empirical CDFs of these lengths, the metric
    is

    .. math::

        D_{KS} = \\sup_L \\lvert F_{\\text{real}}(L) - F_{\\text{synth}}(L)
        \\rvert, \\qquad
        \\text{SequenceLengthSimilarity} = 1 - D_{KS}.

    Attributes:
        table_name: Name of the table (typically the child / event table)
            whose flows are evaluated.
        flow_key_col: Column used to group rows into flows. This is usually
            the foreign key pointing to the parent entity (e.g. ``store_id``
            in Rossmann), not the row-level primary key.
        strict_variable_length: If ``True`` and both real and synthetic flow
            lengths have zero variance, return ``np.nan`` with reason
            ``"fixed-length data"``. Defaults to ``False``.
    """

    def __init__(
        self,
        table_name: str,
        flow_key_col: str,
        strict_variable_length: bool = False,
    ) -> None:
        self.table_name = table_name
        self.flow_key_col = flow_key_col
        self.strict_variable_length = strict_variable_length

    @staticmethod
    def _length_stats(lengths: np.ndarray) -> dict[str, float]:
        if lengths.size == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
            }
        return {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths, ddof=0)),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths)),
            "median": float(np.median(lengths)),
        }

    def _flow_lengths(self, df: pd.DataFrame) -> np.ndarray:
        if df is None or len(df) == 0:
            return np.array([], dtype=np.int64)
        if self.flow_key_col not in df.columns:
            raise KeyError(
                f"flow_key_col '{self.flow_key_col}' not in table "
                f"'{self.table_name}' columns: {list(df.columns)}"
            )
        counts = df.groupby(self.flow_key_col, dropna=False).size().to_numpy()
        return counts.astype(np.int64)

    def compute(
        self,
        real_data: dict[str, pd.DataFrame],
        synth_data: dict[str, pd.DataFrame],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute SequenceLength Similarity.

        Args:
            real_data: Mapping from table name to real DataFrame.
            synth_data: Mapping from table name to synthetic DataFrame.
            schema: Seq2Synth schema (unused here but kept for interface
                consistency).

        Returns:
            ``{"score": 1 - D_KS, "details": {...}}`` with diagnostics, or
            ``{"score": np.nan, "details": {"reason": ...}}`` when not
            applicable.
        """
        del schema  # not needed; signature parity with sibling metrics

        if self.table_name not in real_data or self.table_name not in synth_data:
            return {
                "score": np.nan,
                "details": {
                    "reason": f"table '{self.table_name}' missing from inputs",
                },
            }

        real_lengths = self._flow_lengths(real_data[self.table_name])
        synth_lengths = self._flow_lengths(synth_data[self.table_name])

        if real_lengths.size == 0 or synth_lengths.size == 0:
            return {
                "score": np.nan,
                "details": {
                    "reason": "empty flow set",
                    "n_real_flows": int(real_lengths.size),
                    "n_synth_flows": int(synth_lengths.size),
                },
            }

        real_var = float(np.var(real_lengths))
        synth_var = float(np.var(synth_lengths))

        if self.strict_variable_length and real_var == 0.0 and synth_var == 0.0:
            return {
                "score": np.nan,
                "details": {
                    "reason": "fixed-length data",
                    "n_real_flows": int(real_lengths.size),
                    "n_synth_flows": int(synth_lengths.size),
                    "real_length_stats": self._length_stats(real_lengths),
                    "synth_length_stats": self._length_stats(synth_lengths),
                },
            }

        # Handle degenerate (both-constant) distributions explicitly so the
        # outcome is well-defined regardless of scipy version behavior.
        if real_var == 0.0 and synth_var == 0.0:
            if real_lengths[0] == synth_lengths[0]:
                ks_stat = 0.0
                ks_pvalue = 1.0
            else:
                ks_stat = 1.0
                ks_pvalue = 0.0
        else:
            ks_result = stats.ks_2samp(real_lengths, synth_lengths)
            ks_stat = float(ks_result.statistic)
            ks_pvalue = float(ks_result.pvalue)

        score = float(1.0 - ks_stat)
        score = max(0.0, min(1.0, score))  # numerical safety

        details: dict[str, Any] = {
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "real_length_stats": self._length_stats(real_lengths),
            "synth_length_stats": self._length_stats(synth_lengths),
            "n_real_flows": int(real_lengths.size),
            "n_synth_flows": int(synth_lengths.size),
        }
        logger.debug(
            "SequenceLengthSimilarity: table=%s D_KS=%.4f score=%.4f",
            self.table_name,
            ks_stat,
            score,
        )
        return {"score": score, "details": details}
