"""Base class for structural metrics (Seq2Synth paper, Section 3.3.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStructuralMetric(ABC):
    """Abstract base class for Seq2Synth structural metrics.

    Concrete subclasses must implement :meth:`compute` and return a dictionary
    with a ``"score"`` entry in ``[0, 1]`` (or ``np.nan`` when not applicable)
    and a ``"details"`` entry holding diagnostics.
    """

    @abstractmethod
    def compute(
        self,
        real_data: dict[str, pd.DataFrame],
        synth_data: dict[str, pd.DataFrame],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute the metric.

        Args:
            real_data: Mapping from table name to the real DataFrame.
            synth_data: Mapping from table name to the synthetic DataFrame.
            schema: Seq2Synth schema description.

        Returns:
            A dict ``{"score": float, "details": dict}``. ``score`` is in
            ``[0, 1]`` with higher meaning higher fidelity, or ``np.nan`` if
            the metric is not applicable for the given inputs.
        """
