"""
Timestamp Metrics Calculator for Synthetic Relational Time Series Data.

Metrics
-------
Applicable to ALL datasets:
  1. TemporalOrderConsistency      - Absence of temporal order violations
  2. TimestampUniqueness           - Absence of duplicate consecutive timestamps
  3. TrajectoryDurationSimilarity  - KS complement on per-trajectory durations
  4. TemporalRangeCompliance       - Fraction of timestamps inside Omega_real

Regular series only:
  5. RegularityConsistency         - Violation rate from fixed interval c
  6. GridCompleteness              - Fraction of canonical grid points covered
     GridSupportCoverage           - Fraction of real grid support hit
     GridMultiplicityPreservation  - Occupancy preservation on hit support

Irregular series only:
  7. TimeIntervalDistributionSim   - KS complement on inter-arrival ECDFs
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV or Parquet file into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the input file. Must have a ``.csv`` or ``.parquet`` extension.
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def convert_time_column(
    df: pd.DataFrame, time_col: str, time_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse the time column into datetime or numeric values in-place on a copy.

    If the column is already numeric it is returned unchanged. Otherwise,
    datetime parsing is attempted (using ``time_format`` if provided). If
    all values fail datetime parsing, falls back to numeric coercion.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_col : str
        Name of the column to convert.
    time_format : str, optional
        strftime format string for datetime parsing.
    """
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[time_col]):
        return df
    if time_format is not None:
        converted = pd.to_datetime(df[time_col], format=time_format, errors="coerce")
        if converted.isna().any():
            fallback = pd.to_datetime(df[time_col], errors="coerce")
            converted = converted.fillna(fallback)
    else:
        converted = pd.to_datetime(df[time_col], errors="coerce")
    if converted.isna().all():
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df[time_col] = converted
    return df


def to_numeric_time(series: pd.Series) -> np.ndarray:
    """
    Convert a time Series to a float64 array of seconds (or raw float).

    Datetime series are converted to Unix seconds. Numeric series are cast
    to float directly.

    Parameters
    ----------
    series : pd.Series
        Series of datetime64 or numeric timestamps.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series.astype("int64") / 1e9).values
    return series.astype(float).values


def compute_intervals(
    df: pd.DataFrame, key_col: str, time_col: str, sort: bool = True
) -> np.ndarray:
    """
    Compute consecutive time differences across all trajectories.

    For each trajectory (group identified by ``key_col``), sorts by time
    and computes first-order differences. Returns the concatenated array
    of all inter-arrival intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    key_col : str
        Column identifying individual trajectories.
    time_col : str
        Timestamp column.
    sort : bool
        Whether to sort each group by ``time_col`` before differencing.
    """
    all_intervals = []
    for _, group in df.groupby(key_col, sort=False):
        if len(group) < 2:
            continue
        if sort:
            group = group.sort_values(time_col)
        times = to_numeric_time(group[time_col])
        all_intervals.extend(np.diff(times))
    return np.array(all_intervals)


def detect_periodicity(intervals: np.ndarray) -> Tuple[str, Optional[float]]:
    """
    Classify a series as Regular or Irregular based on interval coefficient of variation.

    A series is considered Regular if the CV (std / mean) of its inter-arrival
    intervals is below 0.01. Returns the classification string and the median
    interval for Regular series (``None`` for Irregular).

    Parameters
    ----------
    intervals : np.ndarray
        Array of consecutive time differences across all trajectories.
    """
    if len(intervals) == 0:
        return "Irregular", None
    mean_interval = np.mean(intervals)
    if mean_interval == 0:
        return "Irregular", None
    cv = np.std(intervals) / mean_interval
    if cv < 0.01:
        return "Regular", float(np.median(intervals))
    return "Irregular", None


def detect_time_representation(
    df: pd.DataFrame, key_col: str, time_col: str
) -> str:
    """
    Detect whether timestamps are Absolute (calendar) or Relative (elapsed).

    Inspects the first timestamp of each trajectory. If all trajectories
    start at the same value or at zero, the series is classified as
    ``"Relative"``; otherwise ``"Absolute"``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    key_col : str
        Column identifying individual trajectories.
    time_col : str
        Timestamp column.
    """
    first_timestamps = []
    for _, group in df.groupby(key_col, sort=False):
        if len(group) == 0:
            continue
        sorted_group = group.sort_values(time_col)
        first_timestamps.append(sorted_group[time_col].iloc[0])

    if not first_timestamps:
        return "Absolute"

    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        first_numeric = np.array([ts.value / 1e9 for ts in first_timestamps])
    else:
        first_numeric = np.array([float(ts) for ts in first_timestamps])

    if np.all(first_numeric == first_numeric[0]) or np.all(first_numeric == 0):
        return "Relative"
    return "Absolute"


def metric_temporal_order_consistency(
    synth_df: pd.DataFrame, key_col: str, time_col: str
) -> float:
    """
    Compute Temporal Order Consistency for synthetic data.

    Measures the fraction of consecutive timestamp pairs within each
    synthetic trajectory that are non-decreasing. A value of 1.0 means
    no ordering violations; lower values indicate temporal reversals.

    Parameters
    ----------
    synth_df : pd.DataFrame
        Synthetic DataFrame.
    key_col : str
        Column identifying individual trajectories.
    time_col : str
        Timestamp column.
    """
    total_intervals = 0
    violations = 0
    for _, group in synth_df.groupby(key_col, sort=False):
        if len(group) < 2:
            continue
        times = to_numeric_time(group[time_col])
        intervals = np.diff(times)
        total_intervals += len(intervals)
        violations += int(np.sum(intervals < 0))

    if total_intervals == 0:
        return 1.0
    return 1.0 - violations / total_intervals


def metric_timestamp_uniqueness(synth_intervals: np.ndarray) -> float:
    """
    Compute Timestamp Uniqueness for synthetic data.

    Measures the fraction of consecutive timestamp pairs that are strictly
    increasing (i.e., no duplicate consecutive timestamps). A value of 1.0
    means no duplicates; lower values indicate repeated timestamps.

    Parameters
    ----------
    synth_intervals : np.ndarray
        Array of consecutive time differences from the synthetic data.
    """
    if len(synth_intervals) == 0:
        return 1.0
    duplication_rate = np.sum(synth_intervals == 0) / len(synth_intervals)
    return 1.0 - duplication_rate


def metric_trajectory_duration_similarity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_col: str,
    time_col: str,
) -> float:
    """
    Compute Trajectory Duration Similarity between real and synthetic data.

    Computes the duration (last − first timestamp) of each trajectory in
    both real and synthetic data, then returns the KS complement
    ``1 - KS`` comparing the two duration distributions.

    Higher values indicate more similar trajectory lengths. Returns 0.0
    if either dataset has no trajectories.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real DataFrame.
    synth_df : pd.DataFrame
        Synthetic DataFrame.
    key_col : str
        Column identifying individual trajectories.
    time_col : str
        Timestamp column.
    """
    def compute_durations(df: pd.DataFrame) -> np.ndarray:
        durations = []
        for _, group in df.groupby(key_col, sort=False):
            times = to_numeric_time(group.sort_values(time_col)[time_col])
            if len(times) >= 1:
                durations.append(times[-1] - times[0])
        return np.array(durations)

    real_dur = compute_durations(real_df)
    synth_dur = compute_durations(synth_df)
    if len(real_dur) == 0 or len(synth_dur) == 0:
        return 0.0
    ks_stat, _ = stats.ks_2samp(real_dur, synth_dur)
    return 1.0 - ks_stat


def metric_temporal_range_compliance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_col: str,
    time_col: str,
    time_repr: str,
) -> float:
    """
    Compute Temporal Range Compliance for synthetic data.

    Measures the fraction of synthetic timestamps that fall within the
    observed temporal range of the real data (Ω_real), using the timestamps as
    generated. A shifted synthetic start point is therefore penalized when it
    falls outside the real range.

    Returns a score in [0, 1] where 1.0 means all synthetic timestamps comply.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real DataFrame.
    synth_df : pd.DataFrame
        Synthetic DataFrame.
    key_col : str
        Column identifying individual trajectories.
    time_col : str
        Timestamp column.
    time_repr : str
        ``"Absolute"`` or ``"Relative"``.
    """
    if time_repr == "Relative" and pd.api.types.is_datetime64_any_dtype(synth_df[time_col]):
        raise ValueError(
            f"Column '{time_col}' was parsed as datetime but time_repr is "
            "'Relative'. Relative series must use numeric elapsed values "
            "(e.g. integer cycle index), not calendar dates. "
            "Check that the column contains numeric data."
        )

    real_times_all = to_numeric_time(real_df[time_col])

    omega_low = float(np.nanmin(real_times_all))
    omega_high = float(np.nanmax(real_times_all))
    synth_times_all = to_numeric_time(synth_df[time_col])

    n_synth = len(synth_times_all)
    if n_synth == 0:
        return 1.0

    out_of_range = int(np.sum((synth_times_all < omega_low) | (synth_times_all > omega_high)))
    violation_rate = out_of_range / n_synth

    return float(np.clip(1.0 - violation_rate, 0.0, 1.0))


def metric_regularity_consistency(
    synth_intervals: np.ndarray,
    representative_interval: float,
) -> float:
    """
    Compute Regularity Consistency for regular synthetic series.

    Measures the fraction of synthetic inter-arrival intervals that deviate
    from the real representative interval by no more than 1 % (epsilon).
    A value of 1.0 means every interval matches the expected cadence.

    Only meaningful for Regular series.

    Parameters
    ----------
    synth_intervals : np.ndarray
        Array of consecutive time differences from the synthetic data.
    representative_interval : float
        Expected fixed interval derived from the real data (typically the median).
    """
    if len(synth_intervals) == 0:
        return 0.0
    epsilon = representative_interval * 0.01
    violations = np.sum(np.abs(synth_intervals - representative_interval) > epsilon)
    return 1.0 - violations / len(synth_intervals)


def metric_grid_completeness(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_col: str,
    time_col: str,
    representative_interval: float,
    time_repr: str = "Absolute",
) -> Optional[float]:
    """
    Compute Grid Completeness for regular synthetic series.

    Uses a population-level multiset over canonical grid positions, so synthetic
    identifiers do not need to match real identifiers. Real trajectories define
    the reference occupancy, and synthetic timestamps are snapped to the same
    grid before comparing occupancy counts.

    Only meaningful for Regular series. Returns ``None`` if no grid points
    exist in the real data.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real DataFrame.
    synth_df : pd.DataFrame
        Synthetic DataFrame.
    key_col : str
        Column identifying individual trajectories.
    time_col : str
        Timestamp column.
    representative_interval : float
        Fixed interval used to define the canonical grid.
    time_repr : str
        ``"Absolute"`` or ``"Relative"``.
    """
    details = metric_grid_completeness_details(
        real_df,
        synth_df,
        key_col,
        time_col,
        representative_interval,
        time_repr=time_repr,
    )
    return details["grid_completeness"]


def _snapped_grid_positions(times: pd.Series, representative_interval: float) -> np.ndarray:
    times_num = to_numeric_time(times)
    times_num = times_num[np.isfinite(times_num)]
    if len(times_num) == 0:
        return np.array([], dtype=int)
    return np.round(times_num / representative_interval).astype(int)


def _grid_multiset(
    df: pd.DataFrame,
    key_col: str,
    time_col: str,
    representative_interval: float,
    canonical_from_start: bool,
) -> Counter:
    grid_counts: Counter = Counter()
    sorted_df = df.sort_values([key_col, time_col]).drop_duplicates(subset=[key_col, time_col])
    for _, group in sorted_df.groupby(key_col, sort=False):
        snapped = _snapped_grid_positions(group[time_col], representative_interval)
        if len(snapped) == 0:
            continue
        if canonical_from_start:
            positions = set(range(int(snapped[0]), int(snapped[0]) + len(np.unique(snapped))))
        else:
            positions = set(snapped)
        grid_counts.update(positions)
    return grid_counts


def metric_grid_completeness_details(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_col: str,
    time_col: str,
    representative_interval: float,
    time_repr: str = "Absolute",
) -> Dict[str, Optional[float]]:
    """
    Return population-level Grid Completeness and diagnostic decompositions.
    """
    real_counts = _grid_multiset(
        real_df,
        key_col,
        time_col,
        representative_interval,
        canonical_from_start=True,
    )
    synth_counts = _grid_multiset(
        synth_df,
        key_col,
        time_col,
        representative_interval,
        canonical_from_start=False,
    )

    total_real = sum(real_counts.values())
    if total_real == 0:
        return {
            "grid_completeness": None,
            "grid_support_coverage": None,
            "grid_multiplicity_preservation": None,
        }

    real_support = set(real_counts)
    synth_support = set(synth_counts)
    covered = sum(min(real_counts[g], synth_counts.get(g, 0)) for g in real_support)
    support_overlap = real_support & synth_support
    support_coverage = len(support_overlap) / len(real_support) if real_support else None
    real_mass_on_overlap = sum(real_counts[g] for g in support_overlap)
    multiplicity_preservation = (
        covered / real_mass_on_overlap if real_mass_on_overlap > 0 else None
    )

    return {
        "grid_completeness": float(covered / total_real),
        "grid_support_coverage": float(support_coverage) if support_coverage is not None else None,
        "grid_multiplicity_preservation": (
            float(multiplicity_preservation) if multiplicity_preservation is not None else None
        ),
    }


def metric_time_interval_distribution(
    real_intervals: np.ndarray, synth_intervals: np.ndarray
) -> float:
    """
    Compute Time Interval Distribution Similarity for irregular series.

    Runs a two-sample KS test on the inter-arrival interval distributions
    of real and synthetic data and returns ``1 - KS`` as the similarity score.

    Higher values indicate better fidelity of the inter-arrival time
    distribution. Only meaningful for Irregular series. Returns 0.0 if
    either array is empty.

    Parameters
    ----------
    real_intervals : np.ndarray
        Inter-arrival intervals from the real data.
    synth_intervals : np.ndarray
        Inter-arrival intervals from the synthetic data.
    """
    if len(real_intervals) == 0 or len(synth_intervals) == 0:
        return 0.0
    ks_stat, _ = stats.ks_2samp(real_intervals, synth_intervals)
    return 1.0 - ks_stat


_NA_REGULAR = "N/A (Irregular series only)"
_NA_IRREGULAR = "N/A (Regular series only)"


def compute_data_characteristics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    key_col: str,
    time_col: str,
    real_intervals: np.ndarray,
    periodicity: str,
    time_repr_override: Optional[str] = None,
) -> Tuple[Dict[str, Any], str, Optional[float], str]:
    """
    Assemble dataset characteristics used for metric computation.

    Derives the representative interval (median of real intervals for Regular
    series) and the time representation (auto-detected or user-overridden),
    then packages these along with row and flow counts into a summary dict.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real DataFrame.
    synth_df : pd.DataFrame
        Synthetic DataFrame.
    key_col : str
        Column identifying individual trajectories.
    time_col : str
        Timestamp column.
    real_intervals : np.ndarray
        Pre-computed inter-arrival intervals from the real data.
    periodicity : str
        ``"Regular"`` or ``"Irregular"``.
    time_repr_override : str, optional
        If provided, overrides auto-detection with ``"Absolute"`` or
        ``"Relative"``.
    """
    if periodicity == "Regular" and len(real_intervals) > 0:
        representative_interval = float(np.median(real_intervals))
        representative_interval = representative_interval if representative_interval > 0 else None
    else:
        representative_interval = None

    if time_repr_override is not None:
        time_repr = time_repr_override
        time_repr_source = "user-specified"
    else:
        time_repr = detect_time_representation(real_df, key_col, time_col)
        time_repr_source = "auto-detected"

    characteristics: Dict[str, Any] = {
        "periodicity": periodicity,
        "time_representation": time_repr,
        "time_representation_source": time_repr_source,
        "real_num_flows": int(real_df[key_col].nunique()),
        "synth_num_flows": int(synth_df[key_col].nunique()),
        "real_total_observations": len(real_df),
        "synth_total_observations": len(synth_df),
    }
    if periodicity == "Regular" and representative_interval is not None:
        characteristics["representative_interval"] = representative_interval

    return characteristics, periodicity, representative_interval, time_repr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Timestamp Metrics for synthetic time series data."
    )
    parser.add_argument("--real_data", required=True, help="Path to real data (CSV or Parquet)")
    parser.add_argument("--synth_data", required=True, help="Path to synthetic data (CSV or Parquet)")
    parser.add_argument("--key_col", required=True, help="Column identifying flows (e.g. store_id, user_id)")
    parser.add_argument("--time_col", required=True, help="Timestamp column name")
    parser.add_argument("--output", default="results.json", help="Output JSON file (default: results.json)")
    parser.add_argument(
        "--periodicity",
        choices=["Regular", "Irregular"],
        required=True,
        help="Periodicity of the dataset.",
    )
    parser.add_argument(
        "--time_representation",
        choices=["Absolute", "Relative"],
        required=True,
        help="Time representation of the dataset.",
    )
    parser.add_argument(
        "--time_col_format",
        default=None,
        help="strftime format string for parsing the time column.",
    )
    args = parser.parse_args()

    print(f"Loading real data  from {args.real_data} ...")
    real_df = load_data(args.real_data)
    print(f"Loading synth data from {args.synth_data} ...")
    synth_df = load_data(args.synth_data)

    for df, label in [(real_df, "Real"), (synth_df, "Synth")]:
        if args.time_col not in df.columns:
            if "Date_month" in df.columns and "Date_day" in df.columns:
                df[args.time_col] = (
                    df["Date_month"].astype(int).map(lambda m: f"{m:02d}")
                    + "-"
                    + df["Date_day"].astype(int).map(lambda d: f"{d:02d}")
                )
                print(f"  {label}: constructed '{args.time_col}' from Date_month + Date_day")
            elif {"Year", "Month", "Day"}.issubset(df.columns):
                df[args.time_col] = (
                    (1900 + df["Year"].astype(int)).astype(str)
                    + "-"
                    + df["Month"].astype(int).map(lambda m: f"{m:02d}")
                    + "-"
                    + df["Day"].astype(int).map(lambda d: f"{d:02d}")
                )
                print(f"  {label}: constructed '{args.time_col}' from Year + Month + Day")
            else:
                raise KeyError(
                    f"{label} data has no '{args.time_col}' column and no "
                    "'Date_month'/'Date_day' or 'Year'/'Month'/'Day' columns "
                    "to construct it from"
                )

    real_df = convert_time_column(real_df, args.time_col, args.time_col_format)
    synth_df = convert_time_column(synth_df, args.time_col, args.time_col_format)

    print("Computing time intervals ...")
    real_intervals = compute_intervals(real_df, args.key_col, args.time_col, sort=True)
    synth_intervals = compute_intervals(synth_df, args.key_col, args.time_col, sort=True)

    print("Detecting data characteristics ...")
    characteristics, periodicity, representative_interval, time_repr = compute_data_characteristics(
        real_df,
        synth_df,
        args.key_col,
        args.time_col,
        real_intervals,
        periodicity=args.periodicity,
        time_repr_override=args.time_representation,
    )

    print("Computing metrics ...")
    is_regular = periodicity == "Regular"
    is_irregular = periodicity == "Irregular"

    metrics: Dict[str, Any] = {}
    metrics["TemporalOrderConsistency"] = metric_temporal_order_consistency(
        synth_df, args.key_col, args.time_col
    )
    metrics["TimestampUniqueness"] = metric_timestamp_uniqueness(synth_intervals)
    metrics["TrajectoryDurationSimilarity"] = metric_trajectory_duration_similarity(
        real_df, synth_df, args.key_col, args.time_col
    )
    metrics["TemporalRangeCompliance"] = metric_temporal_range_compliance(
        real_df, synth_df, args.key_col, args.time_col, time_repr
    )

    if is_regular and representative_interval is not None:
        metrics["RegularityConsistency"] = metric_regularity_consistency(
            synth_intervals, representative_interval
        )
        grid_details = metric_grid_completeness_details(
            real_df,
            synth_df,
            args.key_col,
            args.time_col,
            representative_interval,
            time_repr=time_repr,
        )
        metrics["GridCompleteness"] = (
            grid_details["grid_completeness"]
            if grid_details["grid_completeness"] is not None
            else "N/A"
        )
        metrics["GridSupportCoverage"] = (
            grid_details["grid_support_coverage"]
            if grid_details["grid_support_coverage"] is not None
            else "N/A"
        )
        metrics["GridMultiplicityPreservation"] = (
            grid_details["grid_multiplicity_preservation"]
            if grid_details["grid_multiplicity_preservation"] is not None
            else "N/A"
        )
    else:
        metrics["RegularityConsistency"] = _NA_IRREGULAR
        metrics["GridCompleteness"] = _NA_IRREGULAR
        metrics["GridSupportCoverage"] = _NA_IRREGULAR
        metrics["GridMultiplicityPreservation"] = _NA_IRREGULAR

    if is_irregular:
        metrics["TimeIntervalDistributionSim"] = metric_time_interval_distribution(
            real_intervals, synth_intervals
        )
    else:
        metrics["TimeIntervalDistributionSim"] = _NA_REGULAR

    metrics_rounded = {
        key: (round(value, 6) if isinstance(value, float) else value)
        for key, value in metrics.items()
    }

    output = {
        "data_characteristics": characteristics,
        "summary": metrics_rounded,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    width = 60
    print("\n" + "=" * width)
    print("TIMESTAMP METRICS RESULTS")
    print("=" * width)
    print("\nData Characteristics:")
    for key, value in characteristics.items():
        print(f"  {key}: {value}")
    print("\nMetrics (All datasets):")
    for key in [
        "TemporalOrderConsistency",
        "TimestampUniqueness",
        "TrajectoryDurationSimilarity",
        "TemporalRangeCompliance",
    ]:
        print(f"  {key}: {metrics_rounded[key]}")
    print("\nMetrics (Regular only):")
    for key in [
        "RegularityConsistency",
        "GridCompleteness",
        "GridSupportCoverage",
        "GridMultiplicityPreservation",
    ]:
        print(f"  {key}: {metrics_rounded[key]}")
    print("\nMetrics (Irregular only):")
    print(f"  TimeIntervalDistributionSim: {metrics_rounded['TimeIntervalDistributionSim']}")
    print("=" * width)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
