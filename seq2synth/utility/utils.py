from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_period_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.model_selection import train_test_split
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute


@dataclass(frozen=True)
class TsDatasetBundle:
    dataset: str
    entity_col: str
    label_col: str
    task: str
    real_X: pd.DataFrame
    real_y: pd.Series
    syn_by_method: dict[str, tuple[pd.DataFrame, pd.Series]]
    metadata: dict


@dataclass(frozen=True)
class TsRepresentationBundle:
    dataset: str
    entity_col: str
    real_X: pd.DataFrame
    syn_by_method: dict[str, pd.DataFrame]
    metadata: dict


DEFAULT_PARAM_GRID = {
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.03, 0.05, 0.10],
    "min_child_weight": [1, 3],
}


def parseCsvArg(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parseIntList(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if str(x).strip() != ""]


def parseFloatList(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if str(x).strip() != ""]


def parseBool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def parseRatiosFromBin(bin_size: float, ratio_max: float = 1.0) -> list[float]:
    bin_size = float(bin_size)
    ratio_max = float(ratio_max)
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")
    if ratio_max < 0:
        raise ValueError("ratio_max must be >= 0")
    out = []
    x = 0.0
    tol = bin_size / 1000
    while x <= ratio_max + tol:
        out.append(round(float(x), 10))
        x += bin_size
    if abs(out[-1] - ratio_max) > 1e-6:
        out.append(round(float(ratio_max), 10))
    return out


def loadJson(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def readJson(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolveRuntimePath(path: str | Path, anchor_file: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    anchor_dir = Path(anchor_file).resolve().parent
    relative_to_parent = (anchor_dir.parent / candidate).resolve()
    if relative_to_parent.exists():
        return relative_to_parent
    relative_to_anchor = (anchor_dir / candidate).resolve()
    if relative_to_anchor.exists():
        return relative_to_anchor
    return cwd_candidate


def loadLocalConfig(config_name: str, anchor_file: str | Path) -> dict:
    anchor_dir = Path(anchor_file).resolve().parent
    config_path = anchor_dir / "config" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return readJson(config_path)


def discoverDatasets(data_root: Path) -> list[str]:
    original_root = data_root / "original"
    if not original_root.exists():
        return []
    return sorted(path.name for path in original_root.iterdir() if path.is_dir())


def discoverMethodsForDataset(data_root: Path, dataset: str) -> list[str]:
    base = data_root / "synthetic" / dataset
    if not base.exists():
        return []
    return sorted(path.name for path in base.iterdir() if path.is_dir())


def castNumeric(series: pd.Series, rep: str | None):
    s = pd.to_numeric(series, errors="coerce")
    if rep and rep.startswith("Int"):
        return s.round().astype("Int32")
    return s


def booleanToInt(series: pd.Series):
    if series.dtype == object:
        s = series.astype(str).str.lower().map({"true": 1, "false": 0})
        return pd.to_numeric(s, errors="coerce").fillna(0).astype("int8")
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int8")


def normalizeCategorical(series: pd.Series) -> pd.Series:
    return series.astype("string").where(series.notna(), pd.NA)


def buildCategoricalMaps(df: pd.DataFrame, table_meta: dict) -> dict[str, dict[str, int]]:
    cols_meta = table_meta.get("columns", {})
    category_maps: dict[str, dict[str, int]] = {}
    for col, spec in cols_meta.items():
        if col not in df.columns or spec.get("sdtype") != "categorical":
            continue
        normalized = normalizeCategorical(df[col])
        categories = sorted(normalized.dropna().unique().tolist())
        category_maps[col] = {value: idx for idx, value in enumerate(categories)}
    return category_maps


def categoricalToCode(series: pd.Series, mapping: Optional[dict[str, int]] = None, unknown_value: int = -1):
    normalized = normalizeCategorical(series)
    if mapping is None:
        categories = sorted(normalized.dropna().unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
    codes = normalized.map(mapping)
    return codes.fillna(unknown_value).astype("int32")


def nondatetimeToDatetime(series: pd.Series, fmt: str) -> pd.Series:
    digit_width = {"%Y": 4, "%y": 2, "%m": 2, "%d": 2, "%H": 2, "%M": 2, "%S": 2}
    if is_datetime64_any_dtype(series) or is_datetime64tz_dtype(series) or is_period_dtype(series):
        raise TypeError(f"Series dtype is already datetime/period: {series.dtype}.")
    if is_timedelta64_dtype(series):
        raise TypeError("Timedelta dtype is not supported for datetime parsing.")
    if is_bool_dtype(series):
        raise TypeError("Boolean dtype is not supported for datetime parsing.")
    if not (is_numeric_dtype(series) or is_object_dtype(series) or is_string_dtype(series) or is_categorical_dtype(series)):
        raise TypeError(f"Unsupported dtype for datetime parsing: {series.dtype}")

    tokens = re.findall(r"%[A-Za-z]", fmt)
    if not tokens:
        raise ValueError(f"Invalid fmt={fmt!r}")
    unsupported = [t for t in tokens if t not in digit_width]
    if unsupported:
        raise ValueError(f"Unsupported directive(s) in fmt={fmt!r}: {unsupported}")

    s = series
    if is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce").round().astype("Int64").astype("string")
    else:
        s = s.astype("string")
    s = s.str.strip()

    if re.sub(r"%[A-Za-z]", "", fmt) == "":
        width = sum(digit_width[t] for t in tokens)
        digits = s.str.replace(r"\D", "", regex=True)
        if fmt.startswith(("%Y", "%y")):
            digits = digits.str.slice(0, width)
        elif fmt.endswith(("%Y", "%y")):
            digits = digits.str.slice(-width)
        else:
            digits = digits.str.slice(0, width)
        digits = digits.str.zfill(width)
        out = pd.to_datetime(digits, format=fmt, errors="coerce")
    else:
        out = pd.to_datetime(s, format=fmt, errors="coerce")
    miss = out.isna() & s.notna()
    if miss.any():
        out.loc[miss] = pd.to_datetime(s[miss], errors="coerce")
    return out


def repairDatetimeSeries(series: pd.Series, reference_series: pd.Series, fmt: str | None) -> pd.Series:
    if is_datetime64_any_dtype(series) or is_datetime64tz_dtype(series):
        repaired = pd.to_datetime(series, errors="coerce")
    else:
        repaired = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        if fmt:
            parsed_with_format = pd.to_datetime(series, format=fmt, errors="coerce")
            repaired.loc[parsed_with_format.notna()] = parsed_with_format.loc[parsed_with_format.notna()]
        still_missing = repaired.isna() & series.notna()
        if still_missing.any():
            parsed_generic = pd.to_datetime(series.loc[still_missing], errors="coerce")
            repaired.loc[parsed_generic.notna()] = parsed_generic.loc[parsed_generic.notna()]
        still_missing = repaired.isna() & series.notna()
        if still_missing.any():
            repaired.loc[still_missing] = mapRankedDatesToReference(series.loc[still_missing], reference_series)
    unresolved = series.notna() & repaired.isna()
    if unresolved.any():
        raise ValueError(f"Failed to repair datetime column {series.name!r}.")
    return repaired


def mapRankedDatesToReference(series: pd.Series, reference_series: pd.Series) -> pd.Series:
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    raw = series[series.notna()]
    ref = pd.to_datetime(reference_series, errors="coerce").dropna()
    if raw.empty or ref.empty:
        return out
    raw_unique = pd.Index(pd.Series(raw.unique())).sort_values()
    ref_unique = pd.Index(pd.Series(ref.unique())).sort_values()
    if len(raw_unique) == 1:
        target_positions = np.array([0], dtype=int)
    else:
        target_positions = np.rint(np.linspace(0, len(ref_unique) - 1, num=len(raw_unique))).astype(int)
    mapping = dict(zip(raw_unique.tolist(), ref_unique.take(target_positions).tolist()))
    mapped = raw.map(mapping)
    out.loc[mapped.index] = pd.to_datetime(mapped, errors="coerce")
    return out


def repairDatetimeColumns(df: pd.DataFrame, reference_df: pd.DataFrame, table_meta: dict) -> pd.DataFrame:
    repaired = df.copy()
    for col, spec in table_meta.get("columns", {}).items():
        if spec.get("sdtype") != "datetime" or col not in repaired.columns or col not in reference_df.columns:
            continue
        repaired[col] = repairDatetimeSeries(repaired[col], reference_df[col], spec.get("datetime_format"))
    return repaired


def encodeTables(df: pd.DataFrame, table_meta: dict, categorical_maps: Optional[dict[str, dict[str, int]]] = None) -> pd.DataFrame:
    df = df.copy()
    cols_meta = table_meta.get("columns", {})
    for col, spec in cols_meta.items():
        if col not in df.columns:
            continue
        sdtype = spec.get("sdtype")
        if sdtype == "numerical":
            df[col] = castNumeric(df[col], spec.get("computer_representation"))
        elif sdtype == "boolean":
            df[col] = booleanToInt(df[col])
        elif sdtype == "categorical":
            df[col] = categoricalToCode(df[col], mapping=(categorical_maps or {}).get(col))
        elif sdtype == "datetime":
            if is_datetime64_any_dtype(df[col]) or is_datetime64tz_dtype(df[col]) or is_period_dtype(df[col]):
                continue
            fmt = spec.get("datetime_format")
            if not fmt:
                raise ValueError(f"datetime_format missing for datetime column: {col}")
            df[col] = nondatetimeToDatetime(df[col], fmt)
        elif sdtype == "id":
            continue
    return df


def readCsvWithReference(base: Path, reference_base: Path, filename: str, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = pd.read_csv(base / filename, **kwargs)
    reference = current.copy() if base == reference_base else pd.read_csv(reference_base / filename, **kwargs)
    return current, reference


def encodeWithReference(df: pd.DataFrame, reference_df: pd.DataFrame, table_meta: dict) -> pd.DataFrame:
    df = repairDatetimeColumns(df, reference_df, table_meta)
    category_maps = buildCategoricalMaps(reference_df, table_meta)
    return encodeTables(df, table_meta, categorical_maps=category_maps)


def resolveBase(data_root: Path, dataset_name: str, split: str, method: str = "", run: str = "1", sample: str = "sample1") -> Path:
    base = data_root / ("original" if split == "original" else "synthetic") / dataset_name
    if split != "original":
        base = base / method / str(run) / str(sample)
    if not base.exists():
        raise FileNotFoundError(f"Dataset path not found: {base}")
    return base


def resolveMetadataPath(data_root: Path, dataset: str) -> Path:
    base = data_root / "original" / dataset
    for name in ("metadata.json", "metadata_filtered.json", "metadata_v0.json"):
        path = base / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Metadata file not found for dataset={dataset!r} under {base}")


def loadDatasetMetadata(data_root: Path, dataset: str) -> dict:
    return readJson(resolveMetadataPath(data_root, dataset))


def resolveTableFilename(base: Path, table_name: str) -> str:
    for suffix in (".csv", ".csv.gz"):
        filename = f"{table_name}{suffix}"
        if (base / filename).exists():
            return filename
    raise FileNotFoundError(f"Table file not found for table={table_name!r} under {base}")


def loadSingleTablePair(
    dataset: str,
    table_name: str,
    method: str,
    data_root: Path,
    run: str = "1",
    sample: str = "sample1",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata = loadDatasetMetadata(data_root, dataset)
    tables = metadata.get("tables", {})
    if table_name not in tables:
        raise ValueError(f"Unknown table={table_name!r} for dataset={dataset!r}")
    table_meta = tables[table_name]
    original_base = resolveBase(data_root, dataset, "original")
    synthetic_base = resolveBase(data_root, dataset, "synthetic", method, run, sample)
    filename = resolveTableFilename(original_base, table_name)
    if not (synthetic_base / filename).exists():
        raise FileNotFoundError(
            f"Synthetic table file not found for dataset={dataset!r}, method={method!r}, table={table_name!r}: {synthetic_base / filename}"
        )
    real_df, real_ref = readCsvWithReference(original_base, original_base, filename)
    syn_df, syn_ref = readCsvWithReference(synthetic_base, original_base, filename)
    real_encoded = encodeWithReference(real_df, real_ref, table_meta)
    syn_encoded = encodeWithReference(syn_df, syn_ref, table_meta)
    return real_encoded, syn_encoded, table_meta


def resolveXgbNJobs(default: int = -1) -> int:
    raw = os.environ.get("UTILITY_XGB_NJOBS", "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value != 0 else default


def iterParamGrid(param_grid: dict[str, list[float | int]] | None = None) -> list[dict]:
    grid = param_grid or DEFAULT_PARAM_GRID
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def selectSeededParams(seed: int, param_grid: dict[str, list[float | int]] | None = None) -> dict:
    param_candidates = iterParamGrid(param_grid)
    index = ((int(seed) * 2654435761) & 0xFFFFFFFF) % len(param_candidates)
    return param_candidates[index]


def prepareFeatures(train: pd.DataFrame, test: pd.DataFrame, target_col: str, date_col: str, id_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    drop_cols = [c for c in [target_col, date_col, id_col] if c]
    train_X = train.drop(columns=drop_cols, errors="ignore").copy()
    test_X = test.drop(columns=drop_cols, errors="ignore").copy()
    train_y = train[target_col]
    test_y = test[target_col]

    dt_cols = [c for c in train_X.columns if is_datetime64_any_dtype(train_X[c]) or is_datetime64tz_dtype(train_X[c])]
    for c in dt_cols:
        train_X[c] = train_X[c].dt.year * 100 + train_X[c].dt.month
        test_X[c] = test_X[c].dt.year * 100 + test_X[c].dt.month

    invalid_cols = [c for c in train_X.columns if not (is_numeric_dtype(train_X[c]) or is_bool_dtype(train_X[c]))]
    if invalid_cols:
        raise ValueError(f"Non-numeric feature columns remain after encoding: {invalid_cols}")
    return train_X, test_X, train_y, test_y


def binaryMetrics(test_y: pd.Series, pred: np.ndarray, proba: np.ndarray | None) -> dict:
    auc = np.nan
    if proba is not None and pd.Series(test_y).nunique(dropna=True) > 1:
        try:
            auc = float(roc_auc_score(test_y, proba))
        except ValueError:
            auc = np.nan
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(test_y, pred)),
        "f1": float(f1_score(test_y, pred, zero_division=0)),
        "recall": float(recall_score(test_y, pred, zero_division=0)),
        "precision": float(precision_score(test_y, pred, zero_division=0)),
    }


def multiclassMetrics(test_y: pd.Series, pred: np.ndarray, proba: np.ndarray | None) -> dict:
    auc = np.nan
    if proba is not None and pd.Series(test_y).nunique(dropna=True) > 1:
        try:
            auc = float(roc_auc_score(test_y, proba, multi_class="ovr", average="weighted"))
        except ValueError:
            auc = np.nan
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(test_y, pred)),
        "f1": float(f1_score(test_y, pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(test_y, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(test_y, pred, average="weighted", zero_division=0)),
    }


def regressionMetrics(test_y: pd.Series, pred_y: np.ndarray) -> dict:
    return {
        "mse": float(mean_squared_error(test_y, pred_y)),
        "mae": float(mean_absolute_error(test_y, pred_y)),
        "mape": float(mean_absolute_percentage_error(test_y, pred_y)),
        "r2": float(r2_score(test_y, pred_y)),
    }


def baselinePredictions(task: str, train_y: pd.Series, test_y: pd.Series) -> tuple[np.ndarray, np.ndarray | None]:
    if task == "regression":
        return np.full(len(test_y), float(train_y.mean())), None
    majority_label = train_y.mode(dropna=True).iloc[0]
    pred = np.full(len(test_y), majority_label)
    if train_y.nunique(dropna=True) <= 2:
        proba = np.full(len(test_y), float(majority_label == 1))
        return pred, proba
    return pred, None


def buildModel(task: str, train_y: pd.Series, seed: int, step: int, params: dict):
    common = {
        "n_estimators": int(step),
        "random_state": seed,
        "n_jobs": resolveXgbNJobs(),
        "tree_method": "hist",
        "verbosity": 0,
        **params,
    }
    if task == "classification":
        class_count = train_y.nunique(dropna=True)
        if class_count < 3:
            return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
        return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", num_class=int(class_count), **common)
    if task == "regression":
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="mae", **common)
    raise ValueError(f"Unknown task: {task}")


def evaluatePredictions(task: str, test_y: pd.Series, pred: np.ndarray, proba: np.ndarray | None) -> dict:
    if task == "regression":
        return regressionMetrics(test_y, pred)
    if pd.Series(test_y).nunique(dropna=True) < 3:
        return binaryMetrics(test_y, pred, proba)
    return multiclassMetrics(test_y, pred, proba)


def scoreMetrics(task: str, metrics: dict) -> float:
    if task == "regression":
        return -float(metrics["mse"])
    return float(metrics["f1"])


def fitAndPredict(task: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, seed: int, step: int, params: dict) -> tuple[np.ndarray, np.ndarray | None]:
    if train_X.shape[1] == 0 or train_y.nunique(dropna=True) < (2 if task == "classification" else 1):
        return baselinePredictions(task, train_y, pd.Series(index=test_X.index, dtype=train_y.dtype))
    if task == "classification":
        class_values = pd.Index(sorted(pd.Series(train_y).dropna().unique().tolist()))
        label_to_index = {label: idx for idx, label in enumerate(class_values)}
        index_to_label = {idx: label for label, idx in label_to_index.items()}
        encoded_train_y = train_y.map(label_to_index)
        model = buildModel(task=task, train_y=encoded_train_y, seed=seed, step=step, params=params)
        model.fit(train_X, encoded_train_y)
        proba = model.predict_proba(test_X)
        pred = pd.Series(model.predict(test_X)).map(index_to_label).to_numpy()
        if encoded_train_y.nunique(dropna=True) < 3:
            return pred, proba[:, 1]
        return pred, proba
    model = buildModel(task=task, train_y=train_y, seed=seed, step=step, params=params)
    model.fit(train_X, train_y)
    return model.predict(test_X), None


def selectBestParams(tune_train: pd.DataFrame, valid: pd.DataFrame, target_col: str, date_col: str, id_col: str, task: str, seed: int, step: int) -> tuple[dict, float]:
    param_candidates = iterParamGrid()
    if len(valid) == 0:
        return param_candidates[0], np.nan
    tune_X, valid_X, tune_y, valid_y = prepareFeatures(tune_train, valid, target_col, date_col, id_col)
    if tune_X.shape[1] == 0 or (task == "classification" and tune_y.nunique(dropna=True) < 2):
        return param_candidates[0], np.nan
    best_params = param_candidates[0]
    best_score = -np.inf
    for params in param_candidates:
        pred, proba = fitAndPredict(task=task, train_X=tune_X, train_y=tune_y, test_X=valid_X, seed=seed, step=step, params=params)
        metrics = evaluatePredictions(task, valid_y, pred, proba)
        score = scoreMetrics(task, metrics)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, float(best_score)


def fitDownstreamModel(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    tune_train: pd.DataFrame,
    target_col: str,
    date_col: str,
    id_col: str,
    task: str,
    seed: int,
    step: int = 50,
    tune_hyperparameters: bool = False,
    fixed_params: dict | None = None,
) -> dict:
    if len(train) == 0:
        raise ValueError("train is empty.")
    if len(test) == 0:
        raise ValueError("test is empty.")
    step = int(step)
    if fixed_params is not None:
        best_params = {
            "max_depth": int(fixed_params["max_depth"]),
            "learning_rate": float(fixed_params["fixed_learning_rate"] if "fixed_learning_rate" in fixed_params else fixed_params["learning_rate"]),
            "min_child_weight": int(fixed_params["fixed_min_child_weight"] if "fixed_min_child_weight" in fixed_params else fixed_params["min_child_weight"]),
        }
        valid_score = np.nan
    elif tune_hyperparameters and len(valid) > 0:
        best_params, valid_score = selectBestParams(tune_train, valid, target_col, date_col, id_col, task, seed, step)
    else:
        best_params = selectSeededParams(seed=seed)
        valid_score = np.nan
    train_X, test_X, train_y, test_y = prepareFeatures(train, test, target_col, date_col, id_col)
    if train_X.shape[1] == 0 or (task == "classification" and train_y.nunique(dropna=True) < 2):
        pred, proba = baselinePredictions(task, train_y, test_y)
    else:
        pred, proba = fitAndPredict(task, train_X, train_y, test_X, seed, step, best_params)
    metrics = evaluatePredictions(task, test_y, pred, proba)
    metrics["valid_score"] = valid_score
    metrics["step"] = step
    metrics["tuned"] = bool(fixed_params is None and tune_hyperparameters and len(valid) > 0)
    metrics["param_seed"] = int(seed)
    metrics["param_strategy"] = "fixed_params" if fixed_params is not None else ("validation_search" if metrics["tuned"] else "seeded_grid_pick")
    metrics["search_metric"] = "mse" if task == "regression" else "f1"
    metrics["grid_size"] = 1 if fixed_params is not None else len(iterParamGrid())
    metrics["selected_params"] = json.dumps(best_params, sort_keys=True)
    return metrics


def prepareFlowForTsfresh(flow_df: pd.DataFrame, entity_col: str, time_col: str, value_cols: list[str]) -> pd.DataFrame:
    work = flow_df[[entity_col, time_col, *value_cols]].copy()
    work = work[work[entity_col].notna() & work[time_col].notna()].copy()
    work = work.sort_values([entity_col, time_col], kind="mergesort").reset_index(drop=True)
    return work


def extractWideTsfresh(flow_df: pd.DataFrame, entity_col: str, time_col: str, value_cols: list[str]) -> pd.DataFrame:
    prepared = prepareFlowForTsfresh(flow_df, entity_col, time_col, value_cols)
    features = extract_features(
        prepared,
        column_id=entity_col,
        column_sort=time_col,
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True,
        n_jobs=1,
    )
    impute(features)
    features.index.name = entity_col
    return features.reset_index()


def sanitizeCategoryToken(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return safe[:40] if safe else "empty"


def expandCategoricalSequenceColumns(
    flow_df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    continuous_cols: list[str],
    categorical_cols: list[str],
    reference_categories: dict[str, list[object]],
) -> tuple[pd.DataFrame, list[str]]:
    work = flow_df[[entity_col, time_col, *continuous_cols, *categorical_cols]].copy()
    for col in continuous_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    expanded_cols = list(continuous_cols)
    for col in categorical_cols:
        cat_series = work[col].astype("string").fillna("<NA>")
        categories = list(reference_categories.get(col, []))
        for cat in categories:
            token = sanitizeCategoryToken(cat)
            new_col = f"{col}_is_{token}"
            work[new_col] = (cat_series == str(cat)).astype(float)
            expanded_cols.append(new_col)
        other_col = f"{col}_is_OTHER"
        work[other_col] = (~cat_series.isin([str(cat) for cat in categories])).astype(float) if categories else 1.0
        expanded_cols.append(other_col)
    work = work[[entity_col, time_col, *expanded_cols]]
    return work, expanded_cols


def extractWideTsfreshRepresentation(
    flow_df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    continuous_cols: list[str],
    categorical_cols: list[str],
    reference_categories: dict[str, list[object]],
) -> pd.DataFrame:
    prepared, value_cols = expandCategoricalSequenceColumns(
        flow_df=flow_df,
        entity_col=entity_col,
        time_col=time_col,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        reference_categories=reference_categories,
    )
    return extractWideTsfresh(prepared, entity_col=entity_col, time_col=time_col, value_cols=value_cols)


def cacheKey(cache_dir: Path, dataset: str, method: str, split_name: str) -> Path:
    return cache_dir / f"{dataset}__{method}__{split_name}.pkl"


def loadOrExtract(cache_dir: Path, dataset: str, method: str, split_name: str, flow_df: pd.DataFrame, entity_col: str, time_col: str, value_cols: list[str]) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cacheKey(cache_dir, dataset, method, split_name)
    if path.exists():
        return pd.read_pickle(path)
    features = extractWideTsfresh(flow_df, entity_col=entity_col, time_col=time_col, value_cols=value_cols)
    features.to_pickle(path)
    return features


def loadOrExtractRepresentation(
    cache_dir: Path,
    dataset: str,
    method: str,
    split_name: str,
    flow_df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    continuous_cols: list[str],
    categorical_cols: list[str],
    reference_categories: dict[str, list[object]],
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cacheKey(cache_dir, dataset, method, split_name)
    if path.exists():
        return pd.read_pickle(path)
    features = extractWideTsfreshRepresentation(
        flow_df=flow_df,
        entity_col=entity_col,
        time_col=time_col,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        reference_categories=reference_categories,
    )
    features.to_pickle(path)
    return features


def buildStaticModelFrames(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_col: str = "target",
    id_col: str = "__ts_id",
    date_col: str = "__ts_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = X_train.reset_index(drop=True).copy()
    test_df = X_test.reset_index(drop=True).copy()
    train_df[id_col] = np.arange(len(train_df), dtype=int)
    test_df[id_col] = np.arange(len(test_df), dtype=int)
    train_df[date_col] = pd.Timestamp("2000-01-01")
    test_df[date_col] = pd.Timestamp("2000-01-01")
    train_df[target_col] = y_train.reset_index(drop=True).to_numpy()
    test_df[target_col] = y_test.reset_index(drop=True).to_numpy()
    return train_df, test_df


def splitRealTrainTest(X: pd.DataFrame, y: pd.Series, seed: int, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if y.nunique(dropna=True) > 1 else None
    return train_test_split(X, y, test_size=test_ratio, random_state=seed, stratify=stratify)


def alphaTokenToInt(token: str) -> int:
    value = 0
    for ch in token.lower():
        if "a" <= ch <= "z":
            value = value * 26 + (ord(ch) - ord("a") + 1)
        else:
            return -1
    return value


def addAirbnbEventIdx(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    raw = work["Unnamed: 0"].astype("string")
    num = pd.to_numeric(raw, errors="coerce")
    alpha_mask = raw.str.fullmatch(r"[A-Za-z]+", na=False)
    if alpha_mask.notna().mean() > 0 and alpha_mask.mean() >= 0.95:
        sort_key = raw.fillna("").str.lower().map(alphaTokenToInt).astype(float)
    elif num.notna().mean() >= 0.95:
        sort_key = num.astype(float)
    else:
        sort_key = pd.Series(np.arange(len(work), dtype=float), index=work.index)
    work["__airbnb_sort_key__"] = sort_key
    work = work.sort_values(["user", "__airbnb_sort_key__"], kind="mergesort").copy()
    work["event_idx"] = work.groupby("user", sort=False).cumcount()
    return work.drop(columns=["__airbnb_sort_key__"])


def encodeValueColumnsWithReference(real_df: pd.DataFrame, syn_df: pd.DataFrame, value_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    real = real_df.copy()
    syn = syn_df.copy()
    for col in value_cols:
        if col not in real.columns or col not in syn.columns:
            continue
        real_num = pd.to_numeric(real[col], errors="coerce")
        syn_num = pd.to_numeric(syn[col], errors="coerce")
        if real_num.notna().mean() > 0.95 and syn_num.notna().mean() > 0.95:
            real[col] = real_num.fillna(-1)
            syn[col] = syn_num.fillna(-1)
            continue
        categories = sorted(real[col].astype("string").fillna("<NA>").unique().tolist())
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        real[col] = real[col].astype("string").fillna("<NA>").map(mapping).fillna(-1).astype(float)
        syn[col] = syn[col].astype("string").fillna("<NA>").map(mapping).fillna(-1).astype(float)
    return real, syn


def rossmannLabelFrame(store_df: pd.DataFrame) -> pd.DataFrame:
    labels = store_df[["Store", "Promo2"]].copy()
    labels["Promo2"] = pd.to_numeric(labels["Promo2"], errors="coerce").fillna(0).astype(int)
    return labels


def cmapssLabelFrame(cmapss_df: pd.DataFrame, threshold: float, source_col: str = "setting_1") -> pd.DataFrame:
    labels = (
        cmapss_df.groupby("unit_nr", as_index=False)[source_col]
        .mean(numeric_only=True)
        .rename(columns={source_col: f"{source_col}_unit_mean"})
    )
    labels["setting_1_high_regime"] = (labels[f"{source_col}_unit_mean"] >= threshold).astype(int)
    return labels[["unit_nr", "setting_1_high_regime"]]


def freddiemacLabelFrame(orig_df: pd.DataFrame) -> pd.DataFrame:
    labels = orig_df[["LOAN SEQUENCE NUMBER", "FIRST TIME HOMEBUYER FLAG"]].copy()
    labels["FIRST TIME HOMEBUYER FLAG"] = pd.to_numeric(labels["FIRST TIME HOMEBUYER FLAG"], errors="coerce").fillna(0).astype(int)
    return labels


def walmartThreshold(stores_df: pd.DataFrame) -> float:
    size = pd.to_numeric(stores_df["Size"], errors="coerce")
    return float(size.median())


def walmartLabelFrame(stores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    labels = stores_df[["Store", "Size"]].copy()
    labels["size_ge_threshold"] = (pd.to_numeric(labels["Size"], errors="coerce") >= threshold).astype(int)
    return labels[["Store", "size_ge_threshold"]]


def aggregateWalmartDepts(depts_df: pd.DataFrame) -> pd.DataFrame:
    work = depts_df.copy()
    work["Weekly_Sales"] = pd.to_numeric(work["Weekly_Sales"], errors="coerce")
    agg = (
        work.groupby(["Store", "Date"], as_index=False)
        .agg(
            dept_weekly_sales_sum=("Weekly_Sales", "sum"),
            dept_weekly_sales_mean=("Weekly_Sales", "mean"),
            dept_weekly_sales_std=("Weekly_Sales", "std"),
            dept_count=("Dept", "count"),
        )
    )
    agg["dept_weekly_sales_std"] = agg["dept_weekly_sales_std"].fillna(0.0)
    return agg


def buildCmapssFlowPair(method: str, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    real_df, syn_df, _ = loadSingleTablePair("cmapss_combined", "cmapss", method, data_root)
    threshold = real_df.groupby("unit_nr", as_index=False)["setting_1"].mean(numeric_only=True)["setting_1"].median()
    label_real = cmapssLabelFrame(real_df, threshold=threshold)
    label_syn = cmapssLabelFrame(syn_df, threshold=threshold)
    return real_df, syn_df, label_real, label_syn, float(threshold)


def buildFreddiemacFlowPair(method: str, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_orig, syn_orig, _ = loadSingleTablePair("freddiemac", "orig", method, data_root)
    real_hist, syn_hist, _ = loadSingleTablePair("freddiemac", "hist", method, data_root)
    label_real = freddiemacLabelFrame(real_orig)
    label_syn = freddiemacLabelFrame(syn_orig)
    return real_hist, syn_hist, label_real, label_syn


def buildRossmannFlowPair(method: str, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_store, syn_store, _ = loadSingleTablePair("rossmann_subsampled", "store", method, data_root)
    real_hist, syn_hist, _ = loadSingleTablePair("rossmann_subsampled", "historical", method, data_root)
    label_real = rossmannLabelFrame(real_store)
    label_syn = rossmannLabelFrame(syn_store)
    return real_hist, syn_hist, label_real, label_syn


def buildWalmartFlowPair(method: str, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    real_stores, syn_stores, _ = loadSingleTablePair("walmart_subsampled", "stores", method, data_root)
    real_features, syn_features, _ = loadSingleTablePair("walmart_subsampled", "features", method, data_root)
    real_depts, syn_depts, _ = loadSingleTablePair("walmart_subsampled", "depts", method, data_root)
    real_flow = real_features.merge(aggregateWalmartDepts(real_depts), on=["Store", "Date"], how="left")
    syn_flow = syn_features.merge(aggregateWalmartDepts(syn_depts), on=["Store", "Date"], how="left")
    for df in (real_flow, syn_flow):
        for col in ["dept_weekly_sales_sum", "dept_weekly_sales_mean", "dept_weekly_sales_std", "dept_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    threshold = walmartThreshold(real_stores)
    label_real = walmartLabelFrame(real_stores, threshold)
    label_syn = walmartLabelFrame(syn_stores, threshold)
    return real_flow, syn_flow, label_real, label_syn, threshold


def getRepresentationSpec(dataset: str, config: dict) -> tuple[str, str, list[str], list[str], dict]:
    spec = config["datasets"].get(dataset)
    if spec is None:
        raise ValueError(f"Unsupported ts representation dataset: {dataset!r}")
    return (
        spec["entity_col"],
        spec["time_col"],
        list(spec["continuous_cols"]),
        list(spec["categorical_cols"]),
        dict(spec.get("metadata", {})),
    )


def buildFlowOnlyPair(method: str, data_root: Path, dataset: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = config["datasets"].get(dataset)
    if spec is None:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    flow_builder = spec["flow_builder"]
    if flow_builder == "cmapss_combined":
        real_df, syn_df, _, _, _ = buildCmapssFlowPair(method, data_root)
        return real_df, syn_df
    if flow_builder == "freddiemac":
        real_hist, syn_hist, _, _ = buildFreddiemacFlowPair(method, data_root)
        return real_hist, syn_hist
    if flow_builder == "rossmann_subsampled":
        real_hist, syn_hist, _, _ = buildRossmannFlowPair(method, data_root)
        return real_hist, syn_hist
    if flow_builder == "walmart_subsampled":
        real_flow, syn_flow, _, _, _ = buildWalmartFlowPair(method, data_root)
        return real_flow, syn_flow
    raise ValueError(f"Unsupported flow_builder={flow_builder!r} for dataset={dataset!r}")


def buildTsRepresentationBundleV2(data_root: Path, dataset: str, methods: list[str], cache_dir: Path, config: dict) -> TsRepresentationBundle:
    entity_col, time_col, continuous_cols, categorical_cols, extra_meta = getRepresentationSpec(dataset, config)
    real_flow, _ = buildFlowOnlyPair(methods[0], data_root, dataset, config)
    reference_categories: dict[str, list[object]] = {}
    for col in categorical_cols:
        values = real_flow[col].astype("string").fillna("<NA>").value_counts().sort_index().index.tolist()
        reference_categories[col] = values
    real_X = loadOrExtractRepresentation(
        cache_dir=cache_dir,
        dataset=dataset,
        method="real",
        split_name="repr_v2",
        flow_df=real_flow,
        entity_col=entity_col,
        time_col=time_col,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        reference_categories=reference_categories,
    )
    syn_by_method: dict[str, pd.DataFrame] = {}
    for method in methods:
        _, syn_flow = buildFlowOnlyPair(method, data_root, dataset, config)
        syn_by_method[method] = loadOrExtractRepresentation(
            cache_dir=cache_dir,
            dataset=dataset,
            method=method,
            split_name="repr_v2",
            flow_df=syn_flow,
            entity_col=entity_col,
            time_col=time_col,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            reference_categories=reference_categories,
        )
    metadata = {
        "time_col": time_col,
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "categorical_handling": "onehot_sequence_plus_other",
        **extra_meta,
    }
    return TsRepresentationBundle(dataset=dataset, entity_col=entity_col, real_X=real_X, syn_by_method=syn_by_method, metadata=metadata)
