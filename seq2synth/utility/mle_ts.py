from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from seq2syn.utils import (
    TsDatasetBundle,
    addAirbnbEventIdx,
    aggregateWalmartDepts,
    buildStaticModelFrames,
    discoverMethodsForDataset,
    encodeValueColumnsWithReference,
    fitDownstreamModel,
    loadLocalConfig,
    loadOrExtract,
    loadSingleTablePair,
    parseCsvArg,
    resolveRuntimePath,
    splitRealTrainTest,
)

MLE_TS_CONFIG = loadLocalConfig("mle_ts.json", __file__)


@dataclass(frozen=True)
class MleTaskSpec:
    dataset: str
    entity_col: str
    time_col: str
    label_col: str
    value_cols: list[str]
    metadata: dict[str, Any]


def filter_available_methods_for_dataset(data_root: Path, dataset: str, methods: list[str]) -> list[str]:
    if dataset != "fanniemae":
        return methods
    filtered = []
    for method in methods:
        child_path = data_root / "synthetic" / dataset / method / "1" / "sample1" / "child.csv"
        parent_path = data_root / "synthetic" / dataset / method / "1" / "sample1" / "parent.csv"
        if child_path.exists() and parent_path.exists():
            filtered.append(method)
    return filtered


def alpha_token_to_int(token: str) -> int:
    value = 0
    for ch in token.lower():
        if "a" <= ch <= "z":
            value = value * 26 + (ord(ch) - ord("a") + 1)
        else:
            return -1
    return value


def add_airbnb_event_idx(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    raw = work["Unnamed: 0"].astype("string")
    num = pd.to_numeric(raw, errors="coerce")
    alpha_mask = raw.str.fullmatch(r"[A-Za-z]+", na=False)
    if alpha_mask.notna().mean() > 0 and alpha_mask.mean() >= 0.95:
        sort_key = raw.fillna("").str.lower().map(alpha_token_to_int).astype(float)
    elif num.notna().mean() >= 0.95:
        sort_key = num.astype(float)
    else:
        sort_key = pd.Series(np.arange(len(work), dtype=float), index=work.index)
    work["__airbnb_sort_key__"] = sort_key
    work = work.sort_values(["user", "__airbnb_sort_key__"], kind="mergesort").copy()
    work["event_idx"] = work.groupby("user", sort=False).cumcount()
    return work.drop(columns=["__airbnb_sort_key__"])


def encode_value_columns_with_reference(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    value_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def cmapss_label_frame(cmapss_df: pd.DataFrame, threshold: float, *, source_col: str, label_col: str) -> pd.DataFrame:
    labels = (
        cmapss_df.sort_values(["unit_nr", "time_cycles"], kind="mergesort")
        .groupby("unit_nr", as_index=False)
        .tail(1)[["unit_nr", source_col]]
        .rename(columns={source_col: f"{source_col}_final"})
    )
    labels[label_col] = (labels[f"{source_col}_final"] >= threshold).astype(int)
    return labels[["unit_nr", label_col]]


def aggregate_walmart_depts(depts_df: pd.DataFrame) -> pd.DataFrame:
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


def get_task_spec(dataset: str) -> MleTaskSpec:
    spec = MLE_TS_CONFIG["datasets"].get(dataset)
    if spec is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return MleTaskSpec(
        dataset=dataset,
        entity_col=spec["entity_col"],
        time_col=spec["time_col"],
        label_col=spec["label_col"],
        value_cols=list(spec["value_cols"]),
        metadata=dict(spec["metadata"]),
    )


def build_flow_pair(
    dataset: str,
    method: str,
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = get_task_spec(dataset)
    flow_builder = MLE_TS_CONFIG["datasets"][dataset]["flow_builder"]

    if flow_builder == "cmapss":
        real_df, syn_df, _ = loadSingleTablePair("cmapss", "cmapss", method, data_root)
        threshold = (
            real_df.sort_values(["unit_nr", "time_cycles"], kind="mergesort")
            .groupby("unit_nr", as_index=False)
            .tail(1)["s_11"]
            .median()
        )
        label_real = cmapss_label_frame(real_df, threshold=threshold, source_col="s_11", label_col="final_s11_high_regime")
        label_syn = cmapss_label_frame(syn_df, threshold=threshold, source_col="s_11", label_col="final_s11_high_regime")
        meta = {"label_threshold": float(threshold)}
        return real_df, syn_df, label_real, label_syn, meta

    if flow_builder == "freddiemac":
        real_orig, syn_orig, _ = loadSingleTablePair("freddiemac", "orig", method, data_root)
        real_hist, syn_hist, _ = loadSingleTablePair("freddiemac", "hist", method, data_root)
        label_real = real_orig[["LOAN SEQUENCE NUMBER", "PROPERTY TYPE"]].copy()
        label_syn = syn_orig[["LOAN SEQUENCE NUMBER", "PROPERTY TYPE"]].copy()
        for labels in (label_real, label_syn):
            labels["property_type_is_4"] = (labels["PROPERTY TYPE"].astype("string") == "4").astype(int)
            labels.drop(columns=["PROPERTY TYPE"], inplace=True)
        return real_hist, syn_hist, label_real, label_syn, {}

    if flow_builder == "rossmann_subsampled":
        real_store, syn_store, _ = loadSingleTablePair("rossmann_subsampled", "store", method, data_root)
        real_hist, syn_hist, _ = loadSingleTablePair("rossmann_subsampled", "historical", method, data_root)
        label_real = real_store[["Store", "Promo2"]].copy()
        label_syn = syn_store[["Store", "Promo2"]].copy()
        for labels in (label_real, label_syn):
            labels["Promo2"] = pd.to_numeric(labels["Promo2"], errors="coerce").fillna(0).astype(int)
        return real_hist, syn_hist, label_real, label_syn, {}

    if flow_builder == "walmart_subsampled":
        real_stores = pd.read_csv(data_root / "original" / "walmart_subsampled" / "stores.csv", usecols=["Store", "Type"])
        syn_stores = pd.read_csv(data_root / "synthetic" / "walmart_subsampled" / method / "1" / "sample1" / "stores.csv", usecols=["Store", "Type"])
        real_features, syn_features, _ = loadSingleTablePair("walmart_subsampled", "features", method, data_root)
        real_depts, syn_depts, _ = loadSingleTablePair("walmart_subsampled", "depts", method, data_root)
        real_flow = real_features.merge(aggregate_walmart_depts(real_depts), on=["Store", "Date"], how="left")
        syn_flow = syn_features.merge(aggregate_walmart_depts(syn_depts), on=["Store", "Date"], how="left")
        for df in (real_flow, syn_flow):
            for col in ["dept_weekly_sales_sum", "dept_weekly_sales_mean", "dept_weekly_sales_std", "dept_count"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        label_real = real_stores[["Store", "Type"]].copy()
        label_syn = syn_stores[["Store", "Type"]].copy()
        for labels in (label_real, label_syn):
            labels["store_type_is_A"] = (labels["Type"].astype("string") == "A").astype(int)
            labels.drop(columns=["Type"], inplace=True)
        return real_flow, syn_flow, label_real, label_syn, {}

    if flow_builder == "fanniemae_subsampled":
        real_parent, syn_parent, _ = loadSingleTablePair("fanniemae_subsampled", "parent", method, data_root)
        real_child, syn_child, _ = loadSingleTablePair("fanniemae_subsampled", "child", method, data_root)
        label_real = real_parent[["Loan Identifier", "Property Type"]].copy()
        label_syn = syn_parent[["Loan Identifier", "Property Type"]].copy()
        for labels in (label_real, label_syn):
            labels["property_type_is_4"] = (labels["Property Type"].astype("string") == "4").astype(int)
            labels.drop(columns=["Property Type"], inplace=True)
        return real_child, syn_child, label_real, label_syn, {}

    if flow_builder == "fanniemae":
        real_parent, syn_parent, _ = loadSingleTablePair("fanniemae", "parent", method, data_root)
        real_child, syn_child, _ = loadSingleTablePair("fanniemae", "child", method, data_root)
        label_real = real_parent[["Loan Identifier", "Property Type"]].copy()
        label_syn = syn_parent[["Loan Identifier", "Property Type"]].copy()
        for labels in (label_real, label_syn):
            labels["property_type_is_4"] = (labels["Property Type"].astype("string") == "4").astype(int)
            labels.drop(columns=["Property Type"], inplace=True)
        return real_child, syn_child, label_real, label_syn, {}

    if flow_builder == "hnm_subsampled":
        real_tx, syn_tx, _ = loadSingleTablePair("hnm_subsampled", "transactions_train", method, data_root)
        real_customers = pd.read_csv(data_root / "original" / "hnm_subsampled" / "customers.csv", usecols=["customer_id", "fashion_news_frequency"])
        syn_customers = pd.read_csv(data_root / "synthetic" / "hnm_subsampled" / method / "1" / "sample1" / "customers.csv", usecols=["customer_id", "fashion_news_frequency"])
        label_real = real_customers.copy()
        label_syn = syn_customers.copy()
        for labels in (label_real, label_syn):
            labels["fashion_news_regularly"] = labels["fashion_news_frequency"].astype("string").eq("Regularly").fillna(False).astype(int)
            labels.drop(columns=["fashion_news_frequency"], inplace=True)
        return real_tx, syn_tx, label_real, label_syn, {}

    if flow_builder == "ptbxl_subsampled":
        real_meta = pd.read_csv(data_root / "original" / "ptbxl_subsampled" / "ptbxl_database.csv", usecols=["ecg_id", "sex"])
        syn_meta = pd.read_csv(data_root / "synthetic" / "ptbxl_subsampled" / method / "1" / "sample1" / "ptbxl_database.csv", usecols=["ecg_id", "sex"])
        record_cols = ["ecg_id", "timestamp"] + [f"lead_{idx}" for idx in range(12)]
        real_records = pd.read_csv(data_root / "original" / "ptbxl_subsampled" / "records.csv", usecols=record_cols)
        syn_records = pd.read_csv(data_root / "synthetic" / "ptbxl_subsampled" / method / "1" / "sample1" / "records.csv", usecols=record_cols)
        real_records["timestamp"] = pd.to_datetime(real_records["timestamp"], errors="coerce")
        syn_records["timestamp"] = pd.to_datetime(syn_records["timestamp"], errors="coerce")
        label_real = real_meta.copy()
        label_syn = syn_meta.copy()
        for labels in (label_real, label_syn):
            labels["sex"] = pd.to_numeric(labels["sex"], errors="coerce").fillna(0).astype(int)
        return real_records, syn_records, label_real, label_syn, {}

    if flow_builder == "berka":
        real_parent = pd.read_csv(data_root / "original" / "berka" / "parent.csv")
        syn_parent = pd.read_csv(data_root / "synthetic" / "berka" / method / "1" / "sample1" / "parent.csv")
        real_child = pd.read_csv(data_root / "original" / "berka" / "child.csv")
        syn_child = pd.read_csv(data_root / "synthetic" / "berka" / method / "1" / "sample1" / "child.csv")
        for df in (real_child, syn_child):
            year = pd.to_numeric(df["Year"], errors="coerce")
            year = np.where(year < 100, year + 1900, year)
            df["__date__"] = pd.to_datetime({"year": year, "month": pd.to_numeric(df["Month"], errors="coerce"), "day": pd.to_numeric(df["Day"], errors="coerce")}, errors="coerce")
        real_child, syn_child = encode_value_columns_with_reference(real_child, syn_child, get_task_spec("berka").value_cols)
        label_real = real_parent[["user", "region"]].copy()
        label_syn = syn_parent[["user", "region"]].copy()
        for labels in (label_real, label_syn):
            labels["region_is_moravia"] = labels["region"].astype("string").isin(["south Moravia", "north Moravia"]).astype(int)
            labels.drop(columns=["region"], inplace=True)
        return real_child, syn_child, label_real, label_syn, {}

    if flow_builder == "airbnb":
        real_parent = pd.read_csv(data_root / "original" / "airbnb" / "parent.csv")
        syn_parent = pd.read_csv(data_root / "synthetic" / "airbnb" / method / "1" / "sample1" / "parent.csv")
        real_child = pd.read_csv(data_root / "original" / "airbnb" / "child.csv")
        syn_child = pd.read_csv(data_root / "synthetic" / "airbnb" / method / "1" / "sample1" / "child.csv")
        real_child = addAirbnbEventIdx(real_child)
        syn_child = addAirbnbEventIdx(syn_child)
        real_child, syn_child = encodeValueColumnsWithReference(real_child, syn_child, get_task_spec("airbnb").value_cols)
        label_real = real_parent[["user", "n_sessions"]].copy()
        label_syn = syn_parent[["user", "n_sessions"]].copy()
        for labels in (label_real, label_syn):
            labels["n_sessions_ge_17"] = pd.to_numeric(labels["n_sessions"], errors="coerce").fillna(-1).ge(17).astype(int)
            labels.drop(columns=["n_sessions"], inplace=True)
        return real_child, syn_child, label_real, label_syn, {}

    if flow_builder == "citi_bike":
        cols = [
            "bikeid",
            "starttime",
            "tripduration",
            "start station latitude",
            "start station longitude",
            "end station latitude",
            "end station longitude",
            "birth year",
            "usertype",
            "gender",
        ]
        real_df = pd.read_csv(data_root / "original" / "citi_bike" / "citi_bike.csv", usecols=cols)
        syn_df = pd.read_csv(data_root / "synthetic" / "citi_bike" / method / "1" / "sample1" / "citi_bike.csv", usecols=cols)
        for df in (real_df, syn_df):
            df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
        real_df, syn_df = encodeValueColumnsWithReference(real_df, syn_df, get_task_spec("citi_bike").value_cols)
        label_real = real_df.groupby("bikeid")["gender"].agg(lambda s: int(s.astype("string").mode(dropna=False).iloc[0] == "1")).reset_index(name="mode_gender_is_1")
        label_syn = syn_df.groupby("bikeid")["gender"].agg(lambda s: int(s.astype("string").mode(dropna=False).iloc[0] == "1")).reset_index(name="mode_gender_is_1")
        return real_df, syn_df, label_real, label_syn, {}

    if flow_builder == "rossmann_tabdit":
        real_store = pd.read_csv(data_root / "original" / "rossmann_tabdit" / "store.csv")
        syn_store = pd.read_csv(data_root / "synthetic" / "rossmann_tabdit" / method / "1" / "sample1" / "store.csv")
        real_hist = pd.read_csv(data_root / "original" / "rossmann_tabdit" / "historical.csv")
        syn_hist = pd.read_csv(data_root / "synthetic" / "rossmann_tabdit" / method / "1" / "sample1" / "historical.csv")
        for df in (real_hist, syn_hist):
            df["__date__"] = pd.to_datetime(
                {"year": 2014, "month": pd.to_numeric(df["Date_month"], errors="coerce"), "day": pd.to_numeric(df["Date_day"], errors="coerce")},
                errors="coerce",
            )
        real_hist, syn_hist = encodeValueColumnsWithReference(real_hist, syn_hist, get_task_spec("rossmann_tabdit").value_cols)
        label_real = real_store[["user", "Promo2"]].copy()
        label_syn = syn_store[["user", "Promo2"]].copy()
        for labels in (label_real, label_syn):
            labels["Promo2"] = pd.to_numeric(labels["Promo2"], errors="coerce").fillna(0).astype(int)
        return real_hist, syn_hist, label_real, label_syn, {}

    raise ValueError(f"Unsupported flow_builder={flow_builder!r} for dataset={dataset}")


def build_ts_dataset_bundle_final(data_root: Path, dataset: str, methods: list[str], cache_dir: Path) -> TsDatasetBundle:
    spec = get_task_spec(dataset)
    real_flow, _, real_labels, _, extra_meta = build_flow_pair(dataset, methods[0], data_root)
    real_features = loadOrExtract(cache_dir, dataset, "real", "flow", real_flow, spec.entity_col, spec.time_col, spec.value_cols)
    real_joined = real_features.merge(real_labels, on=spec.entity_col, how="inner").dropna(subset=[spec.label_col]).copy()
    real_X = real_joined.drop(columns=[spec.label_col])
    real_y = real_joined[spec.label_col].astype(int)

    syn_by_method: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for method in methods:
        _, syn_flow, _, syn_labels, _ = build_flow_pair(dataset, method, data_root)
        syn_features = loadOrExtract(cache_dir, dataset, method, "flow", syn_flow, spec.entity_col, spec.time_col, spec.value_cols)
        joined = syn_features.merge(syn_labels, on=spec.entity_col, how="inner").dropna(subset=[spec.label_col]).copy()
        syn_by_method[method] = (joined.drop(columns=[spec.label_col]), joined[spec.label_col].astype(int))

    metadata = {
        "time_col": spec.time_col,
        "flow_source": spec.metadata["flow_source"],
        "label_source_col": spec.metadata["label_source_col"],
        "label_source_type": spec.metadata["label_source_type"],
        "positive_count": int(real_y.sum()),
        "negative_count": int((1 - real_y).sum()),
        "value_cols": spec.value_cols,
        **extra_meta,
    }

    return TsDatasetBundle(
        dataset=dataset,
        entity_col=spec.entity_col,
        label_col=spec.label_col,
        task="classification",
        real_X=real_X,
        real_y=real_y,
        syn_by_method=syn_by_method,
        metadata=metadata,
    )


def split_synthetic_train_only(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.Series]:
    can_stratify = y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2
    stratify = y if can_stratify else None
    try:
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=test_ratio,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=test_ratio,
            random_state=seed,
            stratify=None,
        )
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True)


def evaluate_generated_only_run(
    bundle: TsDatasetBundle,
    method: str,
    seed: int,
    step: int,
    test_ratio: float,
) -> dict:
    _, real_test_X, _, real_test_y = splitRealTrainTest(
        bundle.real_X,
        bundle.real_y,
        seed=seed,
        test_ratio=test_ratio,
    )
    syn_X, syn_y = bundle.syn_by_method[method]
    syn_train_X, syn_train_y = split_synthetic_train_only(
        syn_X,
        syn_y,
        seed=seed,
        test_ratio=test_ratio,
    )

    selected_train_X = syn_train_X.drop(columns=[bundle.entity_col], errors="ignore").copy().fillna(0)
    selected_test_X = real_test_X.drop(columns=[bundle.entity_col], errors="ignore").copy().fillna(0)
    keep_cols = [col for col in selected_train_X.columns if selected_train_X[col].nunique(dropna=False) > 1]
    if keep_cols:
        selected_train_X = selected_train_X[keep_cols].copy()
        selected_test_X = selected_test_X.reindex(columns=keep_cols, fill_value=0).copy()
    else:
        selected_train_X = selected_train_X.iloc[:, 0:0].copy()
        selected_test_X = selected_test_X.iloc[:, 0:0].copy()
    selected_cols = list(selected_train_X.columns)

    train_df, test_df = buildStaticModelFrames(
        selected_train_X,
        syn_train_y,
        selected_test_X,
        real_test_y,
    )

    metrics = fitDownstreamModel(
        train=train_df,
        valid=train_df.iloc[0:0].copy(),
        test=test_df,
        tune_train=train_df,
        target_col="target",
        date_col="__ts_date",
        id_col="__ts_id",
        task="classification",
        seed=seed,
        step=step,
        tune_hyperparameters=False,
    )

    return {
        "dataset": bundle.dataset,
        "target_col": bundle.label_col,
        "method": method,
        "seed": int(seed),
        "train_source": "synthetic_only",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "selected_feature_count": int(len(selected_cols)),
        **metrics,
    }


def evaluate_real_only_run(
    bundle: TsDatasetBundle,
    seed: int,
    step: int,
    test_ratio: float,
) -> dict:
    real_train_X, real_test_X, real_train_y, real_test_y = splitRealTrainTest(
        bundle.real_X,
        bundle.real_y,
        seed=seed,
        test_ratio=test_ratio,
    )

    selected_train_X = real_train_X.drop(columns=[bundle.entity_col], errors="ignore").copy().fillna(0)
    selected_test_X = real_test_X.drop(columns=[bundle.entity_col], errors="ignore").copy().fillna(0)
    keep_cols = [col for col in selected_train_X.columns if selected_train_X[col].nunique(dropna=False) > 1]
    if keep_cols:
        selected_train_X = selected_train_X[keep_cols].copy()
        selected_test_X = selected_test_X.reindex(columns=keep_cols, fill_value=0).copy()
    else:
        selected_train_X = selected_train_X.iloc[:, 0:0].copy()
        selected_test_X = selected_test_X.iloc[:, 0:0].copy()
    selected_cols = list(selected_train_X.columns)

    train_df, test_df = buildStaticModelFrames(
        selected_train_X,
        real_train_y,
        selected_test_X,
        real_test_y,
    )

    metrics = fitDownstreamModel(
        train=train_df,
        valid=train_df.iloc[0:0].copy(),
        test=test_df,
        tune_train=train_df,
        target_col="target",
        date_col="__ts_date",
        id_col="__ts_id",
        task="classification",
        seed=seed,
        step=step,
        tune_hyperparameters=False,
    )

    return {
        "dataset": bundle.dataset,
        "target_col": bundle.label_col,
        "method": "REAL_ONLY",
        "seed": int(seed),
        "train_source": "real_only",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "selected_feature_count": int(len(selected_cols)),
        **metrics,
    }


def build_summary(seed_results: pd.DataFrame) -> pd.DataFrame:
    if seed_results.empty:
        return pd.DataFrame()
    return (
        seed_results.groupby(["dataset", "target_col", "method", "train_source"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            train_rows_mean=("train_rows", "mean"),
            test_rows_mean=("test_rows", "mean"),
            selected_feature_count_mean=("selected_feature_count", "mean"),
            selected_feature_count_std=("selected_feature_count", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
        )
        .sort_values(["dataset", "method"])
        .reset_index(drop=True)
    )


def build_metadata(bundles: list[TsDatasetBundle]) -> list[dict]:
    out = []
    for bundle in bundles:
        out.append({"dataset": bundle.dataset, "label_col": bundle.label_col, "task": bundle.task, **bundle.metadata})
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run final tsfresh-based MLE over datas_revised flow tables.")
    parser.add_argument("--data-root", type=Path, default=Path("./datas_revised"))
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/seq2syn/mle_ts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./outputs/seq2syn/ts_feature_cache_mle"))
    parser.add_argument(
        "--include-real-only",
        action="store_true",
        help="Also evaluate REAL_ONLY train/test baseline alongside synthetic_only TSTR runs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = resolveRuntimePath(args.data_root, __file__)
    args.output_dir = resolveRuntimePath(args.output_dir, __file__)
    args.cache_dir = resolveRuntimePath(args.cache_dir, __file__)
    supported = list(MLE_TS_CONFIG["supported_datasets"])
    datasets = [d for d in (parseCsvArg(args.datasets) if args.datasets else supported) if d in supported]
    seeds = [int(x) for x in parseCsvArg(args.seeds)] if args.seeds else [1, 2, 3]
    method_filter = set(parseCsvArg(args.methods)) if args.methods else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundles: list[TsDatasetBundle] = []
    rows: list[dict] = []

    for dataset in datasets:
        methods = discoverMethodsForDataset(args.data_root, dataset)
        if method_filter is not None:
            methods = [method for method in methods if method in method_filter]
        methods = filter_available_methods_for_dataset(args.data_root, dataset, methods)
        if not methods:
            continue
        bundle = build_ts_dataset_bundle_final(args.data_root, dataset=dataset, methods=methods, cache_dir=args.cache_dir)
        bundles.append(bundle)

        if args.include_real_only:
            for seed in seeds:
                print(f"Running MLE-TS-final real-only dataset={dataset} seed={seed}")
                rows.append(
                    evaluate_real_only_run(
                        bundle=bundle,
                        seed=seed,
                        step=args.step,
                        test_ratio=args.test_ratio,
                    )
                )

        for method in methods:
            for seed in seeds:
                print(f"Running MLE-TS-final generated-only dataset={dataset} method={method} seed={seed}")
                rows.append(
                    evaluate_generated_only_run(
                        bundle=bundle,
                        method=method,
                        seed=seed,
                        step=args.step,
                        test_ratio=args.test_ratio,
                    )
                )

    seed_results = pd.DataFrame(rows)
    summary = build_summary(seed_results)

    for stale in ["ratio_summary.csv", "best_ratio_summary.csv"]:
        stale_path = args.output_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    seed_results.to_csv(args.output_dir / "all_results.csv", index=False)
    seed_results.to_csv(args.output_dir / "seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps(build_metadata(bundles), indent=2), encoding="utf-8")

    print(f"Saved: {(args.output_dir / 'all_results.csv').resolve()}")
    print(f"Saved: {(args.output_dir / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()

