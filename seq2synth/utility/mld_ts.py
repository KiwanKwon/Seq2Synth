from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __package__ in (None, ""):
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from seq2syn.utils import (
    TsRepresentationBundle,
    buildStaticModelFrames,
    buildTsRepresentationBundleV2,
    discoverDatasets,
    discoverMethodsForDataset,
    fitDownstreamModel,
    loadLocalConfig,
    parseCsvArg,
    resolveRuntimePath,
)


MLD_TS_V2_CONFIG = loadLocalConfig("mld_ts_v2.json", __file__)


def prepare_fixed_feature_frames(
    bundle: TsRepresentationBundle,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = X_train.copy().reset_index(drop=True)
    test = X_test.copy().reset_index(drop=True)
    train_features = train.drop(columns=[bundle.entity_col], errors="ignore")
    test_features = test.drop(columns=[bundle.entity_col], errors="ignore")

    train_features = train_features.fillna(0)
    test_features = test_features.fillna(0)

    keep_cols = [col for col in train_features.columns if train_features[col].nunique(dropna=False) > 1]
    if not keep_cols:
        return train_features.iloc[:, 0:0].copy(), test_features.iloc[:, 0:0].copy(), []
    return train_features[keep_cols].copy(), test_features[keep_cols].copy(), keep_cols


def evaluate_mld_ts_v2_seed(bundle: TsRepresentationBundle, method: str, seed: int, n_splits: int = 5) -> dict:
    real_X = bundle.real_X.copy().reset_index(drop=True)
    syn_X = bundle.syn_by_method[method].copy().reset_index(drop=True)

    X = pd.concat([real_X, syn_X], ignore_index=True)
    y = pd.Series([0] * len(real_X) + [1] * len(syn_X), dtype=int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        fixed_train_X, fixed_test_X, keep_cols = prepare_fixed_feature_frames(
            bundle=bundle,
            X_train=X_train,
            X_test=X_test,
        )
        train_df, test_df = buildStaticModelFrames(
            fixed_train_X,
            y_train,
            fixed_test_X,
            y_test,
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
            step=50,
            tune_hyperparameters=False,
        )
        fold_rows.append(
            {
                "fixed_feature_count": len(keep_cols),
                "test_acc": metrics.get("accuracy"),
                "test_auc": metrics.get("auc"),
                "test_f1": metrics.get("f1"),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    return {
        "dataset": bundle.dataset,
        "target_col": "is_synthetic",
        "method": method,
        "seed": int(seed),
        "test_acc": float(fold_df["test_acc"].mean()),
        "test_auc": float(fold_df["test_auc"].mean()),
        "test_f1": float(fold_df["test_f1"].mean()),
        "fixed_feature_count_mean": float(fold_df["fixed_feature_count"].mean()),
    }


def build_summary(seed_results: pd.DataFrame) -> pd.DataFrame:
    if seed_results.empty:
        return pd.DataFrame()
    return (
        seed_results.groupby(["dataset", "target_col", "method"], as_index=False)
        .agg(
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            fixed_feature_count_mean=("fixed_feature_count_mean", "mean"),
            fixed_feature_count_std=("fixed_feature_count_mean", "std"),
        )
        .sort_values(["dataset", "method"])
        .reset_index(drop=True)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run tsfresh-based MLD v2 with fixed feature set and explicit categorical handling.")
    parser.add_argument("--data-root", type=Path, default=Path("./datas_revised"))
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/seq2syn/mld_ts_v2"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./outputs/seq2syn/ts_feature_cache_v2"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = resolveRuntimePath(args.data_root, __file__)
    args.output_dir = resolveRuntimePath(args.output_dir, __file__)
    args.cache_dir = resolveRuntimePath(args.cache_dir, __file__)
    datasets = parseCsvArg(args.datasets) if args.datasets else [d for d in discoverDatasets(args.data_root) if d in MLD_TS_V2_CONFIG["supported_datasets"]]
    seeds = [int(x) for x in parseCsvArg(args.seeds)] if args.seeds else [1, 2, 3]
    method_filter = set(parseCsvArg(args.methods)) if args.methods else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    metadata_rows = []
    skipped_rows = []

    for dataset in datasets:
        methods = discoverMethodsForDataset(args.data_root, dataset)
        if method_filter is not None:
            methods = [method for method in methods if method in method_filter]
        if not methods:
            continue

        bundle = buildTsRepresentationBundleV2(args.data_root, dataset=dataset, methods=methods, cache_dir=args.cache_dir, config=MLD_TS_V2_CONFIG)
        metadata_rows.append({"dataset": dataset, **bundle.metadata, "mld_label_col": "is_synthetic"})
        for method in methods:
            syn_X = bundle.syn_by_method[method]
            if len(syn_X) == 0:
                skipped_rows.append({"dataset": dataset, "method": method, "reason": "empty_synthetic_ts_bundle"})
                print(f"Skipping MLD-TS-v2 dataset={dataset} method={method}: empty synthetic TS bundle")
                continue
            for seed in seeds:
                print(f"Running MLD-TS-v2 dataset={dataset} method={method} seed={seed}")
                seed_rows.append(evaluate_mld_ts_v2_seed(bundle=bundle, method=method, seed=seed))

    seed_results = pd.DataFrame(seed_rows)
    summary = build_summary(seed_results)
    seed_results.to_csv(args.output_dir / "seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(args.output_dir / "skipped_methods.csv", index=False)

    print(f"Saved: {(args.output_dir / 'seed_results.csv').resolve()}")
    print(f"Saved: {(args.output_dir / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()

