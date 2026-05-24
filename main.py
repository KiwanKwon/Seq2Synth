"""Seq2Synth benchmark entry point.

This replaces the shell launchers with one Python CLI while keeping the
existing metric implementations callable during the refactor.

run example:
$ python3 main.py             \
    --metrics timestamp       \
    --datasets airbnb_tabdit  \
    --models CLAVADDPM        \
    --variants default        \
    --parallel                \
    --workers 4

"""


from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from configs import available_models, load_dataset_config, resolve_datasets
from paths import RESULTS_DIR, ROOT_DIR
from seq2synth.metrics.longitudinal.longitudinal_metrics import (
    DEFAULT_TT_WASSERSTEIN_MAX_FLOWS,
    DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
    DEFAULT_TT_WASSERSTEIN_SAMPLE_SEED,
)


METRIC_ALIASES = {
    "all": ["structural", "sd", "timestamp", "cross_sectional", "longitudinal", "privacy"],
    "cs": ["cross_sectional"],
    "cross-sectional": ["cross_sectional"],
    "cross_sectional": ["cross_sectional"],
    "long": ["longitudinal"],
    "longitudinal": ["longitudinal"],
    "privacy": ["privacy"],
}


@dataclass(frozen=True)
class Seq2SynthRun:
    dataset: str
    model: str | None
    metric: str
    command: list[str]


class Seq2Synth:
    """Python orchestrator for the Seq2Synth benchmark metrics."""

    def __init__(
        self,
        datasets: list[str] | None = None,
        models: list[str] | None = None,
        metrics: list[str] | None = None,
        variants: list[str] | None = None,
        parallel: bool = False,
        workers: int | None = None,
        dry_run: bool = False,
        longitudinal_tt_standalone: bool = False,
        longitudinal_tt_num_iter_max: int = DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
        longitudinal_tt_max_flows: int | None = DEFAULT_TT_WASSERSTEIN_MAX_FLOWS,
        longitudinal_tt_sample_seed: int = DEFAULT_TT_WASSERSTEIN_SAMPLE_SEED,
    ) -> None:
        self.datasets = resolve_datasets(datasets)
        self.models = models
        self.metrics = self._normalize_metrics(metrics or ["all"])
        self.variants = variants
        self.parallel = parallel
        self.workers = workers
        self.dry_run = dry_run
        self.longitudinal_tt_standalone = longitudinal_tt_standalone
        self.longitudinal_tt_num_iter_max = longitudinal_tt_num_iter_max
        self.longitudinal_tt_max_flows = longitudinal_tt_max_flows
        self.longitudinal_tt_sample_seed = longitudinal_tt_sample_seed

    @staticmethod
    def _normalize_metrics(metrics: list[str]) -> list[str]:
        normalized: list[str] = []
        for metric in metrics:
            for resolved in METRIC_ALIASES.get(metric, [metric]):
                if resolved not in normalized:
                    normalized.append(resolved)
        return normalized

    def build_runs(self) -> list[Seq2SynthRun]:
        runs: list[Seq2SynthRun] = []
        for metric in self.metrics:
            if metric == "timestamp":
                runs.extend(self._timestamp_runs())
            elif metric == "structural":
                runs.extend(self._structural_runs())
            elif metric == "sd":
                runs.extend(self._sd_runs())
            elif metric == "cross_sectional":
                runs.extend(self._cross_sectional_runs())
            elif metric == "longitudinal":
                runs.extend(self._longitudinal_runs())
            elif metric == "privacy":
                runs.extend(self._privacy_runs())
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return runs

    def run(self) -> int:
        exit_code = 0
        runs = self.build_runs()
        for run in runs:
            printable = " ".join(run.command)
            if self.dry_run:
                print(printable)
                continue
            print(f"\n[Seq2Synth] {run.metric}: {run.dataset}"
                  f"{('/' + run.model) if run.model else ''}")
            if run.metric == "sd":
                exit_code = exit_code or self._run_sd_in_process(run.command)
            else:
                result = subprocess.run(run.command, cwd=ROOT_DIR)
                exit_code = exit_code or result.returncode
        return exit_code

    @staticmethod
    def _run_sd_in_process(command: list[str]) -> int:
        from seq2synth.sdmetric import sdmetric

        args = sdmetric.parse_args(command[2:])
        return sdmetric.run_from_args(args)

    def _selected_models(self, dataset: str) -> list[str]:
        if not self.models or "all" in self.models:
            return available_models(dataset)
        return self.models

    @staticmethod
    def _dict_keys_from_file(path: Path, variable: str) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            value = None
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == variable for t in node.targets
            ):
                value = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == variable
            ):
                value = node.value
            if isinstance(value, ast.Dict):
                return {
                    key.value
                    for key in value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
        return set()

    @staticmethod
    def _timestamp_datasets_from_file(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        datasets: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Name) or func.id != "DatasetConfig":
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    datasets.add(keyword.value.value)
        return datasets

    @staticmethod
    def _sd_datasets_from_file(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        datasets = Seq2Synth._dict_keys_from_file(path, "_SYNTH_TABLE")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            left = node.left
            if not (
                isinstance(left, ast.Attribute)
                and left.attr == "dataset"
                and isinstance(left.value, ast.Name)
                and left.value.id == "args"
            ):
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    datasets.add(comparator.value)
        return datasets

    @staticmethod
    def _filter_supported(metric: str, datasets: list[str], supported: set[str]) -> list[str]:
        selected = [dataset for dataset in datasets if dataset in supported]
        skipped = sorted(set(datasets) - supported)
        if skipped:
            print(f"[Seq2Synth][skip] {metric} does not support: {skipped}")
        return selected

    @staticmethod
    def _filter_cross_sectional_supported(datasets: list[str]) -> list[str]:
        selected: list[str] = []
        skipped: list[str] = []
        for dataset in datasets:
            cfg = load_dataset_config(dataset)
            if cfg.supports_cross_sectional:
                selected.append(dataset)
            else:
                skipped.append(dataset)
        if skipped:
            print(
                "[Seq2Synth][skip] cross_sectional requires "
                "time_representation=absolute and trajectory_independence=dependent: "
                f"{sorted(skipped)}"
            )
        return selected

    def _variant_args(self) -> list[str]:
        return ["--variants", *self.variants] if self.variants else []

    def _timestamp_runs(self) -> list[Seq2SynthRun]:
        _FIDELITY_RUNNER = (
            ROOT_DIR / "seq2synth" / "metrics" / "timestamp" / "run_timestamp_fidelity.py"
        )
        datasets = self._filter_supported(
            "timestamp",
            self.datasets,
            set(resolve_datasets(["all"])),
        )
        if not datasets:
            return []
        cmd = [
            sys.executable,
            str(_FIDELITY_RUNNER),
            "--datasets",
            *datasets,
        ]
        if self.models and "all" not in self.models:
            cmd.extend(["--models", *self.models])
        if self.parallel:
            cmd.append("--parallel")
        if self.workers is not None:
            cmd.extend(["--workers", str(self.workers)])
            cmd.extend(["--n_jobs", str(self.workers)])
        if self.dry_run:
            cmd.append("--dry_run")
        return [Seq2SynthRun("all", None, "timestamp", cmd)]

    def _structural_runs(self) -> list[Seq2SynthRun]:
        runs: list[Seq2SynthRun] = []
        for dataset in self._filter_supported(
            "structural",
            self.datasets,
            set(resolve_datasets(["all"])),
        ):
            cmd = [
                sys.executable,
                str(Path("seq2synth") / "metrics" / "structural" / "run_structural.py"),
                "--dataset",
                dataset,
                "--model",
                *(self.models or ["all"]),
                "--out_root",
                str(Path("results")),
                *self._variant_args(),
            ]
            if self.parallel:
                cmd.append("--parallel")
            if self.workers is not None:
                cmd.extend(["--workers", str(self.workers)])
                cmd.extend(["--n_jobs", str(self.workers)])
            runs.append(Seq2SynthRun(dataset, None, "structural", cmd))
        return runs

    def _sd_runs(self) -> list[Seq2SynthRun]:
        runs: list[Seq2SynthRun] = []
        for dataset in self._filter_supported(
            "sd",
            self.datasets,
            set(resolve_datasets(["all"])),
        ):
            cfg = load_dataset_config(dataset)
            for model in self._selected_models(dataset):
                synth_base_dir = cfg.synth_dir / model
                if not self.dry_run and not synth_base_dir.exists():
                    print(f"[Seq2Synth][skip] missing model dir: {synth_base_dir}")
                    continue
                cmd = [
                    sys.executable,
                    str(ROOT_DIR / "seq2synth" / "sdmetric" / "sdmetric.py"),
                    "--dataset",
                    dataset,
                    "--real_dir",
                    str(cfg.real_dir),
                    "--synth_base_dir",
                    str(synth_base_dir),
                    "--metadata",
                    str(cfg.metadata_path),
                    "--output_dir",
                    str(cfg.results_dir / model / "sd"),
                    *self._variant_args(),
                ]
                if self.parallel:
                    cmd.append("--parallel")
                if self.workers is not None:
                    cmd.extend(["--workers", str(self.workers)])
                    cmd.extend(["--n_jobs", str(self.workers)])
                runs.append(Seq2SynthRun(dataset, model, "sd", cmd))
        return runs

    def _privacy_runs(self) -> list[Seq2SynthRun]:
        runs: list[Seq2SynthRun] = []
        for dataset in self._filter_supported(
            "privacy",
            self.datasets,
            set(resolve_datasets(["all"])),
        ):
            models = self._selected_models(dataset)
            if not models:
                continue
            cmd = [
                sys.executable,
                str(Path("seq2synth") / "metrics" / "privacy" / "run_privacy.py"),
                "--dataset",
                dataset,
                "--model",
                *models,
                "--results_base",
                str(Path("results")),
                *self._variant_args(),
            ]
            if self.parallel:
                cmd.append("--parallel")
            if self.workers is not None:
                cmd.extend(["--workers", str(self.workers)])
                cmd.extend(["--n_jobs", str(self.workers)])
            runs.append(Seq2SynthRun(dataset, None, "privacy", cmd))
        return runs

    def _cross_sectional_runs(self) -> list[Seq2SynthRun]:
        datasets = self._filter_cross_sectional_supported(
            self._filter_supported(
                "cross_sectional",
                self.datasets,
                set(resolve_datasets(["all"])),
            )
        )
        if not datasets:
            return []
        cmd = [
            sys.executable,
            str(Path("seq2synth") / "metrics" / "cross_sectional" / "run_cross_sectional.py"),
            "--datasets",
            *datasets,
        ]
        if self.models and "all" not in self.models:
            cmd.extend(["--models", *self.models])
        if self.variants:
            cmd.extend(["--variants", *self.variants])
        if self.parallel:
            cmd.append("--parallel")
        if self.workers is not None:
            cmd.extend(["--workers", str(self.workers)])
            cmd.extend(["--n_jobs", str(self.workers)])
        if self.dry_run:
            cmd.append("--dry_run")
        return [Seq2SynthRun("all", None, "cross_sectional", cmd)]

    def _longitudinal_runs(self) -> list[Seq2SynthRun]:
        datasets = self._filter_supported(
            "longitudinal",
            self.datasets,
            set(resolve_datasets(["all"])),
        )
        if not datasets:
            return []
        cmd = [
            sys.executable,
            str(Path("seq2synth") / "metrics" / "longitudinal" / "run_longitudinal.py"),
            "--datasets",
            *datasets,
        ]
        if self.models and "all" not in self.models:
            cmd.extend(["--models", *self.models])
        if self.variants:
            cmd.extend(["--variants", *self.variants])
        if not self.longitudinal_tt_standalone:
            cmd.append("--no-tt_standalone")
        cmd.extend(["--tt_num_iter_max", str(self.longitudinal_tt_num_iter_max)])
        if self.longitudinal_tt_max_flows is not None:
            cmd.extend(["--tt_max_flows", str(self.longitudinal_tt_max_flows)])
        cmd.extend(["--tt_sample_seed", str(self.longitudinal_tt_sample_seed)])
        if self.parallel:
            cmd.append("--parallel")
        if self.workers is not None:
            cmd.extend(["--workers", str(self.workers)])
            cmd.extend(["--n_jobs", str(self.workers)])
        if self.dry_run:
            cmd.append("--dry_run")
        return [Seq2SynthRun("all", None, "longitudinal", cmd)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        help="Metrics to run: all, structural, sd, timestamp, cross_sectional, longitudinal, privacy.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to run, or all.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model folders to run, or all.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional postprocessing variants to run.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run supported long-running metric internals concurrently.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count for --parallel where supported.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Alias for --workers for metrics that use n_jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--longitudinal-tt-standalone",
        action="store_true",
        default=False,
        help="Compute longitudinal per-feature standalone TT-Wasserstein distances.",
    )
    parser.add_argument(
        "--no-longitudinal-tt-standalone",
        action="store_false",
        dest="longitudinal_tt_standalone",
        help="Skip longitudinal per-feature standalone TT-Wasserstein distances.",
    )
    parser.add_argument(
        "--longitudinal-tt-num-iter-max",
        type=int,
        default=DEFAULT_TT_WASSERSTEIN_NUM_ITER_MAX,
        help="POT EMD numItermax for longitudinal exact TT-Wasserstein.",
    )
    parser.add_argument(
        "--longitudinal-tt-max-flows",
        type=int,
        default=DEFAULT_TT_WASSERSTEIN_MAX_FLOWS,
        help=(
            "Maximum trajectories per side for longitudinal exact TT-Wasserstein. "
            "Use 0 to disable deterministic TT-W trajectory sampling."
        ),
    )
    parser.add_argument(
        "--longitudinal-tt-sample-seed",
        type=int,
        default=DEFAULT_TT_WASSERSTEIN_SAMPLE_SEED,
        help="Seed for deterministic longitudinal TT-W trajectory sampling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Seq2Synth(
        datasets=args.datasets,
        models=args.models,
        metrics=args.metrics,
        variants=args.variants,
        parallel=args.parallel,
        workers=args.workers if args.workers is not None else args.n_jobs,
        dry_run=args.dry_run,
        longitudinal_tt_standalone=args.longitudinal_tt_standalone,
        longitudinal_tt_num_iter_max=args.longitudinal_tt_num_iter_max,
        longitudinal_tt_max_flows=args.longitudinal_tt_max_flows,
        longitudinal_tt_sample_seed=args.longitudinal_tt_sample_seed,
    )
    return runner.run()

if __name__ == "__main__":
    raise SystemExit(main())
