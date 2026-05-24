from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import REAL_DIR, RESULTS_DIR, SYNTH_DIR


NUMERIC_SDTYPES = {"numerical", "float", "integer"}
CATEGORICAL_SDTYPES = {"categorical", "boolean"}
DATETIME_SDTYPES = {"datetime"}
ID_SDTYPES = {"id"}

VARIANT_SUFFIXES: dict[str, str] = {
    "raw": "",
    "default": "_postprocessed",
    "sequential": "_postprocessed_sequential",
    "id_fill": "_postprocessed_id_fill",
    "cross_fill": "_postprocessed_cross_fill",
    "prev_fill": "_postprocessed_prev_fill",
    "interp_fill": "_postprocessed_interp_fill",
}

TABLE_TIME_COL_OVERRIDES: dict[str, dict[str, str]] = {
    "home_credit": {
        "bureau": "DAYS_CREDIT",
        "bureau_balance": "MONTHS_BALANCE",
        "previous_application": "DAYS_DECISION",
        "POS_CASH_balance": "MONTHS_BALANCE",
        "installments_payments": "DAYS_INSTALMENT",
        "credit_card_balance": "MONTHS_BALANCE",
    },
}


@dataclass(frozen=True)
class TableConfig:
    name: str
    filename: str
    primary_key: str | None
    columns: dict[str, dict[str, Any]]

    @property
    def numeric_cols(self) -> list[str]:
        return _cols_by_sdtype(self.columns, NUMERIC_SDTYPES)

    @property
    def categorical_cols(self) -> list[str]:
        return _cols_by_sdtype(self.columns, CATEGORICAL_SDTYPES)

    @property
    def datetime_cols(self) -> list[str]:
        return _cols_by_sdtype(self.columns, DATETIME_SDTYPES)

    @property
    def id_cols(self) -> list[str]:
        return _cols_by_sdtype(self.columns, ID_SDTYPES)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    real_dir: Path
    synth_dir: Path
    results_dir: Path
    metadata_path: Path
    tables: dict[str, TableConfig]
    relationships: list[dict[str, Any]]
    sequence_table: str | None = None
    sequence_key: str | None = None
    timestamp_table: str | None = None
    timestamp_key: str | None = None
    timestamp_col: str | None = None
    timestamp_periodicity: str | None = None
    timestamp_repr: str | None = None
    timestamp_format: str | None = None

    @property
    def child_tables(self) -> list[str]:
        children = [
            rel["child_table_name"]
            for rel in self.relationships
            if "child_table_name" in rel
        ]
        return list(dict.fromkeys(children))

    @property
    def primary_child_table(self) -> str | None:
        if self.sequence_table:
            return self.sequence_table
        if self.child_tables:
            return self.child_tables[0]
        if self.tables:
            return next(iter(self.tables))
        return None


def _cols_by_sdtype(
    columns: dict[str, dict[str, Any]], sdtypes: set[str]
) -> list[str]:
    return [name for name, info in columns.items() if info.get("sdtype") in sdtypes]


def table_filename(table_name: str) -> str:
    return f"{table_name}.csv"


def infer_dataset_fields(
    tables: dict[str, TableConfig],
    relationships: list[dict[str, Any]],
) -> dict[str, Any]:
    """Infer common benchmark fields directly from SDV-style metadata."""
    child_tables = [
        rel["child_table_name"]
        for rel in relationships
        if rel.get("child_table_name") in tables
    ]
    table_name = child_tables[0] if child_tables else next(iter(tables), None)
    if table_name is None:
        return {}

    table = tables[table_name]
    relationship = next(
        (
            rel
            for rel in relationships
            if rel.get("child_table_name") == table_name
        ),
        {},
    )
    key_col = (
        relationship.get("child_foreign_key")
        or table.primary_key
        or (table.id_cols[0] if table.id_cols else None)
    )
    timestamp_col = table.datetime_cols[0] if table.datetime_cols else None
    timestamp_format = None
    if timestamp_col:
        timestamp_format = table.columns[timestamp_col].get("datetime_format")

    return {
        "sequence_table": table_name,
        "sequence_key": key_col,
        "timestamp_table": table_name,
        "timestamp_key": key_col,
        "timestamp_col": timestamp_col,
        "timestamp_format": timestamp_format,
    }


def load_metadata(dataset: str) -> dict[str, Any]:
    path = REAL_DIR / dataset / "metadata.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset_config(dataset: str) -> DatasetConfig:
    metadata_path = REAL_DIR / dataset / "metadata.json"
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    tables = {
        name: TableConfig(
            name=name,
            filename=table_filename(name),
            primary_key=table.get("primary_key"),
            columns=table.get("columns", {}),
        )
        for name, table in metadata.get("tables", {}).items()
    }
    inferred = infer_dataset_fields(tables, metadata.get("relationships", []))
    return DatasetConfig(
        name=dataset,
        real_dir=REAL_DIR / dataset,
        synth_dir=SYNTH_DIR / dataset,
        results_dir=RESULTS_DIR / dataset,
        metadata_path=metadata_path,
        tables=tables,
        relationships=metadata.get("relationships", []),
        **inferred,
    )


def available_datasets() -> list[str]:
    return sorted(
        path.name
        for path in REAL_DIR.iterdir()
        if path.is_dir() and (path / "metadata.json").exists()
    )


def available_models(dataset: str) -> list[str]:
    synth_dir = SYNTH_DIR / dataset
    if not synth_dir.exists():
        return []
    return sorted(path.name for path in synth_dir.iterdir() if path.is_dir())


def resolve_datasets(names: list[str] | None) -> list[str]:
    datasets = available_datasets()
    if not names or "all" in names:
        return datasets
    unknown = sorted(set(names) - set(datasets))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Known: {datasets}")
    return names


def resolve_models(dataset: str, names: list[str] | None) -> list[str]:
    models = available_models(dataset)
    if not names or "all" in names:
        return models
    return names


def discover_synth_variants(synth_model_dir: Path, table_name: str) -> list[tuple[str, Path]]:
    return [
        (variant, synth_model_dir / f"{table_name}{suffix}.csv")
        for variant, suffix in VARIANT_SUFFIXES.items()
        if (synth_model_dir / f"{table_name}{suffix}.csv").exists()
    ]
