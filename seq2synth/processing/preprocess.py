import os
from pathlib import Path
import argparse
import json
import shutil
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", help="Dataset name to preprocess.")
args = parser.parse_args()

DATASET_NAME = args.dataset_name

BENCHMARK_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BENCHMARK_DIR / "data" / "real" / DATASET_NAME
PROCESSED_DATA_DIR = DATA_DIR.with_name(f"{DATA_DIR.name}_processed")

metadata_path = os.path.join(DATA_DIR, "metadata.json")

# -----------------------------
# 1. Load metadata
# -----------------------------
with open(metadata_path, "r") as f:
    metadata = json.load(f)

tables = metadata["tables"]

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
shutil.copy2(metadata_path, PROCESSED_DATA_DIR / "metadata.json")

# Classify sdtypes
NUMERIC_TYPES = {"numerical", "float", "integer"}
CATEGORICAL_TYPES = {"categorical", "boolean",}

# -----------------------------
# 2. Process each table
# -----------------------------
for table_name, table_meta in tables.items():

    csv_path = os.path.join(DATA_DIR, f"{table_name}.csv")

    if not os.path.exists(csv_path):
        print(f"Skip (no file): {csv_path}")
        continue

    print(f"Processing {csv_path}")

    df = pd.read_csv(csv_path)

    columns_meta = table_meta["columns"]

    for col, col_meta in columns_meta.items():

        if col not in df.columns:
            continue

        sdtype = col_meta["sdtype"]
        computer_representation = col_meta.get("computer_representation")

        if sdtype in NUMERIC_TYPES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)

            if computer_representation == "Int64":
                df[col] = df[col].round().astype("Int64")

        elif sdtype in CATEGORICAL_TYPES:
            df[col] = df[col].fillna("?")

        else:
            # Fallback for unknown types
            df[col] = df[col].fillna("@")

    # -----------------------------
    # 3. Save processed table
    # -----------------------------
    df.to_csv(PROCESSED_DATA_DIR / f"{table_name}.csv", index=False)

print("Done.")
