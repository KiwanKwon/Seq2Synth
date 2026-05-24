"""
Extracts a referentially-consistent subsample of the Google Cluster Trace 2011-2 dataset,
seeded by machine_id.

Relationship graph:
  machine_events      (PK: machine_id)
      └── machine_attributes  (FK: machine_id)
  job_events          (PK: job_id)
      └── task_events (FK: job_id; also FK: machine_id → machine_events)
              ├── task_usage        (FK: job_id + task_index)
              └── task_constraints  (FK: job_id + task_index)

Strategy:
  1.  Sample N unique machines from machine_events.
  2.  Collect machine_events rows for those machines.
  3.  Collect machine_attributes rows for those machines.
  4.  Scan task_events shards for rows whose machine_id is in the sampled set;
      collect job_ids and (job_id, task_index) pairs.
  5.  Scan task_usage shards for rows matching those (job_id, task_index) pairs.
  6.  Scan task_constraints shards for rows matching those (job_id, task_index) pairs.
  7.  Scan job_events shards for rows matching the collected job_ids.
  8.  Write all six tables to OUTPUT_DIR as CSV with headers.
  9.  Write a machine-table-free copy (task_events / task_usage /
      task_constraints / job_events) to FINAL_DIR, mirroring the
      google_cluster_processed schema used by SynthereLA.
"""

import csv
import glob
import gzip
import random
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parents[2]
REAL_DIR      = BENCHMARK_DIR / "data" / "real"
BASE_DIR      = REAL_DIR / "google_cluster"
OUTPUT_DIR    = BASE_DIR / "subsample_by_machine"
FINAL_DIR     = BASE_DIR / "subsample_by_machine_processed"

N_MACHINES          = 500
N_MACHINE_SHARDS    = None  # small table – read all shards
N_ATTR_SHARDS       = None  # small table – read all shards
N_TASK_SHARDS       = 5
N_USAGE_SHARDS      = 5
N_CONSTRAINT_SHARDS = 5
N_JOB_SHARDS        = 5

RANDOM_SEED = 42
# ───────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

COLUMNS = {
    "job_events": [
        "time", "missing_info", "job_id", "event_type",
        "user", "scheduling_class", "job_name", "logical_job_name",
    ],
    "task_events": [
        "time", "missing_info", "job_id", "task_index", "machine_id",
        "event_type", "user", "scheduling_class", "priority",
        "cpu_request", "memory_request", "disk_space_request",
        "different_machines_restriction",
    ],
    "machine_events": [
        "time", "machine_id", "event_type", "platform_id", "cpus", "memory",
    ],
    "machine_attributes": [
        "time", "machine_id", "attribute_name",
        "attribute_value", "attribute_deleted",
    ],
    "task_constraints": [
        "time", "job_id", "task_index",
        "comparison_operator", "attribute_name", "attribute_value",
    ],
    "task_usage": [
        "start_time", "end_time", "job_id", "task_index", "machine_id",
        "cpu_rate", "canonical_memory_usage", "assigned_memory_usage",
        "unmapped_page_cache", "total_page_cache", "maximum_memory_usage",
        "disk_io_time", "local_disk_space_usage", "maximum_cpu_rate",
        "maximum_disk_io_time", "cycles_per_instruction",
        "memory_accesses_per_instruction", "sample_portion",
        "aggregation_type", "sampled_cpu_usage",
    ],
}

TOTAL_STEPS = 9


def iter_gz_csv(paths):
    """Yield rows (as lists) from a sequence of gzip-compressed CSV files."""
    for p in paths:
        with gzip.open(p, "rt") as f:
            yield from csv.reader(f)


def shards(table, n=None):
    """Return sorted shard paths for *table*, optionally capped at *n*."""
    files = sorted(glob.glob(str(BASE_DIR / table / "part-*.csv.gz")))
    return files[:n] if n is not None else files


def write_csv(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)
    print(f"    wrote {len(rows):>8,} rows  →  {path.relative_to(BASE_DIR)}")


def step(n, msg):
    print(f"\n[{n}/{TOTAL_STEPS}] {msg}")


# ── Step 1: sample machines ────────────────────────────────────────────────────
step(1, "Reading machine_events …")
t0 = time.time()

all_machine_rows, seen_machines = [], set()
for row in iter_gz_csv(shards("machine_events", N_MACHINE_SHARDS)):
    if len(row) >= 2:
        seen_machines.add(row[1])
        all_machine_rows.append(row)

sampled_machines = (
    seen_machines if len(seen_machines) <= N_MACHINES
    else set(random.sample(sorted(seen_machines), N_MACHINES))
)
machine_rows = [r for r in all_machine_rows if len(r) >= 2 and r[1] in sampled_machines]
print(f"  {len(seen_machines):,} unique machines total; "
      f"sampled {len(sampled_machines):,} → {len(machine_rows):,} rows  "
      f"({time.time() - t0:.1f}s)")

# ── Step 2: machine_attributes ─────────────────────────────────────────────────
step(2, "Reading machine_attributes …")
t0 = time.time()

attr_rows = [
    row for row in iter_gz_csv(shards("machine_attributes", N_ATTR_SHARDS))
    if len(row) >= 2 and row[1] in sampled_machines
]
print(f"  kept {len(attr_rows):,} rows  ({time.time() - t0:.1f}s)")

# ── Step 3: task_events ────────────────────────────────────────────────────────
step(3, f"Scanning {N_TASK_SHARDS} task_events shard(s) …")
t0 = time.time()

task_rows, job_ids, task_keys = [], set(), set()
for row in iter_gz_csv(shards("task_events", N_TASK_SHARDS)):
    if len(row) >= 5 and row[4] in sampled_machines:
        task_rows.append(row)
        job_ids.add(row[2])
        task_keys.add((row[2], row[3]))

print(f"  kept {len(task_rows):,} rows | "
      f"{len(job_ids):,} unique jobs | "
      f"{len(task_keys):,} (job, task) pairs  ({time.time() - t0:.1f}s)")

# ── Step 4: task_usage ─────────────────────────────────────────────────────────
step(4, f"Scanning {N_USAGE_SHARDS} task_usage shard(s) …")
t0 = time.time()

usage_rows = [
    row for row in iter_gz_csv(shards("task_usage", N_USAGE_SHARDS))
    if len(row) >= 4 and (row[2], row[3]) in task_keys
]
print(f"  kept {len(usage_rows):,} rows  ({time.time() - t0:.1f}s)")

# ── Step 5: task_constraints ───────────────────────────────────────────────────
step(5, f"Scanning {N_CONSTRAINT_SHARDS} task_constraints shard(s) …")
t0 = time.time()

constraint_rows = [
    row for row in iter_gz_csv(shards("task_constraints", N_CONSTRAINT_SHARDS))
    if len(row) >= 3 and (row[1], row[2]) in task_keys
]
print(f"  kept {len(constraint_rows):,} rows  ({time.time() - t0:.1f}s)")

# ── Step 6: job_events ─────────────────────────────────────────────────────────
step(6, f"Scanning {N_JOB_SHARDS} job_events shard(s) …")
t0 = time.time()

job_rows = [
    row for row in iter_gz_csv(shards("job_events", N_JOB_SHARDS))
    if len(row) >= 3 and row[2] in job_ids
]
print(f"  kept {len(job_rows):,} rows for {len(job_ids):,} jobs  ({time.time() - t0:.1f}s)")

# ── Step 7: summary ────────────────────────────────────────────────────────────
step(7, "Row counts")
print(f"  {'machine_events':<22} {len(machine_rows):>8,}")
print(f"  {'machine_attributes':<22} {len(attr_rows):>8,}")
print(f"  {'task_events':<22} {len(task_rows):>8,}")
print(f"  {'task_usage':<22} {len(usage_rows):>8,}")
print(f"  {'task_constraints':<22} {len(constraint_rows):>8,}")
print(f"  {'job_events':<22} {len(job_rows):>8,}")

# ── Step 8: write full subsample ───────────────────────────────────────────────
step(8, f"Writing full subsample → {OUTPUT_DIR.name}/")

write_csv(OUTPUT_DIR / "machine_events.csv",     COLUMNS["machine_events"],     machine_rows)
write_csv(OUTPUT_DIR / "machine_attributes.csv", COLUMNS["machine_attributes"], attr_rows)
write_csv(OUTPUT_DIR / "task_events.csv",        COLUMNS["task_events"],        task_rows)
write_csv(OUTPUT_DIR / "task_usage.csv",         COLUMNS["task_usage"],         usage_rows)
write_csv(OUTPUT_DIR / "task_constraints.csv",   COLUMNS["task_constraints"],   constraint_rows)
write_csv(OUTPUT_DIR / "job_events.csv",         COLUMNS["job_events"],         job_rows)

# ── Step 9: write machine-table-free copy (SynthereLA schema) ──────────────────
step(9, f"Writing SynthereLA-compatible copy → {FINAL_DIR.name}/  "
        f"(machine_events & machine_attributes excluded)")

write_csv(FINAL_DIR / "task_events.csv",      COLUMNS["task_events"],      task_rows)
write_csv(FINAL_DIR / "task_usage.csv",       COLUMNS["task_usage"],       usage_rows)
write_csv(FINAL_DIR / "task_constraints.csv", COLUMNS["task_constraints"], constraint_rows)
write_csv(FINAL_DIR / "job_events.csv",       COLUMNS["job_events"],       job_rows)

print("\nDone.")
