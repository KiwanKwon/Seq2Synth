import pandas as pd
import numpy as np
import wfdb
import ast
from pathlib import Path

# ----------------------------
# 1. Load raw waveforms
# ----------------------------
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(str(path / f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(str(path / f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

BENCHMARK_DIR = Path(__file__).resolve().parents[2]
REAL_DIR = BENCHMARK_DIR / "data" / "real"
path = REAL_DIR / "ptbxl"
sampling_rate = 100
step_us = int(1_000_000 / sampling_rate)  # 100Hz -> 10ms = 10000us

# ----------------------------
# 2. Load metadata
# ----------------------------
Y = pd.read_csv(path / 'ptbxl_database.csv', index_col='ecg_id', low_memory=False)
Y['recording_date'] = pd.to_datetime(Y['recording_date'], errors='coerce')

Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

agg_df = pd.read_csv(path / 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Convert to strings for GAN-family model stability
Y['diagnostic_superclass'] = Y['diagnostic_superclass'].apply(
    lambda x: '|'.join(sorted(x)) if isinstance(x, list) else str(x)
)

# ----------------------------
# 3. Load waveforms
# ----------------------------
X = load_raw_data(Y, sampling_rate, path)

N, T, D = X.shape

# ----------------------------
# 4. Convert to long format
# ----------------------------
X_2d = X.reshape(N * T, D)
lead_cols = [f'lead_{i}' for i in range(D)]
df_X = pd.DataFrame(X_2d, columns=lead_cols)

ecg_ids = Y.index.to_numpy()
ecg_id_rep = np.repeat(ecg_ids, T)

start_times = np.repeat(Y['recording_date'].to_numpy(), T)
offset = np.tile(np.arange(T), N)
delta = pd.to_timedelta(offset * step_us, unit='us')

timestamps = pd.to_datetime(start_times) + delta

df_idx = pd.DataFrame({
    'ecg_id': ecg_id_rep,
    'timestamp': timestamps
})

# Repeat Y metadata
df_Y_rep = Y.loc[ecg_id_rep].reset_index(drop=True)

# Final merge
df_out = pd.concat(
    [df_idx.reset_index(drop=True),
     df_Y_rep.reset_index(drop=True),
     df_X],
    axis=1
)

df_out.to_csv(path / 'ptbxl_long_with_timestamp.csv', index=False)

print(df_out.shape)
print(df_out.head())
