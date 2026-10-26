"""
peek_parquet.py - extract 50 rows with ALL columns from the big parquet, to find where a data problem lies.

What it does
  1. Takes rows spread evenly over the whole file (several row groups, a few rows from each), so you see
     different parts of Odisha, not just one corner.
  2. Keeps only rows where sand/silt/clay at 0 cm are present (use --include-missing to keep all).
  3. Saves ALL columns of those rows to  peek_50_rows.csv  (open it in Excel).
  4. Prints (a) the soil columns for the 50 rows, (b) the first row listed column by column.

USAGE (from the sm_pipeline folder):
    python scripts/peek_parquet.py
    python scripts/peek_parquet.py --rows 100
    python scripts/peek_parquet.py --include-missing
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "static" / "source" / \
    "ECOSTRESS_SMAP_SM_Odisha_2018_2026_with_LULC2_SOILTEX_GWE_inputs.parquet"
OUT = ROOT / "peek_50_rows.csv"
SOIL = ["lat", "lon", "sand_pct_0cm", "silt_pct_0cm", "clay_pct_0cm", "soil_tex_0cm", "soil_tex_0cm_name"]

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(SRC))
ap.add_argument("--out", default=str(OUT))
ap.add_argument("--rows", type=int, default=50)
ap.add_argument("--groups", type=int, default=10, help="how many row groups to sample from")
ap.add_argument("--include-missing", action="store_true")
a = ap.parse_args()

pf = pq.ParquetFile(a.src)
n_groups = pf.num_row_groups
picks = sorted({int(x) for x in np.linspace(0, n_groups - 1, min(a.groups, n_groups))})
per_group = int(np.ceil(a.rows / len(picks)))
print(f"File has {pf.metadata.num_rows:,} rows, {len(pf.schema.names)} columns, {n_groups} row groups.")
print(f"Sampling {per_group} rows from each of row groups {picks} ...\n")

parts = []
for g in picks:
    df = pf.read_row_group(g).to_pandas()
    if not a.include_missing:
        need = [c for c in ("sand_pct_0cm", "silt_pct_0cm", "clay_pct_0cm") if c in df.columns]
        df = df.dropna(subset=need)
    if len(df) == 0:
        continue
    take = np.linspace(0, len(df) - 1, min(per_group, len(df))).astype(int)
    s = df.iloc[take].copy()
    s.insert(0, "row_group", g)
    parts.append(s)
    del df

out = pd.concat(parts).head(a.rows).reset_index(drop=True)
out.to_csv(a.out, index=False)
print(f"Saved {len(out)} rows x {out.shape[1]} columns  ->  {a.out}\n")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)
cols = [c for c in SOIL if c in out.columns]
print("SOIL COLUMNS (0 cm) for the sampled rows:")
print(out[["row_group"] + cols].to_string(index=False))

print("\nFIRST ROW, every column:")
for c, v in out.iloc[0].items():
    print(f"  {c:40s} {v}")
