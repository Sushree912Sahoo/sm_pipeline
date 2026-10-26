"""
00_build_static_layer.py   (RUN ONCE)
--------------------------------------
Builds the static 70 m layer that the downscaling model needs every day,
from the big training parquet. Reads only 4 columns (lat, lon, LULC_2025,
soil_tex_0cm), so it does not need the whole table in memory.

For every 70 m point it records:
  - its row/col on the 70 m grid (EPSG:4326, regular spacing),
  - which 9 km SMAP cell contains it (global cell id),
  - its LULC class and soil texture class.
Points are then grouped into unique (cell, LULC, soil) COMBOS. Every point in
a combo gets the same model inputs, so the downscaler predicts once per combo
(about 100 thousand rows) instead of once per point (millions).

Needs ONE SMAP .h5 file (any date) to read the 9 km grid from.

Outputs (in static/):
  points_70m.parquet   row, col, combo_id        (one row per 70 m point)
  combos.parquet       combo_id, cell_id, lulc, soil_tex
  grid_meta.json       70 m grid origin / spacing / size

Usage:
    python scripts/00_build_static_layer.py
    python scripts/00_build_static_layer.py --source D:/data/training.parquet --smap-file path/to/file.h5
"""

import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import h5_to_tiff

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from rasterio.warp import transform as warp_transform


def regular_axis(values, name):
    """Spacing of a (nearly) regular axis from its unique values."""
    u = np.unique(np.round(values, 6))
    res = float(np.median(np.diff(u)))
    print(f"  {name}: {len(u):,} distinct values, spacing {res:.7f} deg")
    return res


def main(source, smap_file):
    # ---- 1. read the 4 columns we need ---------------------------------
    cols = [config.STATIC_LAT_COL, config.STATIC_LON_COL, config.STATIC_LULC_COL, config.STATIC_SOIL_COL]
    print(f"Reading columns {cols} from {source} ...")
    tbl = ds.dataset(source, format="parquet").to_table(columns=cols).to_pandas()
    print(f"  {len(tbl):,} points")
    lat = tbl[config.STATIC_LAT_COL].to_numpy(np.float64)
    lon = tbl[config.STATIC_LON_COL].to_numpy(np.float64)
    lulc = tbl[config.STATIC_LULC_COL].fillna(0).to_numpy(np.int64)
    soil_null = tbl[config.STATIC_SOIL_COL].isna()
    soil = tbl[config.STATIC_SOIL_COL].fillna(0).to_numpy(np.int64)
    n_soil_null = int(soil_null.sum())
    del tbl

    # ---- 2. 70 m grid (regular lat/lon) ---------------------------------
    print("Deriving the 70 m grid ...")
    res_x = regular_axis(lon, "lon")
    res_y = regular_axis(lat, "lat")
    lon_min, lat_max = float(lon.min()), float(lat.max())
    colf = (lon - lon_min) / res_x
    rowf = (lat_max - lat) / res_y
    col = np.rint(colf).astype(np.int64)
    row = np.rint(rowf).astype(np.int64)
    off = max(np.abs(colf - col).max(), np.abs(rowf - row).max())
    if off > 0.1:
        print(f"  WARNING: points are up to {off:.2f} cell from a regular grid; check the source table.")
    ncols, nrows = int(col.max()) + 1, int(row.max()) + 1
    print(f"  grid {nrows:,} rows x {ncols:,} cols")

    # ---- 3. which 9 km SMAP cell holds each 70 m grid column / row ------
    print(f"Reading the SMAP grid from {os.path.basename(smap_file)} ...")
    g = h5_to_tiff.read_smap_grid(smap_file)
    # EPSG:6933: x depends only on lon, y only on lat -> map columns and rows separately
    lon_centres = lon_min + np.arange(ncols) * res_x
    lat_centres = lat_max - np.arange(nrows) * res_y
    x70, _ = warp_transform("EPSG:4326", config.SMAP_CRS, lon_centres.tolist(), [20.0] * ncols)
    _, y70 = warp_transform("EPSG:4326", config.SMAP_CRS, [84.0] * nrows, lat_centres.tolist())
    x70, y70 = np.asarray(x70), np.asarray(y70)

    cell_col = np.floor((x70 - (g["xs"][0] - g["res_x"] / 2)) / g["res_x"]).astype(np.int64)
    cell_row = np.floor(((g["ys"][0] + g["res_y"] / 2) - y70) / g["res_y"]).astype(np.int64)
    ncol_crop, nrow_crop = len(g["xs"]), len(g["ys"])
    col_ok = (cell_col >= 0) & (cell_col < ncol_crop)
    row_ok = (cell_row >= 0) & (cell_row < nrow_crop)

    pt_ok = col_ok[col] & row_ok[row]
    n_out = int((~pt_ok).sum())
    if n_out:
        print(f"  WARNING: {n_out:,} points lie outside the cropped SMAP grid and are dropped. "
              f"Widen config.CROP_MARGIN_DEG if this is large.")
    row, col, lulc, soil = row[pt_ok], col[pt_ok], lulc[pt_ok], soil[pt_ok]

    cell_id = ((g["r0"] + cell_row[row]) * g["ncols_full"] + (g["c0"] + cell_col[col])).astype(np.int64)

    # ---- 4. unique (cell, lulc, soil) combos -----------------------------
    print("Grouping points into unique (cell, LULC, soil) combos ...")
    key = cell_id * 65536 + lulc * 256 + soil
    ukeys, combo_id = np.unique(key, return_inverse=True)
    combos = pd.DataFrame({
        "combo_id": np.arange(len(ukeys), dtype=np.int32),
        "cell_id": (ukeys // 65536).astype(np.int64),
        "lulc": ((ukeys % 65536) // 256).astype(np.int16),
        "soil_tex": (ukeys % 256).astype(np.int16),
    })
    print(f"  {len(row):,} points -> {len(combos):,} combos "
          f"({len(row) / max(len(combos), 1):.0f} points per combo)")

    # ---- 5. save ---------------------------------------------------------
    os.makedirs(config.STATIC_DIR, exist_ok=True)
    pd.DataFrame({
        "row": row.astype(np.int16), "col": col.astype(np.int16),
        "combo_id": combo_id.astype(np.int32),
    }).to_parquet(os.path.join(config.STATIC_DIR, "points_70m.parquet"), index=False)
    combos.to_parquet(os.path.join(config.STATIC_DIR, "combos.parquet"), index=False)
    meta = dict(
        lon_min=lon_min, lat_max=lat_max, res_x=res_x, res_y=res_y,
        west=lon_min - res_x / 2, north=lat_max + res_y / 2,
        nrows=nrows, ncols=ncols, crs="EPSG:4326",
        n_points=int(len(row)), n_combos=int(len(combos)),
        soil_null_points=n_soil_null, lulc_column=config.STATIC_LULC_COL,
    )
    with open(os.path.join(config.STATIC_DIR, "grid_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved static layer -> {config.STATIC_DIR}")
    print(f"  Points with no soil texture (all-zero soil inputs): {n_soil_null:,} "
          f"({100 * n_soil_null / max(meta['n_points'], 1):.2f}%)")
    print("Next: run the daily pipeline  (python run_pipeline.py)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the static 70 m layer (run once).")
    ap.add_argument("--source", default=config.STATIC_SOURCE,
                    help="parquet file or folder of parquet files (default: config.STATIC_SOURCE)")
    ap.add_argument("--smap-file", default=None, help="any SMAP .h5 (default: first in 01_raw_data/smap)")
    a = ap.parse_args()

    smap = a.smap_file
    if smap is None:
        found = sorted(glob.glob(os.path.join(config.SMAP_RAW_DIR, "*.h5")))
        if not found:
            sys.exit(f"No SMAP .h5 found in {config.SMAP_RAW_DIR}. Run 01_download_smap.py first "
                     f"or pass --smap-file.")
        smap = found[0]
    if not os.path.exists(a.source):
        sys.exit(f"Source parquet not found: {a.source}\nPass --source or set config.STATIC_SOURCE.")
    main(a.source, smap)
