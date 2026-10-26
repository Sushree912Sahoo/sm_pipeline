"""
05_downscale_to_70m.py
-----------------------
Runs the downscaling model (RF.pkl) to turn the predicted 9 km SMAP into 70 m
soil moisture.

Model inputs per 70 m pixel: smap (its parent 9 km cell's predicted value),
doy_sin, doy_cos, LULC one-hot, soil texture one-hot. Because pixels in the
same (cell, LULC, soil) combo share identical inputs, the model is run once
per combo (static/combos.parquet) and the result is painted onto the 70 m grid.

Output (in 03_downscaled_sm/):
  sm_70m_<T>.tif    float32 GeoTIFF, EPSG:4326, nodata = -9999

Usage:
    python scripts/05_downscale_to_70m.py
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import h5_to_tiff

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree


def build_model_inputs(combos, smap, T, bundle):
    """18-column input frame in the exact order stored in RF.pkl."""
    prep = bundle["prep_state"]
    doy = pd.Timestamp(T).dayofyear
    X = pd.DataFrame(index=combos.index)
    X["smap"] = smap
    X["smap"] = X["smap"].fillna(prep["med"]["smap"])
    X["doy_sin"] = np.sin(2 * np.pi * doy / config.DOY_PERIOD)
    X["doy_cos"] = np.cos(2 * np.pi * doy / config.DOY_PERIOD)
    source = {"lulc": combos["lulc"], "soil_tex_0cm": combos["soil_tex"]}
    for name in prep["feat_cat"]:
        for c in prep["cats"][name]:
            X[f"{name}_{c}"] = (source[name] == c).astype(np.float32)
    return X[bundle["columns"]]


def run():
    meta_path = os.path.join(config.LAG_DIR, "latest_lag_run.json")
    if not os.path.exists(meta_path):
        sys.exit("No latest_lag_run.json. Run steps 03 and 04 first.")
    meta = json.load(open(meta_path))
    if "lag_pred_file" not in meta:
        sys.exit("Lag prediction missing. Run 04_lag_predict.py first.")
    T = meta["target_date"]

    pts_path = os.path.join(config.STATIC_DIR, "points_70m.parquet")
    if not os.path.exists(pts_path):
        sys.exit("Static layer missing. Run scripts/00_build_static_layer.py once first.")
    grid = json.load(open(os.path.join(config.STATIC_DIR, "grid_meta.json")))
    pts = pd.read_parquet(pts_path)
    combos = pd.read_parquet(os.path.join(config.STATIC_DIR, "combos.parquet"))

    lag = pd.read_parquet(meta["lag_pred_file"]).set_index("cell_id")["sm_pred_lag"]

    # 9 km cells that hold 70 m pixels but got no lag prediction (SMAP gap, far from stations,
    # coast...) are filled from the nearest predicted cell instead of leaving a hole.
    ncols_full = meta.get("ncols_full") or h5_to_tiff.read_smap_grid(meta["anchor_file"])["ncols_full"]
    need = np.setdiff1d(combos["cell_id"].unique(), lag.index.to_numpy())
    filled_cells = pd.Index([])
    if len(need) and config.LAG_FILL_MAX_CELLS > 0 and len(lag):
        have = lag.index.to_numpy(np.int64)
        tree = cKDTree(np.column_stack([have // ncols_full, have % ncols_full]))
        dist, idx = tree.query(np.column_stack([need // ncols_full, need % ncols_full]))
        ok = dist <= config.LAG_FILL_MAX_CELLS
        fill = pd.Series(lag.to_numpy()[idx[ok]], index=need[ok])
        filled_cells = fill.index
        lag = pd.concat([lag, fill])
        print(f"{len(need):,} 9 km cell(s) had no lag prediction; {int(ok.sum()):,} filled from the nearest "
              f"predicted cell (max {config.LAG_FILL_MAX_CELLS} cells away), {int((~ok).sum()):,} could not be.")

    smap = combos["cell_id"].map(lag)
    valid = smap.notna().to_numpy()
    combo_filled = combos["cell_id"].isin(filled_cells).to_numpy()
    print(f"Target date {T}: {valid.sum():,} of {len(combos):,} combos have a 9 km SMAP value to downscale.")

    bundle = joblib.load(config.DOWNSCALE_MODEL_PATH)
    model = bundle["model"]
    model.n_jobs = -1
    X = build_model_inputs(combos, smap, T, bundle)

    combo_pred = np.full(len(combos), np.nan, dtype=np.float32)
    combo_pred[valid] = np.clip(model.predict(X[valid]), 0.0, 1.0)

    point_pred = combo_pred[pts["combo_id"].to_numpy()]
    raster = np.full((grid["nrows"], grid["ncols"]), config.NODATA, dtype=np.float32)
    good = np.isfinite(point_pred)
    raster[pts["row"].to_numpy()[good].astype(np.int64),
           pts["col"].to_numpy()[good].astype(np.int64)] = point_pred[good]

    transform = from_origin(grid["west"], grid["north"], grid["res_x"], grid["res_y"])
    os.makedirs(config.DOWNSCALED_DIR, exist_ok=True)
    out = os.path.join(config.DOWNSCALED_DIR, f"sm_70m_{T}.tif")
    with rasterio.open(out, "w", driver="GTiff", height=raster.shape[0], width=raster.shape[1],
                       count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=config.NODATA, compress="deflate", predictor=3,
                       tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(raster, 1)

    # QA map: 0 = not in the static layer, 1 = direct, 2 = filled from a neighbouring 9 km cell,
    # 3 = in the static layer but no prediction possible
    qa = np.zeros((grid["nrows"], grid["ncols"]), dtype=np.uint8)
    pt_filled = combo_filled[pts["combo_id"].to_numpy()]
    qa_val = np.where(good, np.where(pt_filled, 2, 1), 3).astype(np.uint8)
    qa[pts["row"].to_numpy().astype(np.int64), pts["col"].to_numpy().astype(np.int64)] = qa_val
    qa_out = os.path.join(config.DOWNSCALED_DIR, f"sm_70m_{T}_QA.tif")
    with rasterio.open(qa_out, "w", driver="GTiff", height=qa.shape[0], width=qa.shape[1], count=1,
                       dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0,
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(qa, 1)

    p = point_pred[good]
    n = len(point_pred)
    print(f"\nCoverage of the {n:,} pixels in the static layer:")
    print(f"  direct prediction           : {int((qa_val == 1).sum()):,} ({100 * (qa_val == 1).sum() / n:.1f}%)")
    print(f"  filled from nearby 9 km cell: {int((qa_val == 2).sum()):,} ({100 * (qa_val == 2).sum() / n:.1f}%)")
    print(f"  no prediction possible      : {int((qa_val == 3).sum()):,} ({100 * (qa_val == 3).sum() / n:.1f}%)")
    print("  Pixels outside the static layer (never in your training table) stay nodata by design.")
    print(f"QA map -> {qa_out}   (1 = direct, 2 = filled, 3 = no prediction)")
    print(f"Soil moisture: min {p.min():.3f}  mean {p.mean():.3f}  max {p.max():.3f}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    run()
