"""
04_lag_predict.py
------------------
Runs the lag-adjustment model (XGBoost.pkl) on the features from step 03 and
saves the predicted 9 km SMAP for the target date.

Outputs (in 02_lag_adjusted_sm/):
  lag_smap_pred_<T>.parquet   cell_id, lat, lon, sm_pred_lag
  lag_smap_pred_<T>.tif       the same as a 9 km float32 GeoTIFF (EPSG:6933)

Usage:
    python scripts/04_lag_predict.py
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


def run():
    meta_path = os.path.join(config.LAG_DIR, "latest_lag_run.json")
    if not os.path.exists(meta_path):
        sys.exit("No latest_lag_run.json. Run 03_build_lag_features.py first.")
    with open(meta_path) as f:
        meta = json.load(f)
    T = meta["target_date"]

    feats = pd.read_parquet(meta["features_file"])
    X = feats[config.FEATURE_COLUMNS].to_numpy(dtype=float)   # fixed column order, no names stored in the model

    if not os.path.exists(config.LAG_MODEL_PATH):
        sys.exit(f"Lag model not found: {config.LAG_MODEL_PATH}")
    model = joblib.load(config.LAG_MODEL_PATH)
    if getattr(model, "n_features_in_", len(config.FEATURE_COLUMNS)) != len(config.FEATURE_COLUMNS):
        sys.exit(f"Model expects {model.n_features_in_} features but config has {len(config.FEATURE_COLUMNS)}.")
    try:
        model.set_params(device="cpu")   # trained on GPU; predict on CPU
    except Exception:
        pass

    print(f"Predicting {len(X):,} SMAP cells for {T} (anchor {meta['anchor_date']}) ...")
    pred = np.clip(model.predict(X), 0.0, 1.0).astype(np.float32)

    out = feats[["cell_id", "lat", "lon"]].copy()
    out["sm_pred_lag"] = pred
    out["date"] = T
    pq = os.path.join(config.LAG_DIR, f"lag_smap_pred_{T}.parquet")
    out.to_parquet(pq, index=False)

    # 9 km GeoTIFF of the prediction
    g = h5_to_tiff.read_smap_grid(meta["anchor_file"])
    arr = np.full(g["sm"].shape, np.nan, dtype=np.float32)
    ids = out["cell_id"].to_numpy(np.int64)
    r = ids // g["ncols_full"] - g["r0"]
    c = ids % g["ncols_full"] - g["c0"]
    arr[r, c] = pred
    tif = h5_to_tiff.write_float_geotiff(
        os.path.join(config.LAG_DIR, f"lag_smap_pred_{T}.tif"), arr, g["transform"], g["crs"])

    meta["lag_pred_file"] = pq
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  predicted SMAP: min {pred.min():.3f}  mean {pred.mean():.3f}  max {pred.max():.3f}")
    print(f"Saved -> {pq}\n         {tif}")


if __name__ == "__main__":
    run()
