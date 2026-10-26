"""
================================================================================
07_gw_predict.py   -   run daily, BEFORE 08_irrigation_advice.py

For every farm in farm_inputs/farms.csv (only farm_id, lat, lon are needed) it
  1. finds the nearest point in the GWE-inputs parquet,
  2. reads that point's GroundElevation_amsl, TP, Climate, Geology, LULC,
  3. adds Year, sin_day, cos_day from the chosen date,
  4. runs rf_gwe_final.pkl (same way 04_lag_predict.py runs the lag model),
  5. converts the result to depth-to-water and writes  farm_inputs/gwl_today.csv
     (farm_id, gwl_depth_m, gwe_pred, ground_elevation_amsl, snap_m, warnings)  for  08 --gwl-file.
     gwe_pred and ground_elevation_amsl are in m above mean sea level (Decision 3 needs both).

FILES
    static/source/ECOSTRESS_SMAP_SM_Odisha_2018_2026_with_LULC2_SOILTEX_GWE_inputs.parquet
    models/groundwater/rf_gwe_final.pkl
    models/groundwater/feature_config.json         (your feature list / order)
    farm_inputs/farms.csv

FIRST TIME (one-time, builds a small lookup so the 27-million-row file is not
read every day -> static/gw_point_lookup.parquet):
    python scripts/07_gw_predict.py --list-columns       # shows ALL columns + types
    python scripts/07_gw_predict.py --date 2026-10-12    # builds the lookup, then predicts

DAILY:
    python scripts/07_gw_predict.py --date 2026-10-12
    python scripts/08_irrigation_advice.py --date 2026-10-12 --gwl-file farm_inputs/gwl_today.csv

WHAT YOU MAY NEED TO EDIT (top of file): COLUMN_OVERRIDES if auto-detection picks
the wrong parquet column.

THINGS TO CONFIRM
  * --gwe-mode elevation (default): model output GWE = groundwater level in m above
    sea level, depth = GroundElevation_amsl - GWE.   --gwe-mode depth: output is
    already depth below ground. Depth should look like ~1-40 m.
  * TP: NOT in your parquet. Give it with --tp VALUE (all farms) or --tp-file (farm_id,TP).
  * Climate/Geology/LULC must already be numeric training codes in the parquet.
================================================================================
"""
import argparse
import json
import math
import re
import sys
from datetime import date as date_cls, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# ----------------------------------------------------------------------------- #
# CONFIG
# ----------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
SRC_PARQUET = ROOT / "static" / "source" / \
    "input.parquet"
LOOKUP_PATH = ROOT / "static" / "gw_point_lookup.parquet"
MODEL_PATH = ROOT / "models" / "groundwater" / "rf_gwe_final.pkl"
CONFIG_PATH = ROOT / "models" / "groundwater" / "feature_config.json"
FARMS_PATH = ROOT / "farm_inputs" / "farms.csv"
OUT_PATH = ROOT / "farm_inputs" / "gwl_today.csv"

# Put the exact parquet column name here to skip auto-detection.
COLUMN_OVERRIDES = {"lat": None, "lon": None, "GroundElevation_amsl": None, "TP": None,
                    "Climate": None, "Geology": None, "LULC": None, "date": None}

ALIASES = {
    "lat": ["Latitude", "lat", "latitude", "LAT"],
    "lon": ["Longitude", "lon", "longitude", "LON"],
    "GroundElevation_amsl": ["GroundElevation_amsl", "ground_elevation_amsl", "Elevation", "elevation"],
    "TP": ["TP", "tp"],
    "Climate": ["Climate", "climate"],
    "Geology": ["Geology", "geology"],
    "LULC": ["LULC", "lulc"],
    "date": ["DateModified", "date", "Date"],
}
STATIC_FEATS = ["GroundElevation_amsl", "TP", "Climate", "Geology", "LULC"]
SNAP_WARN_M, SNAP_REFUSE_M = 300.0, 1000.0
DEPTH_MIN_M, DEPTH_MAX_M = 0.0, 100.0


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- #
# COLUMN DETECTION
# ----------------------------------------------------------------------------- #
def detect_columns(all_cols, year):
    found, notes = {}, []
    lower = {c.lower(): c for c in all_cols}
    for key, names in ALIASES.items():
        if COLUMN_OVERRIDES.get(key):
            found[key] = COLUMN_OVERRIDES[key]
            continue
        hit = next((lower[n.lower()] for n in names if n.lower() in lower), None)
        if hit is None and key == "LULC":
            yrs = {}
            for c in all_cols:
                m = re.fullmatch(r"LULC_?(\d{4})", c, re.I)
                if m:
                    yrs[int(m.group(1))] = c
            if yrs:
                y = year if year in yrs else max(yrs)
                hit = yrs[y]
                notes.append(f"LULC column '{hit}' used (year {y}; available: {sorted(yrs)})")
        if hit is None and key == "TP":
            notes.append("No TP column in the parquet -> TP must be given with --tp or --tp-file")
        if hit is None and key not in ("date", "TP"):
            raise SystemExit(f"Could not find a column for '{key}'. Run --list-columns and set "
                             f"COLUMN_OVERRIDES['{key}'] at the top of this script.")
        if hit:
            found[key] = hit
    return found, notes


# ----------------------------------------------------------------------------- #
# LOOKUP (built once from the big parquet)
# ----------------------------------------------------------------------------- #
def build_lookup(src, out, year):
    if not Path(src).exists():
        raise SystemExit(f"GWE-inputs parquet not found: {src}")
    dset = ds.dataset(src, format="parquet")
    cols, notes = detect_columns(dset.schema.names, year)
    for n in notes:
        log(" ", n)
    log("Using columns:", cols)
    df = dset.to_table(columns=list(cols.values())).to_pandas()
    df = df.rename(columns={v: k for k, v in cols.items()})
    log(f"Loaded {len(df):,} rows.")

    feats_here = [c for c in STATIC_FEATS if c in df.columns]
    for c in ["lat", "lon"] + feats_here:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("Climate", "Geology", "LULC"):
        if df[c].notna().sum() == 0:
            raise SystemExit(f"Column '{c}' has no numeric values. It must already hold the numeric "
                             f"training codes (same codebook as training).")

    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        log(f"Date column found: {d.min()} .. {d.max()}  ({d.nunique()} distinct dates)")
        if d.nunique() > 1:
            log("  Several dates per point: keeping the LATEST row of each point "
                "(TP is therefore the latest stored value; use --tp to override).")
            df = df.assign(_d=d).sort_values("_d").drop_duplicates(["lat", "lon"], keep="last") \
                   .drop(columns="_d")
        df = df.drop(columns="date")

    n0 = len(df)
    df = df.dropna(subset=["lat", "lon"])
    log(f"Rows without lat/lon dropped: {n0 - len(df):,}")
    for c in feats_here:
        log(f"  {c}: {df[c].isna().mean() * 100:.1f}% missing, range {df[c].min():.3g} .. {df[c].max():.3g}")
    df = df.sort_values(["lat", "lon"]).reset_index(drop=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df[["lat", "lon"] + feats_here].to_parquet(out, index=False)
    log(f"Saved {out}  ({Path(out).stat().st_size / 1e6:.1f} MB)")


def nearest_points(lookup, lats, lons):
    """Nearest lookup row per farm (lookup is sorted by lat). Returns row index and distance in m."""
    lat_arr = lookup["lat"].to_numpy()
    lon_arr = lookup["lon"].to_numpy()
    idx = np.full(len(lats), -1, dtype=np.int64)
    dist = np.full(len(lats), np.nan)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        for half in (0.01, 0.05, 0.2):
            a, b = np.searchsorted(lat_arr, [la - half, la + half])
            if b > a:
                dy = (lat_arr[a:b] - la) * 111_320.0
                dx = (lon_arr[a:b] - lo) * 111_320.0 * math.cos(math.radians(la))
                d = np.hypot(dx, dy)
                j = int(np.argmin(d))
                if d[j] <= half * 111_320.0:       # inside the search window -> trustworthy
                    idx[i], dist[i] = a + j, d[j]
                    break
    return idx, dist


# ----------------------------------------------------------------------------- #
# PREDICTION
# ----------------------------------------------------------------------------- #
def load_model(path):
    import joblib
    model = joblib.load(path)
    try:
        model.set_params(device="cpu")      # same as 04_lag_predict.py (harmless if unsupported)
    except Exception:
        pass
    return model


def read_lean(path, float32=(), int16=()):
    """Load a parquet into a DataFrame using compact dtypes, one column at a time (low peak memory).
    lat/lon stay float64 (needed for exact nearest-point search); the listed columns are shrunk."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = {}
    for name in pf.schema_arrow.names:
        col = pf.read(columns=[name]).column(0)
        if name in float32 and pa.types.is_floating(col.type):
            col = col.cast(pa.float32())
        elif name in int16 and pa.types.is_integer(col.type) and col.null_count == 0 and len(col):
            mn_mx = pc.min_max(col)
            if mn_mx["min"].as_py() >= -32768 and mn_mx["max"].as_py() <= 32767:
                col = col.cast(pa.int16())
        cols[name] = col.to_numpy()
        del col
    return pd.DataFrame(cols, copy=False)


class GwPredictor:
    """For the dashboard: loads the lookup, model and config ONCE, then predicts per request.
    (The command-line flow in main() below is unchanged.)"""

    def __init__(self, gwe_mode="elevation", model_path=MODEL_PATH, config_path=CONFIG_PATH,
                 lookup_path=LOOKUP_PATH):
        for pth, hint in ((model_path, "put rf_gwe_final.pkl in models/groundwater/"),
                          (config_path, "put feature_config.json in models/groundwater/"),
                          (lookup_path, "run  python scripts/07_gw_predict.py --date YYYY-MM-DD  once "
                                        "to build static/gw_point_lookup.parquet")):
            if not Path(pth).exists():
                raise FileNotFoundError(f"Not found: {pth} -> {hint}")
        self.cfg = json.load(open(config_path, encoding="utf-8"))
        self.feats = self.cfg["model_features_in_order"]
        self.model = load_model(model_path)
        n_exp = getattr(self.model, "n_features_in_", len(self.feats))
        if n_exp != len(self.feats):
            raise ValueError(f"Model expects {n_exp} inputs but the config lists {len(self.feats)}.")
        self.lookup = read_lean(lookup_path, float32=("GroundElevation_amsl",),
                                int16=("Climate", "Geology", "LULC"))
        self.gwe_mode = gwe_mode

    def predict(self, lat, lon, when, tp):
        """when: datetime.date (IST). tp: rainfall in the unit the model was trained on.
        Returns a dict; gwl_depth_m is None when no trustworthy prediction is possible."""
        out = {"gwl_depth_m": None, "gwe_pred": None, "snap_m": None,
               "ground_elevation_m": None, "ground_elevation_amsl": None, "inputs": {}, "warnings": []}
        if tp is None or tp != tp:
            out["warnings"].append("no rainfall (TP) value -> no groundwater prediction")
            return out
        idx, dist = nearest_points(self.lookup, [float(lat)], [float(lon)])
        if idx[0] < 0 or dist[0] > SNAP_REFUSE_M:
            out["warnings"].append("no groundwater-input point within 1 km -> no prediction")
            return out
        out["snap_m"] = round(float(dist[0]), 0)
        if dist[0] > SNAP_WARN_M:
            out["warnings"].append(f"nearest input point is {dist[0]:.0f} m away")
        row = self.lookup.iloc[idx[0]]
        doy = when.timetuple().tm_yday
        vals = {"Latitude": float(lat), "Longitude": float(lon),
                "GroundElevation_amsl": float(row["GroundElevation_amsl"]), "TP": float(tp),
                "Climate": float(row["Climate"]), "Geology": float(row["Geology"]),
                "LULC": float(row["LULC"]), "Year": float(when.year),
                "sin_day": math.sin(2 * math.pi * doy / 365.25),
                "cos_day": math.cos(2 * math.pi * doy / 365.25)}
        out["inputs"] = {k: round(v, 4) for k, v in vals.items()}
        X = pd.DataFrame([vals])[self.feats]
        if X.isna().any(axis=None):
            miss = [c for c in self.feats if pd.isna(X.at[0, c])]
            out["warnings"].append(f"missing input(s) {miss} at the nearest point -> no prediction")
            return out
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)      # 'X does not have valid feature names'
            pred = float(self.model.predict(X.to_numpy(dtype=float))[0])
        out["gwe_pred"] = round(pred, 3)
        elev = vals["GroundElevation_amsl"]
        out["ground_elevation_m"] = round(elev, 1)
        out["ground_elevation_amsl"] = round(elev, 2)      # GSE for Decision 3 (m above mean sea level)
        depth = (elev - pred) if self.gwe_mode == "elevation" else pred
        if not (DEPTH_MIN_M <= depth <= DEPTH_MAX_M):
            out["warnings"].append(f"implausible depth {depth:.1f} m (check GWE mode / TP) -> ignored")
            return out
        out["gwl_depth_m"] = round(depth, 2)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date_cls.today().isoformat())
    ap.add_argument("--farms", default=str(FARMS_PATH))
    ap.add_argument("--src", default=str(SRC_PARQUET))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--gwe-mode", choices=["elevation", "depth"], default="elevation")
    ap.add_argument("--tp", type=float, default=None, help="TP for ALL farms (one value)")
    ap.add_argument("--tp-file", default=None, help="CSV with columns farm_id,TP (one value per farm)")
    ap.add_argument("--rebuild-lookup", action="store_true")
    ap.add_argument("--build-lookup-only", action="store_true",
                    help="only build static/gw_point_lookup.parquet (needed once by the dashboard) and stop")
    ap.add_argument("--list-columns", action="store_true")
    a = ap.parse_args()

    when = datetime.strptime(a.date, "%Y-%m-%d").date()

    if a.list_columns:
        sch = ds.dataset(a.src, format="parquet").schema
        log(f"{a.src}\n{len(sch.names)} columns:")
        for n, t in zip(sch.names, sch.types):
            log(f"  {n}  [{t}]")
        return

    if a.build_lookup_only:
        if LOOKUP_PATH.exists() and not a.rebuild_lookup:
            log(f"{LOOKUP_PATH} already exists. Add --rebuild-lookup to build it again.")
            return
        log("Building the small lookup from the big parquet (one time) ...")
        build_lookup(a.src, LOOKUP_PATH, when.year)
        return

    if a.rebuild_lookup or not LOOKUP_PATH.exists():
        log("Building the small lookup from the big parquet (one time) ...")
        build_lookup(a.src, LOOKUP_PATH, when.year)

    for p, hint in ((MODEL_PATH, "put rf_gwe_final.pkl in models/groundwater/"),
                    (CONFIG_PATH, "put feature_config.json in models/groundwater/"),
                    (Path(a.farms), "create farm_inputs/farms.csv (farm_id, lat, lon)")):
        if not Path(p).exists():
            raise SystemExit(f"Not found: {p}\n -> {hint}")

    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    feats = cfg["model_features_in_order"]
    farms = pd.read_csv(a.farms)
    for c in ("farm_id", "lat", "lon"):
        if c not in farms.columns:
            raise SystemExit(f"farms file needs a '{c}' column")

    lookup = pd.read_parquet(LOOKUP_PATH)
    idx, dist = nearest_points(lookup, farms["lat"].to_numpy(float), farms["lon"].to_numpy(float))

    if "TP" not in lookup.columns and a.tp is None and a.tp_file is None:
        raise SystemExit(
            "TP is not in the parquet, and none was given.\n"
            "The groundwater model needs TP (same definition/units/period as in training).\n"
            "Give it with  --tp VALUE  (one value for all farms)  or  --tp-file file.csv  "
            "(columns: farm_id,TP).")
    tp_map = None
    if a.tp_file:
        t = pd.read_csv(a.tp_file)
        tp_map = dict(zip(t["farm_id"], pd.to_numeric(t["TP"], errors="coerce")))

    doy = when.timetuple().tm_yday
    rows = []
    for i in range(len(farms)):
        r = {"Latitude": farms.at[i, "lat"], "Longitude": farms.at[i, "lon"],
             "Year": when.year,
             "sin_day": math.sin(2 * math.pi * doy / 365.25),
             "cos_day": math.cos(2 * math.pi * doy / 365.25)}
        if idx[i] >= 0:
            for c in STATIC_FEATS:
                r[c] = lookup.at[idx[i], c] if c in lookup.columns else np.nan
        else:
            for c in STATIC_FEATS:
                r[c] = np.nan
        if tp_map is not None:
            r["TP"] = tp_map.get(farms.at[i, "farm_id"], np.nan)
        if a.tp is not None:
            r["TP"] = a.tp
        rows.append(r)
    X = pd.DataFrame(rows)[feats]          # exact training order
    ok = X.notna().all(axis=1).to_numpy() & (idx >= 0) & (dist <= SNAP_REFUSE_M)

    model = load_model(MODEL_PATH)
    n_exp = getattr(model, "n_features_in_", len(feats))
    if n_exp != len(feats):
        raise SystemExit(f"Model expects {n_exp} inputs but the config lists {len(feats)}.")

    pred = np.full(len(farms), np.nan)
    if ok.any():
        pred[ok] = model.predict(X.loc[ok].to_numpy(dtype=float))

    elev = X["GroundElevation_amsl"].to_numpy(float)
    depth = (elev - pred) if a.gwe_mode == "elevation" else pred.copy()

    warns = []
    for i in range(len(farms)):
        w = []
        if idx[i] < 0 or dist[i] > SNAP_REFUSE_M:
            w.append("no parquet point within 1 km -> no groundwater prediction")
        elif not ok[i]:
            w.append(f"missing input(s) {[f for f in feats if pd.isna(X.at[i, f])]} at nearest point")
        else:
            if dist[i] > SNAP_WARN_M:
                w.append(f"nearest point {dist[i]:.0f} m away")
            if not (DEPTH_MIN_M <= depth[i] <= DEPTH_MAX_M):
                w.append(f"implausible depth {depth[i]:.1f} m (check --gwe-mode / TP) -> ignored")
                depth[i] = np.nan
        warns.append("; ".join(w))

    res = pd.DataFrame({"farm_id": farms["farm_id"], "gwl_depth_m": np.round(depth, 2),
                        "gwe_pred": np.round(pred, 3),
                        "ground_elevation_amsl": np.round(elev, 2),
                        "snap_m": np.round(dist, 0),
                        "warnings": warns})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    log(res.to_string(index=False))
    other = "depth" if a.gwe_mode == "elevation" else "elevation"
    log(f"\nMode used: GWE is {a.gwe_mode}. Depth to water should be roughly 1-40 m; "
        f"if it is not, re-run with --gwe-mode {other}.")
    log(f"Saved {a.out}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)
