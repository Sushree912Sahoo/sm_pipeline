"""
03_build_lag_features.py
--------------------------
Builds the 11 lag-model features on the 9 km SMAP grid.

Same feature definitions as your earlier 03 script:
  - sm_4_prior            = measured SMAP soil moisture on anchor date D
  - sum_rainfall_1..4     = rainfall summed over D+3 / D+3..D+2 / ... / D+3..D
  - mean_temp_1..4        = the matching temperature means
  - doy_sin, doy_cos      = day of year of the TARGET date T = D + 4

The anchor D is chosen so that the TARGET DAY T = D + 4 IS TODAY (IST):
D = today - 4. The model was trained with exactly this 4-day gap, and the rain and
temperature days D..D+3 are all complete past days (today's partial rain is not used).
If there is no SMAP file or no rain/temperature for D = today - 4, the script falls back to the
newest OLDER anchor that works (target = an earlier day) and says so loudly. It never picks
an anchor whose target would be AFTER today (tomorrow's map).

Rainfall/temperature (station points) are joined to each SMAP cell by
nearest neighbour within config.NN_TOLERANCE_DEG, taking the nearest station
that has a valid value.

Outputs (in 02_lag_adjusted_sm/):
  lag_features_<T>.parquet   cell_id, lat, lon, the 11 features, dates
  latest_lag_run.json        anchor date, target date, SMAP file used

Usage:
    python scripts/03_build_lag_features.py
    python scripts/03_build_lag_features.py --anchor 2026-10-02
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import h5_to_tiff

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


IST = timezone(timedelta(hours=5, minutes=30))
LAG_GAP_DAYS = 4          # the model's fixed gap: SMAP on D  ->  prediction for D + 4


def today_ist():
    return datetime.now(IST).date()


def smap_files_by_date():
    out = {}
    for f in sorted(glob.glob(os.path.join(config.SMAP_RAW_DIR, "*.h5"))):
        d = h5_to_tiff.extract_date(f)
        if d:
            out[d] = f
    return out


def load_station_day(date_iso):
    """Rainfall + temperature points for one date, or None if either CSV is missing."""
    rp = os.path.join(config.RAINFALL_DIR, f"rainfall_{date_iso}.csv")
    tp = os.path.join(config.TEMPERATURE_DIR, f"temperature_{date_iso}.csv")
    if not (os.path.exists(rp) and os.path.exists(tp)):
        return None
    r = pd.read_csv(rp)
    t = pd.read_csv(tp)
    r["rainfall"] = pd.to_numeric(r["Rain (in mm.)"], errors="coerce")
    t["temperature"] = pd.to_numeric(t["temperature"], errors="coerce")
    r = r.rename(columns={"Latitude": "lat", "Longitude": "lon"})[["lat", "lon", "rainfall"]]
    t = t.rename(columns={"Latitude": "lat", "Longitude": "lon"})[["lat", "lon", "temperature"]]
    return r.dropna(subset=["lat", "lon"]), t.dropna(subset=["lat", "lon"])


def nn_values(cell_xy, points, col):
    """Nearest station value (stations with a valid value only) per cell.
    Within NN_TOLERANCE_DEG normally; if none, the nearest one up to NN_FALLBACK_DEG.
    Returns (values, n_within_tolerance, n_from_farther_station)."""
    pts = points.dropna(subset=[col])
    if pts.empty:
        return np.full(len(cell_xy), np.nan), 0, 0
    tree = cKDTree(pts[["lat", "lon"]].to_numpy())
    dist, idx = tree.query(cell_xy, k=1)
    vals = pts[col].to_numpy(float)[idx]
    near = dist <= config.NN_TOLERANCE_DEG
    far = (~near) & (dist <= config.NN_FALLBACK_DEG)
    vals[~(near | far)] = np.nan
    return vals, int(near.sum()), int(far.sum())


def fill_invalid_smap(sm, max_cells):
    """Fill NaN cells of the SMAP crop from the nearest valid cell (up to max_cells away).
    Returns (filled_sm, number_filled)."""
    bad = ~np.isfinite(sm)
    if max_cells <= 0 or not bad.any() or bad.all():
        return sm, 0
    gi, gj = np.indices(sm.shape)
    tree = cKDTree(np.column_stack([gi[~bad], gj[~bad]]))
    dist, idx = tree.query(np.column_stack([gi[bad], gj[bad]]))
    out = sm.copy()
    ok = dist <= max_cells
    vals = sm[~bad][idx]
    tmp = out[bad]
    tmp[ok] = vals[ok]
    out[bad] = tmp
    return out, int(ok.sum())


def choose_anchor(smap_dates, forced=None, today=None):
    """Best anchor D with target T = D + 4 not after today. Returns (anchor, window, report)."""
    report = []
    if forced:
        dates = [forced]
    else:
        today = today or today_ist()
        latest_ok = (pd.Timestamp(today) - timedelta(days=LAG_GAP_DAYS)).strftime("%Y-%m-%d")
        dates = sorted((d for d in smap_dates if d <= latest_ok), reverse=True)
        for d in sorted((d for d in smap_dates if d > latest_ok), reverse=True):
            report.append(f"{d}: skipped (its target day {(pd.Timestamp(d) + timedelta(days=LAG_GAP_DAYS)).date()} "
                          f"is after today {today})")
    for d in dates:
        if d not in smap_dates:
            report.append(f"{d}: no SMAP file")
            continue
        D = pd.Timestamp(d)
        window = [(D + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]
        missing = [w for w in window if load_station_day(w) is None]
        if not missing:
            return d, window, report
        report.append(f"{d}: missing rainfall/temperature for {missing}")
    return None, None, report


def run(forced_anchor=None, today=None):
    files = smap_files_by_date()
    if not files:
        sys.exit(f"No SMAP files in {config.SMAP_RAW_DIR}. Run 01_download_smap.py first.")

    anchor, window, report = choose_anchor(set(files), forced_anchor, today)
    if anchor is None:
        print("Cannot build features yet. For each SMAP date tried:")
        for line in report:
            print("  ", line)
        sys.exit("Need rainfall + temperature CSVs for all four days D..D+3 of some SMAP date. "
                 "Run 02_scrape_rainfall_temp.py (daily scraping builds this up).")

    D = pd.Timestamp(anchor)
    T = D + timedelta(days=4)
    today_d = pd.Timestamp(today or today_ist())
    print(f"Today (IST)        : {today_d.date()}")
    print(f"Anchor SMAP date D : {anchor}")
    print(f"Target date T=D+4  : {T.date()}")
    if T.normalize() == today_d.normalize():
        print("  -> the map is for TODAY.")
    elif T.normalize() < today_d.normalize():
        print(f"!! WARNING: the map is for {T.date()}, {(today_d - T.normalize()).days} day(s) BEFORE today: "
              f"the SMAP file or the rain/temperature for the anchor {(today_d - timedelta(days=4)).date()} "
              f"is missing. Tried:")
        for line in report:
            print("     ", line)
    else:
        print(f"!! WARNING: the map is for {T.date()}, AFTER today (anchor was forced).")
    print(f"Rain/temp window   : {window}")

    g = h5_to_tiff.read_smap_grid(files[anchor])
    n_invalid = int((~np.isfinite(g["sm"])).sum())
    sm, n_filled = fill_invalid_smap(g["sm"], config.SMAP_FILL_MAX_CELLS)
    lat, lon = g["lat"], g["lon"]
    print(f"SMAP crop: {n_invalid:,} invalid cell(s) on D, {n_filled:,} filled from the nearest valid cell "
          f"(max {config.SMAP_FILL_MAX_CELLS} cells away), {n_invalid - n_filled:,} left empty.")
    ii, jj = np.indices(sm.shape)
    cell_id = ((g["r0"] + ii) * g["ncols_full"] + (g["c0"] + jj)).ravel()
    sm_f, lat_f, lon_f = sm.ravel(), lat.ravel(), lon.ravel()

    keep = np.isfinite(sm_f)
    cell_id, sm_f, lat_f, lon_f = cell_id[keep], sm_f[keep], lat_f[keep], lon_f[keep]
    print(f"SMAP cells with a valid value on D: {len(sm_f):,}")
    cell_xy = np.column_stack([lat_f, lon_f])

    rain, temp = [], []
    for d in window:
        r_pts, t_pts = load_station_day(d)
        rv, r_near, r_far = nn_values(cell_xy, r_pts, "rainfall")
        tv, t_near, t_far = nn_values(cell_xy, t_pts, "temperature")
        print(f"  {d}: rainfall {r_near:,} within {config.NN_TOLERANCE_DEG} deg + {r_far:,} from a farther "
              f"station, {int((~np.isfinite(rv)).sum()):,} none | temperature {t_near:,} + {t_far:,}, "
              f"{int((~np.isfinite(tv)).sum()):,} none  (of {len(rv):,} cells)")
        rain.append(rv)
        temp.append(tv)
    rain, temp = np.column_stack(rain), np.column_stack(temp)   # oldest -> newest: D..D+3

    feat = pd.DataFrame({"cell_id": cell_id, "lat": lat_f, "lon": lon_f, "sm_4_prior": sm_f})
    feat["sum_rainfall_1"] = rain[:, 3]
    feat["sum_rainfall_2"] = rain[:, 3] + rain[:, 2]
    feat["sum_rainfall_3"] = rain[:, 3] + rain[:, 2] + rain[:, 1]
    feat["sum_rainfall_4"] = rain[:, 3] + rain[:, 2] + rain[:, 1] + rain[:, 0]
    feat["mean_temp_1"] = temp[:, 3]
    feat["mean_temp_2"] = (temp[:, 3] + temp[:, 2]) / 2
    feat["mean_temp_3"] = (temp[:, 3] + temp[:, 2] + temp[:, 1]) / 3
    feat["mean_temp_4"] = temp.sum(axis=1) / 4
    doy = T.dayofyear
    feat["doy_sin"] = np.sin(2 * np.pi * doy / config.DOY_PERIOD)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / config.DOY_PERIOD)
    feat["date"] = T
    feat["anchor_smap_date"] = D

    complete = feat.dropna(subset=config.FEATURE_COLUMNS).reset_index(drop=True)
    print(f"\n{len(complete):,} of {len(feat):,} cells have all 11 features.")
    lost = feat[feat[config.FEATURE_COLUMNS].isna().any(axis=1)]
    if len(lost):
        why = lost[config.FEATURE_COLUMNS].isna().sum()
        print(f"  {len(lost):,} cell(s) still incomplete; missing counts per feature:")
        print("   ", ", ".join(f"{k}={int(v)}" for k, v in why.items() if v))
        print("  (these are usually far from every station, or outside Odisha)")
    if complete.empty:
        sys.exit("No cell has a complete feature row. Check the station coordinates and "
                 "config.NN_TOLERANCE_DEG.")

    os.makedirs(config.LAG_DIR, exist_ok=True)
    out = os.path.join(config.LAG_DIR, f"lag_features_{T.date()}.parquet")
    complete.to_parquet(out, index=False)
    with open(os.path.join(config.LAG_DIR, "latest_lag_run.json"), "w") as f:
        json.dump(dict(anchor_date=anchor, target_date=str(T.date()),
                       anchor_file=files[anchor], features_file=out,
                       ncols_full=g["ncols_full"],
                       n_cells=int(len(complete))), f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=None, help="force anchor SMAP date YYYY-MM-DD")
    ap.add_argument("--today", default=None, help="pretend today is YYYY-MM-DD (for testing)")
    a = ap.parse_args()
    run(a.anchor, a.today)
