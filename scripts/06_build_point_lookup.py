"""
================================================================================
06_build_point_lookup.py   -   run ONCE (re-run only if the source parquet changes)

Builds a lean lookup table from the big static parquet so the daily irrigation
script never has to open the 27-million-row file:

    static/point_lookup.parquet
        columns: lon, lat, sand, silt, clay (0 cm, in %), slope_pct (%), geology (int code)
        sorted by lat, then lon (needed for fast nearest-point search)

NEW in this version
  * slope_pct is computed from GroundElevation_amsl on the regular lat/lon grid (see slope_grid.py).
    It is computed on the FULL grid BEFORE any row is dropped, so water / no-texture cells still help
    their neighbours. A grid report is printed: check that the spacing is regular and the grid complete.
  * geology = the Geology code of the point (matches the Class column of GeologyClass_Ksat.xlsx).
  * the parquet may hold several dates per point; the points are de-duplicated by lat/lon first.
  * the source file is config.INPUT_PARQUET (static/source/input.parquet).

WHAT YOU MAY NEED TO EDIT:
  - COLUMN_OVERRIDES   if the auto-detection below guesses the wrong column names

USAGE:
    python scripts/06_build_point_lookup.py
    python scripts/06_build_point_lookup.py --src "F:\\...\\input.parquet"
    python scripts/06_build_point_lookup.py --list-columns     # just print candidate columns
================================================================================
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import config                                  # noqa: E402
from slope_grid import slope_pct_from_points   # noqa: E402

# ----------------------------------------------------------------------------- #
# CONFIG
# ----------------------------------------------------------------------------- #
SRC_PARQUET = Path(config.INPUT_PARQUET)
OUT_PARQUET = Path(config.POINT_LOOKUP_PATH)

# Fill any of these with the exact column name to skip auto-detection.
COLUMN_OVERRIDES = {"lon": None, "lat": None,
                    "sand": "sand_pct_0cm", "silt": "silt_pct_0cm", "clay": "clay_pct_0cm",
                    "elev": None, "geology": None}
ALIASES = {"lon": ["lon", "Longitude", "longitude", "LON"],
           "lat": ["lat", "Latitude", "latitude", "LAT"],
           "elev": ["GroundElevation_amsl", "ground_elevation_amsl", "Elevation", "elevation"],
           "geology": ["Geology", "geology"]}


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- #
# COLUMN DETECTION
# ----------------------------------------------------------------------------- #
def _candidates(columns, word):
    """Columns whose name contains `word` (case-insensitive)."""
    return [c for c in columns if word in c.lower()]


def _pick_surface(cands, word):
    """From several candidate columns pick the 0 cm / surface one, if it is clear."""
    if len(cands) == 1:
        return cands[0]
    pat = re.compile(r"(^|[^0-9])0(cm|_|-|$)|top|surf", re.I)
    hits = [c for c in cands if pat.search(c)]
    if len(hits) == 1:
        return hits[0]
    return None


def detect_columns(all_cols):
    found = {}
    lower = {c.lower(): c for c in all_cols}
    for key in ("lon", "lat", "elev", "geology"):
        if COLUMN_OVERRIDES[key]:
            found[key] = COLUMN_OVERRIDES[key]
            continue
        hit = next((lower[n.lower()] for n in ALIASES[key] if n.lower() in lower), None)
        if hit is None:
            raise SystemExit(f"Could not find a '{key}' column (tried {ALIASES[key]}). "
                             f"Set COLUMN_OVERRIDES['{key}'] at the top of this script.")
        found[key] = hit

    for key in ("sand", "silt", "clay"):
        if COLUMN_OVERRIDES[key]:
            found[key] = COLUMN_OVERRIDES[key]
            continue
        cands = _candidates(all_cols, key)
        chosen = _pick_surface(cands, key)
        if chosen is None and key == "sand":
            found[key] = None          # sand can be derived as 100 - silt - clay
            continue
        if chosen is None:
            raise SystemExit(
                f"Could not decide which '{key}' column is the 0 cm layer.\n"
                f"Candidates found: {cands}\n"
                f"Set COLUMN_OVERRIDES['{key}'] to the right one and re-run.")
        found[key] = chosen
    return found


# ----------------------------------------------------------------------------- #
# MAIN
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_PARQUET))
    ap.add_argument("--out", default=str(OUT_PARQUET))
    ap.add_argument("--list-columns", action="store_true",
                    help="print columns that look like sand/silt/clay/lon/lat and exit")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Source parquet not found: {src}\nEdit INPUT_PARQUET in config.py or pass --src.")

    pf = pq.ParquetFile(src)
    all_cols = pf.schema.names
    log(f"Source: {src}  |  rows: {pf.metadata.num_rows:,}  |  columns: {len(all_cols)}")

    if args.list_columns:
        for w in ("lon", "lat", "sand", "silt", "clay", "tex", "elev", "geolog"):
            log(f"  '{w}' -> {_candidates(all_cols, w)}")
        return

    cols = detect_columns(all_cols)
    log("Using columns:", {k: v for k, v in cols.items()})

    read_cols = [v for v in cols.values() if v]
    df = pd.read_parquet(src, columns=read_cols)
    df = df.rename(columns={v: k for k, v in cols.items() if v})
    log(f"Loaded {len(df):,} rows.")

    for c in ("lon", "lat", "sand", "silt", "clay", "elev", "geology"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # one row per point (the parquet may repeat each point for several dates)
    n_all = len(df)
    df = df.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["lat", "lon"], keep="last")
    log(f"{n_all:,} rows -> {len(df):,} distinct points")
    df = df.reset_index(drop=True)

    # ---- slope on the FULL grid, before any row is dropped ----
    log("Computing slope_pct from GroundElevation_amsl ...")
    slope, rep = slope_pct_from_points(df["lat"].to_numpy(), df["lon"].to_numpy(), df["elev"].to_numpy())
    df["slope_pct"] = slope
    log("GRID REPORT (check that this looks like a regular ~70 m grid):")
    log(f"  grid {rep['n_rows']:,} rows x {rep['n_cols']:,} cols, {rep['grid_filled_pct']:.1f}% of cells have a point")
    log(f"  cell size {rep['dy_m']:.1f} m (north-south) x {rep['dx_m_mid_lat']:.1f} m (east-west)  "
        f"[dlat {rep['dlat_deg']:.7f} deg, dlon {rep['dlon_deg']:.7f} deg]")
    log(f"  worst offset of a point from its grid cell: {rep['worst_offset_fraction_of_cell']:.3f} of a cell "
        f"(should be close to 0)")
    log(f"  cells holding 2+ points: {rep['cells_with_2plus_points']:,} (should be 0)")
    log(f"  slope left empty for {rep['n_slope_missing']:,} points")
    if "slope_pct_p50_p95_max" in rep:
        p50, p95, mx = rep["slope_pct_p50_p95_max"]
        log(f"  slope_pct median {p50:.2f}, 95th percentile {p95:.2f}, max {mx:.2f}")

    # ---- texture ----
    if "sand" not in df.columns:
        log("No sand column - deriving sand = 100 - silt - clay")
        df["sand"] = 100.0 - df["silt"] - df["clay"]

    top = float(np.nanmax(df[["sand", "silt", "clay"]].to_numpy()))
    if top <= 1.0:
        log("Texture values look like fractions (max <= 1) -> converting to percent (x100)")
        df[["sand", "silt", "clay"]] *= 100.0

    # keep only valid rows; nearest-point search should only see usable points
    n0 = len(df)
    df = df.dropna(subset=["lon", "lat", "sand", "silt", "clay"])
    df = df[(df[["sand", "silt", "clay"]] >= 0).all(axis=1)]
    n1 = len(df)
    # points with no texture information (sand+silt+clay ~ 0, e.g. water/urban cells) must not
    # be picked as the "nearest point" for a farm
    df = df[(df["sand"] + df["silt"] + df["clay"]) >= 50]
    log(f"Dropped {n1 - len(df):,} rows whose sand+silt+clay is below 50% (no usable texture)")
    log(f"Dropped {n0 - len(df):,} rows in total -> {len(df):,} usable points")

    tot = (df["sand"] + df["silt"] + df["clay"])
    bad = ((tot < 90) | (tot > 110)).mean()
    log(f"Sand+silt+clay between 90-110% for {100 * (1 - bad):.1f}% of points "
        f"(the rest will be flagged by the advice script)")

    # geology code as a nullable integer (empty stays empty)
    df["geology"] = df["geology"].round().astype("Int32")
    codes = df["geology"].value_counts(dropna=False).head(12)
    log("Geology codes in the lookup (code: points):  " + ", ".join(f"{k}: {v:,}" for k, v in codes.items()))

    df = df.sort_values(["lat", "lon"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["lon", "lat", "sand", "silt", "clay", "slope_pct", "geology"]].to_parquet(out, index=False)
    log(f"Saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # keep the message readable for a non-developer run
        log(f"FAILED: {e}")
        sys.exit(1)
