"""
test_latest_run.py  -  a one-off TEST on the newest data, start to finish.

    python scripts/test_latest_run.py

What it does:
  1. Finds the latest SMAP date D0 and downloads it plus the 3 days before it
     (4 SMAP files: D0-3, D0-2, D0-1, D0).
  2. Gets rainfall + temperature for the same 4 days (same method as step 02:
     rainfall.nic.in first, Open-Meteo backup for past days the site refuses).
  3. Runs the lag model with anchor D0-3 (rainfall/temperature D0-3 .. D0), which
     predicts SMAP for D0+1, then downscales it to 70 m.
     -> 03_downscaled_sm/sm_70m_<D0+1>.tif

  Note: the lag model uses the SMAP value of the ANCHOR day only (D0-3). The other
  three SMAP files are downloaded for the pattern you asked for but are not model inputs.

Optional real check:
    python scripts/test_latest_run.py --validate
  also downloads D0-4 and predicts D0 from it, then compares the prediction with the
  REAL SMAP of D0 (RMSE, bias, correlation) and with the simplest guess, "tomorrow
  looks like the anchor day" (persistence). A useful model should beat persistence.

Other options:
    --latest-date 2026-10-05   treat this as the latest SMAP date (searches backwards from it)
    --skip-download            use SMAP files already on disk (latest one found there)
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
import config
import h5_to_tiff

import numpy as np
import pandas as pd

BACK = 3                                   # days before D0 -> 4 SMAP files in total
SMAP_BASE = ("https://cmr.earthdata.nasa.gov/virtual-directory/collections/"
             "C3480440639-NSIDC_CPRD/temporal")
SMAP_TIME = "T090000"


def load_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def smap_files_on_disk():
    out = {}
    for f in sorted(os.listdir(config.SMAP_RAW_DIR)):
        if f.endswith(".h5"):
            d = h5_to_tiff.extract_date(f)
            if d:
                out[date.fromisoformat(d)] = os.path.join(config.SMAP_RAW_DIR, f)
    return out


# ------------------------------------------------------------------ 1. SMAP
def download_smap(dl, latest_arg, n_back):
    session = dl.get_session()

    def granule(d):
        return dl.find_target_h5_recursive(session, f"{SMAP_BASE}/{d.year}/{d.month:02d}/{d.day:02d}/", SMAP_TIME)

    start = date.fromisoformat(latest_arg) if latest_arg else date.today()
    print(f"Searching backwards from {start} for the latest SMAP granule ...")
    D0, link0 = None, None
    for i in range(config.MAX_LOOKBACK_DAYS):
        d = start - timedelta(days=i)
        print(f"  checking {d} ...")
        link0 = granule(d)
        if link0:
            D0 = d
            break
    if D0 is None:
        sys.exit("No SMAP granule found. Check your Earthdata login (credentials.env) and internet.")
    print(f"Latest SMAP date D0 = {D0}")

    files = {D0: dl.download_file(session, link0, config.SMAP_RAW_DIR)}
    for k in range(1, n_back + 1):
        d = D0 - timedelta(days=k)
        link = granule(d)
        files[d] = dl.download_file(session, link, config.SMAP_RAW_DIR) if link else None
        if not link:
            print(f"  no SMAP granule for {d} (gap)")
    return D0, files


# ------------------------------------------------------------ 2. rain + temp
def get_rain_temp(sc, dates):
    allow = getattr(config, "RAIN_FALLBACK_OPENMETEO", True)
    today = date.today()
    how = {}
    for d in sorted(dates):
        iso = d.isoformat()
        if d > today:
            how[d] = "in the future, skipped"
            continue
        if sc.have_day(iso) and d != today:
            how[d] = "already on disk"
            continue
        print(f"\n{'=' * 60}\nRainfall + temperature for {iso}\n{'=' * 60}")
        r = sc.fetch_day(iso, allow)
        how[d] = {"site": "rainfall.nic.in", "openmeteo": "Open-Meteo BACKUP (not station data)",
                  None: "NOT AVAILABLE"}[r]
    return how


# --------------------------------------------------------------- 3. models
def run_script(name, *args):
    cmd = [sys.executable, os.path.join(HERE, name), *args]
    print(f"\n{'=' * 60}\n>> {name} {' '.join(args)}\n{'=' * 60}", flush=True)
    return subprocess.run(cmd, cwd=os.path.dirname(HERE)).returncode


def lag_run(anchor, with_downscale):
    for name, args in [("03_build_lag_features.py", ("--anchor", anchor.isoformat())),
                       ("04_lag_predict.py", ()),
                       ("05_downscale_to_70m.py", ())]:
        if name.startswith("05") and not with_downscale:
            break
        if run_script(name, *args) != 0:
            print(f"\nStopped: {name} failed (see the message above).")
            return False
    return True


# -------------------------------------------------------------- validation
def validate(D0, files, anchor):
    meta = json.load(open(os.path.join(config.LAG_DIR, "latest_lag_run.json")))
    pred = pd.read_parquet(meta["lag_pred_file"]).set_index("cell_id")["sm_pred_lag"]
    truth = h5_to_tiff.read_smap_grid(files[D0])
    base = h5_to_tiff.read_smap_grid(files[anchor])
    ids = pred.index.to_numpy(np.int64)
    r = ids // truth["ncols_full"] - truth["r0"]
    c = ids % truth["ncols_full"] - truth["c0"]
    inside = (r >= 0) & (r < truth["sm"].shape[0]) & (c >= 0) & (c < truth["sm"].shape[1])
    ids, r, c = ids[inside], r[inside], c[inside]
    p = pred.to_numpy()[inside]
    actual, persist = truth["sm"][r, c], base["sm"][r, c]
    ok = np.isfinite(actual) & np.isfinite(persist)
    p, actual, persist = p[ok], actual[ok], persist[ok]
    if len(p) < 10:
        print("Too few cells to validate.")
        return
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    m, pr = rmse(p, actual), rmse(persist, actual)
    lines = [
        f"VALIDATION: predicted SMAP for {D0} from anchor {anchor}, compared with the real SMAP of {D0}",
        f"  cells compared              : {len(p):,}",
        f"  model RMSE                  : {m:.4f} m3/m3",
        f"  persistence RMSE (baseline) : {pr:.4f} m3/m3   ('{D0} looks like {anchor}')",
        f"  model bias (pred - real)    : {float(np.mean(p - actual)):+.4f}",
        f"  correlation (model vs real) : {float(np.corrcoef(p, actual)[0, 1]):.3f}",
        "  VERDICT: model beats the persistence baseline." if m < pr else
        "  VERDICT: model does NOT beat the persistence baseline on this case.",
    ]
    print("\n" + "\n".join(lines))
    with open(os.path.join(config.LAG_DIR, f"validation_{D0}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Test the pipeline on the newest SMAP data.")
    ap.add_argument("--latest-date", default=None)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args(argv)

    dl = load_script("01_download_smap.py", "dl01")
    sc = load_script("02_scrape_rainfall_temp.py", "sc02")
    n_back = BACK + (1 if a.validate else 0)

    # 1. SMAP
    if a.skip_download:
        on_disk = smap_files_on_disk()
        if not on_disk:
            sys.exit("No SMAP files on disk; run without --skip-download.")
        D0 = date.fromisoformat(a.latest_date) if a.latest_date else max(on_disk)
        files = {D0 - timedelta(days=k): on_disk.get(D0 - timedelta(days=k)) for k in range(n_back + 1)}
    else:
        D0, files = download_smap(dl, a.latest_date, n_back)

    anchor = D0 - timedelta(days=BACK)
    smap_days = [D0 - timedelta(days=k) for k in range(BACK, -1, -1)]
    missing = [d for d in smap_days if not files.get(d)]
    if not files.get(D0) or not files.get(anchor):
        sys.exit(f"Need SMAP files for {anchor} and {D0}; missing: {[str(d) for d in missing]}")

    # 2. rainfall + temperature for the same days
    rain_days = set(smap_days)
    if a.validate:
        rain_days.add(D0 - timedelta(days=BACK + 1))
    how = get_rain_temp(sc, rain_days)

    # 3. optional validation run first (so the final run is the "latest" one)
    if a.validate:
        v_anchor = D0 - timedelta(days=BACK + 1)
        if files.get(v_anchor):
            print(f"\n##### VALIDATION RUN: anchor {v_anchor} -> predicts {D0} #####")
            if lag_run(v_anchor, with_downscale=False):
                validate(D0, files, v_anchor)
        else:
            print(f"\nNo SMAP file for {v_anchor}; skipping validation.")

    # 4. the main test run
    target = anchor + timedelta(days=4)
    print(f"\n##### MAIN RUN: anchor {anchor} -> target {target} #####")
    ok = lag_run(anchor, with_downscale=True)

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    print("SMAP files        :", ", ".join(f"{d}{'' if files.get(d) else ' (MISSING)'}" for d in smap_days))
    for d, h in sorted(how.items()):
        print(f"rain+temp {d}: {h}")
    print(f"Anchor / target   : {anchor} -> {target}")
    out = os.path.join(config.DOWNSCALED_DIR, f"sm_70m_{target}.tif")
    if ok and os.path.exists(out):
        print(f"FINAL 70 m MAP    : {out}")
        print(f"QA map            : {out.replace('.tif', '_QA.tif')}")
    else:
        print("The final map was not produced; see the messages above.")


if __name__ == "__main__":
    main()
