"""
================================================================================
08_irrigation_advice.py   -   daily irrigation advice for each farm (THREE decisions)

For every farm in farm_inputs/farms.csv, after today's 70 m soil-moisture GeoTIFF exists:
    Decision 1   pump ON / OFF                        (SM vs theta_trigger)
    Decision 2   irrigation depth                     (d_net, d_gross with Ea)
    Decision 3   pump capacity and running time       (well discharge Q, then flood / furrow / sprinkler)

This file only reads files and writes results. ALL formulas are in irrigation_core.py and all
lookups in irrigation_data.py, so the dashboard gives exactly the same numbers.

INPUTS
    03_downscaled_sm/sm_70m_<DATE>.tif     daily 70 m soil moisture (m3/m3)         [step 05]
    static/point_lookup.parquet            sand, silt, clay, slope_pct, geology     [step 06]
    farm_inputs/gwl_today.csv  (--gwl-file) gwl_depth_m, gwe_pred, ground_elevation_amsl [step 07]
    reference/crop_table.csv               Zr (zr_max_m) and MAD (p)
    reference/efficiency_table.csv         Ea (rows kind = application: flood, furrow, sprinkler)
    reference/GeologyClass_Ksat.xlsx       K (m/day), column "Ksat (m/day)"
    farm_inputs/farms.csv                  one row per farm (see farms_template.csv)

OUTPUTS (05_irrigation/)
    advice_<DATE>.csv    one flat row per farm (column names carry their units)
    advice_<DATE>.json   one object per farm: decision1 / decision2 / decision3 blocks, warnings
    advice_<DATE>.png    soil moisture vs trigger

USAGE
    python scripts/08_irrigation_advice.py --selftest
    python scripts/08_irrigation_advice.py --date 2026-10-22 --gwl-file farm_inputs/gwl_today.csv

The old version is kept in scripts/_backup/08_irrigation_advice_v1_OLD.py
================================================================================
"""
import argparse
import json
import sys
import traceback
from datetime import date as date_cls, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import config                                              # noqa: E402
import irrigation_core as core                             # noqa: E402
import irrigation_data as data                             # noqa: E402

OUT_DIR = Path(config.IRRIGATION_OUT_DIR)
FARMS_PATH = Path(config.BASE_DIR) / "farm_inputs" / "farms.csv"
GW_COLS = ["gwl_depth_m", "gwe_pred", "ground_elevation_amsl"]


def log(*a):
    print(*a, flush=True)


def plot_summary(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    need = ["theta_wp_pct", "theta_fc_pct", "theta_trigger_pct", "sm_pct"]
    if not all(c in df.columns for c in need):
        return
    ok = df.dropna(subset=need)
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(ok) + 1.6))
    for i, (_, r) in enumerate(ok.iterrows()):
        ax.hlines(i, r.theta_wp_pct, r.theta_fc_pct, color="lightgray", lw=10, zorder=1)
        ax.plot([r.theta_trigger_pct], [i], "|", color="black", ms=18, zorder=2)
        ax.plot([r.sm_pct], [i], "o", ms=9, zorder=3, color="tab:red" if r.pump_on else "tab:green")
    ax.set_yticks(range(len(ok)))
    ax.set_yticklabels([f"{r.farm_id} ({r.crop})" for _, r in ok.iterrows()])
    ax.set_xlabel("soil moisture (% vol):  grey = wilting point to field capacity, "
                  "black tick = pump-on trigger")
    ax.set_title("Today's soil moisture vs. trigger   (red = pump ON, green = pump OFF)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date_cls.today().isoformat())
    ap.add_argument("--farms", default=str(FARMS_PATH))
    ap.add_argument("--gwl-file", default=None,
                    help="CSV from 07_gw_predict.py: farm_id, gwl_depth_m, gwe_pred, ground_elevation_amsl")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(1 if core.selftest() else 0)

    today = datetime.strptime(args.date, "%Y-%m-%d")
    farms = pd.read_csv(args.farms)
    gw_map = {}
    if args.gwl_file and Path(args.gwl_file).exists():
        g = pd.read_csv(args.gwl_file)
        for c in GW_COLS + ["warnings"]:
            if c not in g.columns:
                g[c] = np.nan
        gw_map = {r["farm_id"]: {c: r[c] for c in GW_COLS + ["warnings"]} for _, r in g.iterrows()}
        if g["ground_elevation_amsl"].isna().all():
            log("!! gwl file has no ground_elevation_amsl: re-run 07_gw_predict.py (new version).")
    log(f"{len(farms)} farm(s), date {today:%Y-%m-%d}, T_P_MAX_HR={config.T_P_MAX_HR:g}")

    ctx = data.Context()
    sm_file = data.find_sm_file(today)
    qa_file = sm_file.with_name(sm_file.stem + "_QA.tif")

    # nearest database point for every farm, then ONE raster read for all of them
    pts = [ctx.lookup.nearest(float(f.lat), float(f.lon)) for f in farms.itertuples()]
    lons = [p[0]["lon"] for p in pts]
    lats = [p[0]["lat"] for p in pts]
    sm_vals = data.sample_raster(sm_file, lons, lats)
    qa_vals = data.sample_raster(qa_file, lons, lats) if qa_file.exists() else [np.nan] * len(farms)

    results = []
    for (_, farm), (pt, snap), sm, qa in zip(farms.iterrows(), pts, sm_vals, qa_vals):
        try:
            results.append(data.advice_for_farm(
                farm.to_dict(), today, ctx, sm_value=sm, sm_qa=qa, point=pt, snap_m=snap,
                gw=gw_map.get(farm.get("farm_id"))))
        except Exception as e:                                  # one bad farm must not stop the run
            log(f"  farm {farm.get('farm_id')} FAILED: {e}")
            if not isinstance(e, (KeyError, ValueError)):
                log("  " + traceback.format_exc().replace("\n", "\n  "))
            def _txt(v):
                return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
            results.append(dict(farm_id=_txt(farm.get("farm_id")), date=f"{today:%Y-%m-%d}",
                                crop=_txt(farm.get("crop")),
                                irrigation_type=_txt(farm.get("irrigation_type")),
                                status="ERROR", decision1={}, decision2={}, decision3={},
                                warnings=[str(e).strip("'\"")]))

    flat = pd.DataFrame([core.flatten_result(r) for r in results])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"advice_{today:%Y-%m-%d}"
    flat.to_csv(f"{stem}.csv", index=False)
    with open(f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, allow_nan=False)      # None -> null, never NaN
    plot_summary(flat, f"{stem}.png")

    show = [c for c in ("farm_id", "crop", "irrigation_type", "sm_pct", "theta_trigger_pct", "pump_state",
                        "d_net_mm", "d_gross_mm", "q_m3_hr", "t_p_hr", "status") if c in flat.columns]
    log("\n" + flat[show].to_string(index=False))
    for r in results:
        for w in r.get("warnings", []):
            log(f"  note {r.get('farm_id')}: {w}")
    log(f"\nSaved {stem}.csv / .json / .png")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)
