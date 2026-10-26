"""
run_pipeline.py - runs the whole daily pipeline in order, from SMAP download to groundwater and irrigation advice.

  1. 01_download_smap.py          SMAP .h5 + float32 GeoTIFF         -> 01_raw_data/smap
  2. 02_scrape_rainfall_temp.py   rainfall + temperature CSVs         -> 01_raw_data/rainfall, temperature
  3. 03_build_lag_features.py     lag features on the 9 km grid       -> 02_lag_adjusted_sm
  4. 04_lag_predict.py            XGBoost -> predicted 9 km SMAP      -> 02_lag_adjusted_sm
  5. 05_downscale_to_70m.py       RF -> 70 m soil moisture GeoTIFF    -> 03_downscaled_sm
  --- everything below is for the farms in farm_inputs/farms.csv ---
  6. 02_scrape... (--date today)  today's station rainfall file (only when it is after 09:30 IST)
  7. 02_scrape... (--last24h)     rainfall of the last 24 h per farm  -> farm_inputs/tp_last24h.csv
                                  (station file first, Open-Meteo as back-up)
  8. 07_gw_predict.py             groundwater depth per farm          -> farm_inputs/gwl_today.csv
  9. 08_irrigation_advice.py      Decision 1 pump ON/OFF, Decision 2 irrigation depth,
                                  Decision 3 pump capacity and time (flood/furrow/sprinkler) -> 05_irrigation
 10. daily summary                one table per farm (rain, GWL, SM, pump decision, depth, Q, pump time)
                                                                      -> 05_irrigation/daily_summary_<date>.csv

Run ONCE before the first daily run:
  scripts/00_build_static_layer.py              (70 m static layer)
  scripts/06_build_point_lookup.py              (soil texture, slope and geology lookup, needed by step 9;
                                                re-run it once after this update to add slope_pct and geology)
  scripts/07_gw_predict.py --build-lookup-only  (groundwater input lookup, needed by step 8)
Steps 6-9 never stop the run: the 70 m map is already saved by then. A step that cannot run is
reported in plain words and the later steps carry on without it (no rainfall -> no groundwater
prediction -> Decision 3 is left empty with a warning). Missing rainfall is never replaced by zero.

Usage:
    python run_pipeline.py                          (one run, then exit)
    python run_pipeline.py --skip-download          (reuse files already in 01_raw_data)
    python run_pipeline.py --skip-scrape            (reuse rainfall/temperature already scraped)
    python run_pipeline.py --anchor 2026-10-02      (force the SMAP anchor date)
    python run_pipeline.py --skip-irrigation        (stop after the 70 m map)

  Continuous mode (keep the Anaconda Prompt window open; stop with Ctrl+C):
    python run_pipeline.py --loop                   (run now, then every day at 10:00 IST)
    python run_pipeline.py --loop --at 10:30        (run now, then every day at 10:30 IST)
    python run_pipeline.py --loop --every-hours 6   (run now, then every 6 hours)
    python run_pipeline.py --loop --wait-first      (do not run now; wait for the first scheduled time)
Schedule time of about 10:00 IST means today's station rainfall file is already final (see run_daily.bat).
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

import config

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
ROOT = os.path.dirname(SCRIPTS)
FARMS_CSV = os.path.join(ROOT, "farm_inputs", "farms.csv")
TP_CSV = os.path.join(ROOT, "farm_inputs", "tp_last24h.csv")
GWL_CSV = os.path.join(ROOT, "farm_inputs", "gwl_today.csv")
GW_MODEL = os.path.join(ROOT, "models", "groundwater", "rf_gwe_final.pkl")
GW_LOOKUP = os.path.join(config.STATIC_DIR, "gw_point_lookup.parquet")
SOIL_LOOKUP = os.path.join(config.STATIC_DIR, "point_lookup.parquet")
IRR_DIR = os.path.join(ROOT, "05_irrigation")
REF_FILES = [config.CROP_TABLE_PATH, config.EFFICIENCY_TABLE_PATH, config.GEOLOGY_K_XLSX]

IST = timezone(timedelta(hours=5, minutes=30))
STATION_FINAL_AFTER = (9, 30)    # today's station file counts as final after 08:30 IST + 60 min
STALE_MAP_DAYS = config.STALE_MAP_DAYS   # warn if the newest 70 m map is older than this (config.py: 3)


class PipelineError(Exception):
    """A fatal problem for one run (the loop mode reports it and tries again later)."""


def emit(text, log=None):
    """Print and also write to the run log."""
    print(text, flush=True)
    if log:
        log.write(text + "\n")


def rain_script():
    """The 02_scrape_rainfall_temp*.py that contains the last-24-hours code (it is a superset of the
    original scraper, so it is also used for the normal daily scrape). Falls back to the usual name."""
    for p in sorted(glob.glob(os.path.join(SCRIPTS, "02_scrape_rainfall_temp*.py"))):
        with open(p, encoding="utf-8", errors="ignore") as f:
            if "def tp_from_stations" in f.read():
                return os.path.basename(p)
    return "02_scrape_rainfall_temp.py"


def run_step(name, args=(), log=None):
    cmd = [sys.executable, os.path.join(SCRIPTS, name), *args]
    print(f"\n{'=' * 70}\n>> {name} {' '.join(args)}\n{'=' * 70}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + (("\n" + proc.stderr) if proc.stderr.strip() else "")
    print(out, flush=True)
    if log:
        log.write(f"\n>> {name}\n{out}\n")
    return proc.returncode


def missing_rain_dates():
    """Last 4 days up to yesterday that lack a rainfall+temperature pair."""
    today = datetime.now().date()
    wanted = [(today - timedelta(days=i)).isoformat() for i in range(1, 5)]
    return [d for d in sorted(wanted)
            if not (os.path.exists(os.path.join(config.RAINFALL_DIR, f"rainfall_{d}.csv"))
                    and os.path.exists(os.path.join(config.TEMPERATURE_DIR, f"temperature_{d}.csv")))]


def latest_sm_date():
    """Date of the newest 70 m soil-moisture map (sm_70m_YYYY-MM-DD.tif) that is NOT after today (IST).
    Maps dated tomorrow or later (from the old D+4 logic) are ignored."""
    today = datetime.now(IST).date().isoformat()
    dates = []
    for f in glob.glob(os.path.join(config.DOWNSCALED_DIR, "sm_70m_*.tif")):
        m = re.search(r"sm_70m_(\d{4}-\d{2}-\d{2})\.tif$", os.path.basename(f))
        if m and m.group(1) <= today:
            dates.append(m.group(1))
    return max(dates) if dates else None


def tp_file_has_values():
    """True if tp_last24h.csv exists and at least one farm has a rainfall value."""
    if not os.path.exists(TP_CSV):
        return False
    import pandas as pd
    try:
        return bool(pd.read_csv(TP_CSV)["TP"].notna().any())
    except Exception:
        return False


def make_tp(log, rain):
    """Rainfall of the last 24 h per farm. Station file first, Open-Meteo as back-up.
    Returns the source used, or None if neither worked."""
    if os.path.exists(TP_CSV):
        os.remove(TP_CSV)                       # never reuse yesterday's file
    for src in ("stations", "openmeteo"):
        rc = run_step(rain, ("--last24h", "--source", src, "--farms", FARMS_CSV, "--out", TP_CSV), log)
        if rc == 0 and tp_file_has_values():
            if src != "stations":
                print("!! station rainfall was not usable; Open-Meteo (gridded model data) was used instead.")
            return src
        print(f"!! rainfall source '{src}' gave no usable value (code {rc}).")
        if os.path.exists(TP_CSV):
            os.remove(TP_CSV)
    return None


def run_irrigation(log):
    """Steps 6-9 for the farms in farm_inputs/farms.csv. Never stops the pipeline.
    Returns a small dict used by the daily summary."""
    info = dict(date=None, tp_src=None, gwl=False, advice_ok=False)

    if not os.path.exists(FARMS_CSV):
        print(f"-- {FARMS_CSV} not found; skipping rainfall, groundwater and irrigation advice.")
        return info
    rain = rain_script()
    now = datetime.now(IST)
    today = now.date().isoformat()

    # 6. today's station file, only once it can be final
    if (now.hour, now.minute) >= STATION_FINAL_AFTER:
        rc = run_step(rain, ("--date", today), log)
        if rc != 0:
            print(f"!! scraping today's station rainfall failed (code {rc}); an earlier file or Open-Meteo is used.")
    else:
        print(f"-- It is {now:%H:%M} IST, before {STATION_FINAL_AFTER[0]}:{STATION_FINAL_AFTER[1]:02d}: "
              f"today's station file is not final yet, so it is not scraped now.")

    # 7. rainfall of the last 24 h at every farm
    tp_src = make_tp(log, rain)
    info["tp_src"] = tp_src

    # 8. groundwater level (needs the rainfall value)
    gw_args = ()
    if os.path.exists(GWL_CSV):
        os.remove(GWL_CSV)                      # never reuse yesterday's file
    if tp_src is None:
        print("!! No rainfall value for the last 24 h -> no groundwater prediction; "
              "Decision 3 (pump capacity) will be empty.")
    elif not os.path.exists(GW_MODEL):
        print("-- Groundwater model not found; Decision 3 (pump capacity) will be empty.")
    elif not os.path.exists(GW_LOOKUP):
        print("!! static/gw_point_lookup.parquet missing. Run once:  "
              "python scripts/07_gw_predict.py --build-lookup-only\n   Decision 3 (pump capacity) will be empty.")
    else:
        rc = run_step("07_gw_predict.py", ("--date", today, "--tp-file", TP_CSV), log)
        if rc == 0 and os.path.exists(GWL_CSV):
            gw_args = ("--gwl-file", GWL_CSV)
            info["gwl"] = True
        else:
            print(f"!! 07_gw_predict.py failed (code {rc}); Decision 3 (pump capacity) will be empty.")

    # 9. irrigation advice
    d = latest_sm_date()
    info["date"] = d
    if not d:
        print("!! No sm_70m_*.tif found; skipping irrigation advice.")
        return info
    if not os.path.exists(SOIL_LOOKUP):
        print("!! static/point_lookup.parquet missing. Run scripts/06_build_point_lookup.py once. "
              "Skipping irrigation advice.")
        return info
    missing_ref = [p for p in REF_FILES if not os.path.exists(p)]
    if missing_ref:
        print("!! Missing reference file(s): " + ", ".join(missing_ref) +
              "\n   Skipping irrigation advice. The 70 m map is still fine.")
        return info
    rc = run_step("08_irrigation_advice.py", ("--date", d, *gw_args), log)
    if rc != 0:
        print(f"!! 08_irrigation_advice.py failed (code {rc}). The 70 m map is still fine.")
    else:
        info["advice_ok"] = True
    return info


def show_summary(info, log):
    """Step 10: one table per farm with rainfall, groundwater, soil moisture and the pump decision."""
    d = info.get("date")
    emit("\n" + "=" * 70 + "\nDAILY SUMMARY\n" + "=" * 70, log)
    if not d:
        emit("No 70 m soil-moisture map found, so there is nothing to summarise.", log)
        return
    emit(f"70 m soil-moisture map : {os.path.join(config.DOWNSCALED_DIR, f'sm_70m_{d}.tif')}", log)
    age = (datetime.now().date() - datetime.strptime(d, "%Y-%m-%d").date()).days
    if age > STALE_MAP_DAYS:
        emit(f"!! The newest map is {age} days old. Check the SMAP download and the lag steps.", log)
    emit(f"Rainfall source (24 h) : {info.get('tp_src') or 'NONE (no rainfall value)'}", log)
    emit(f"Groundwater prediction : {'done' if info.get('gwl') else 'not available today'}", log)

    adv = os.path.join(IRR_DIR, f"advice_{d}.csv")
    if not info.get("advice_ok") or not os.path.exists(adv):
        emit("Irrigation advice    : not produced (see messages above).", log)
        return

    import pandas as pd
    try:
        df = pd.read_csv(adv)
        # add the 24 h rainfall and the predicted groundwater depth per farm
        if os.path.exists(TP_CSV):
            tp = pd.read_csv(TP_CSV)
            if "farm_id" in tp.columns and "TP" in tp.columns:
                df = df.merge(tp[["farm_id", "TP"]].rename(columns={"TP": "rain_24h_mm"}),
                              on="farm_id", how="left")
        if os.path.exists(GWL_CSV):
            gw = pd.read_csv(GWL_CSV)
            if "farm_id" in gw.columns and "gwl_depth_m" in gw.columns:
                df = df.drop(columns=["gwl_depth_m"], errors="ignore").merge(
                    gw[["farm_id", "gwl_depth_m"]], on="farm_id", how="left")
        out_csv = os.path.join(IRR_DIR, f"daily_summary_{d}.csv")
        df.to_csv(out_csv, index=False)

        cols = ["farm_id", "crop", "irrigation_type", "rain_24h_mm", "gwl_depth_m", "sm_pct",
                "theta_trigger_pct", "pump_state", "d_net_mm", "d_gross_mm", "q_m3_hr", "t_p_hr", "status"]
        show = [c for c in cols if c in df.columns]
        emit("\n" + df[show].to_string(index=False), log)
        if "pump_on" in df.columns:
            n_on = int((df["pump_on"] == True).sum())          # noqa: E712
            emit(f"\nPump ON for {n_on} of {len(df)} farm(s).", log)
        if "warnings" in df.columns:
            for _, r in df[df["warnings"].notna() & (df["warnings"].astype(str) != "")].iterrows():
                emit(f"  note {r['farm_id']}: {r['warnings']}", log)
        emit(f"\nSaved {out_csv}", log)
    except Exception as e:
        emit(f"!! Could not build the summary table: {e}", log)


def run_once(a, anchor):
    """One complete pipeline run. Raises PipelineError if the 70 m map cannot be produced."""
    if not os.path.exists(os.path.join(config.STATIC_DIR, "points_70m.parquet")):
        raise PipelineError("Static layer not built yet. Run once:  python scripts/00_build_static_layer.py")

    os.makedirs(config.LOGS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(config.LOGS_DIR, f"run_{stamp}.log"), "w", encoding="utf-8") as log:
        steps = []
        if not a.skip_download:
            steps.append(("01_download_smap.py", ()))
        if not a.skip_scrape:
            for d in missing_rain_dates():
                steps.append((rain_script(), ("--date", d)))
        steps.append(("03_build_lag_features.py", ("--anchor", anchor) if anchor else ()))
        steps.append(("04_lag_predict.py", ()))
        steps.append(("05_downscale_to_70m.py", ()))

        for name, args in steps:
            rc = run_step(name, args, log)
            if rc != 0:
                # a failed download/scrape is not fatal if older files already exist;
                # 03 will stop with a clear message if anything essential is missing
                if name.startswith(("01_", "02_")):
                    print(f"!! {name} failed (code {rc}); continuing with files already on disk.")
                    continue
                raise PipelineError(f"Pipeline stopped at {name} (exit code {rc}). See logs/run_{stamp}.log")

        print("\n70 m soil-moisture map finished. Output is in", config.DOWNSCALED_DIR)
        if not a.skip_irrigation:
            info = run_irrigation(log)
            show_summary(info, log)
    print("\nPipeline finished.")


def next_run_time(now, at_str, last_start, every_hours):
    """When the next cycle starts (IST)."""
    if every_hours:
        return last_start + timedelta(hours=every_hours)
    hh, mm = (int(x) for x in at_str.split(":"))
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    return t


def sleep_until(target):
    """Sleep in short pieces so Ctrl+C works straight away."""
    while True:
        remaining = (target - datetime.now(IST)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(30.0, remaining))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-scrape", action="store_true")
    ap.add_argument("--skip-irrigation", action="store_true")
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--loop", action="store_true",
                    help="keep running: now, then again every day (see --at) or every N hours")
    ap.add_argument("--at", default="10:00",
                    help="daily start time in IST, HH:MM (loop mode, default 10:00)")
    ap.add_argument("--every-hours", type=float, default=None,
                    help="loop mode: repeat every N hours instead of once a day at --at")
    ap.add_argument("--wait-first", action="store_true",
                    help="loop mode: do not run now, wait for the first scheduled time")
    a = ap.parse_args()

    if not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", a.at):
        sys.exit("--at must look like HH:MM, for example 10:00")
    if a.every_hours is not None and a.every_hours <= 0:
        sys.exit("--every-hours must be greater than 0")

    # ---- single run (unchanged behaviour) ----
    if not a.loop:
        try:
            run_once(a, a.anchor)
        except PipelineError as e:
            sys.exit(str(e))
        return

    # ---- continuous mode ----
    plan = (f"every {a.every_hours:g} hour(s)" if a.every_hours else f"every day at {a.at} IST")
    print(f"Continuous mode: the pipeline will run {plan}. Press Ctrl+C to stop.", flush=True)
    cycle = 0
    try:
        while True:
            now = datetime.now(IST)
            if cycle == 0 and a.wait_first:
                last_start = now
            else:
                cycle += 1
                last_start = now
                print(f"\n{'#' * 70}\n# CYCLE {cycle} started {now:%Y-%m-%d %H:%M} IST\n{'#' * 70}", flush=True)
                try:
                    # --anchor is only meant for the first cycle; later cycles use the newest data
                    run_once(a, a.anchor if cycle == 1 else None)
                except PipelineError as e:
                    print(f"!! Cycle {cycle} failed: {e}\n   Will try again at the next scheduled time.", flush=True)
                except Exception as e:                      # nothing may kill the loop
                    print(f"!! Cycle {cycle} crashed: {e}\n{traceback.format_exc()}"
                          f"   Will try again at the next scheduled time.", flush=True)
            nxt = next_run_time(datetime.now(IST), a.at, last_start, a.every_hours)
            if a.wait_first and cycle == 0:
                cycle = -1 if False else cycle                # keep first-cycle logic simple
            print(f"\nNext run: {nxt:%Y-%m-%d %H:%M} IST. Waiting... (Ctrl+C to stop)", flush=True)
            sleep_until(nxt)
            if cycle == 0:
                cycle = 0                                      # first scheduled run counts as cycle 1
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C). Goodbye.", flush=True)


if __name__ == "__main__":
    main()