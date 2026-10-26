"""
services.py - the dashboard's contact with the pipeline. NO FORMULAS HERE.

  load_context()            the same Context the daily script uses (tables + soil/slope/geology lookup)
  map_date()                date of the newest 70 m map that is not after today (IST)
  rainfall_24h(lat, lon)    rainfall of the last 24 h at a point, with the SAME code as the pipeline
                            (02_scrape_rainfall_temp*.py --last24h): station file first, Open-Meteo back-up
  run_advice(farm, ctx)     rainfall -> groundwater -> advice_for_farm(...) -> one result dictionary
"""
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "scripts"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config                    # noqa: E402
import irrigation_data as data   # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))


def today_ist():
    return datetime.now(IST).date()


def load_context():
    return data.Context()


def map_date():
    return data.latest_map_date(today_ist())


def _rain_script():
    for p in sorted((ROOT / "scripts").glob("02_scrape_rainfall_temp*.py")):
        try:
            if "def tp_from_stations" in p.read_text(encoding="utf-8", errors="ignore"):
                return p
        except OSError:
            pass
    return ROOT / "scripts" / "02_scrape_rainfall_temp.py"


def rainfall_24h(lat, lon, timeout=240):
    """Returns {'tp_mm': float or None, 'source': 'stations'|'openmeteo'|None, 'note': str}."""
    script = _rain_script()
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    notes = []
    for src in ("stations", "openmeteo"):
        with tempfile.TemporaryDirectory() as td:
            farms, out = Path(td) / "farm.csv", Path(td) / "tp.csv"
            pd.DataFrame({"farm_id": ["dashboard"], "lat": [lat], "lon": [lon]}).to_csv(farms, index=False)
            cmd = [sys.executable, str(script), "--last24h", "--source", src,
                   "--farms", str(farms), "--out", str(out)]
            try:
                p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
                                   env=env, encoding="utf-8", errors="replace")
            except subprocess.TimeoutExpired:
                notes.append(f"rainfall source '{src}' timed out")
                continue
            if p.returncode == 0 and out.exists():
                try:
                    tp = pd.to_numeric(pd.read_csv(out)["TP"], errors="coerce").iloc[0]
                except Exception:
                    tp = float("nan")
                if tp == tp:
                    if src != "stations":
                        notes.append("station rainfall was not usable; Open-Meteo (gridded model data) was used")
                    return {"tp_mm": float(tp), "source": src, "note": "; ".join(notes)}
            notes.append(f"rainfall source '{src}' gave no usable value")
    return {"tp_mm": None, "source": None, "note": "; ".join(notes)}


def run_advice(farm, ctx, rain_fn=None):
    """farm: dict (farm_id, lat, lon, crop, irrigation_type, area_ha, well_depth_m, well_radius_m, furrow_*,
    sprinkler_*, lateral_spacing_m). Returns the nested result dict from advice_for_farm plus 'rain'."""
    day = map_date()
    if day is None:
        raise FileNotFoundError("No 70 m soil-moisture map found in 03_downscaled_sm. Run run_pipeline.py first.")
    rain = (rain_fn or rainfall_24h)(float(farm["lat"]), float(farm["lon"]))
    extra, gw = [], None
    if rain["tp_mm"] is None:
        extra.append("no rainfall value for this location, so groundwater and well discharge are not calculated")
    else:
        try:
            gw = ctx.predict_gw(float(farm["lat"]), float(farm["lon"]), today_ist(), rain["tp_mm"])
        except Exception as e:                     # model / lookup missing: the rest of the advice still works
            extra.append(f"groundwater model not available ({e}), so well discharge is not calculated")
    if rain.get("note"):
        extra.append("rainfall: " + rain["note"])
    res = data.advice_for_farm(farm, day, ctx, gw=gw)
    for w in extra:
        if w not in res["warnings"]:
            res["warnings"].append(w)
    res["rain"] = rain
    res["map_date"] = day.isoformat()
    return res
