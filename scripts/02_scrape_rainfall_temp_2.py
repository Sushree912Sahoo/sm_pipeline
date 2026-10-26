"""
02_scrape_rainfall_temp.py
----------------------------
ONE command, no options needed:

    python scripts/02_scrape_rainfall_temp.py

It works out by itself which days of rainfall + temperature the lag model needs,
fetches only the ones that are missing, and tells you what is ready.

How it decides (every run, so skipped days are caught up automatically):
  1. Find the most recent SMAP date on disk (01_raw_data/smap), call it D0.
  2. The lag model for anchor D needs rainfall + temperature for D, D+1, D+2, D+3
     and predicts D+4. Needed days = D0-1 ... D0+3, never later than today.
     (D0-1 is included so the previous anchor also stays usable.)
  3. Days already on disk are skipped. Today's day is re-fetched on every run
     (it may have grown). Past days are never overwritten.
  4. Rainfall: rainfall.nic.in. If the site refuses a PAST day, that day is filled
     from Open-Meteo (config.RAIN_FALLBACK_OPENMETEO) and marked source=openmeteo.
     Temperature: Open-Meteo for every day.
  5. Prints which anchors/target dates are READY for the next steps.

Run 01_download_smap.py first, so the latest SMAP date is known.

Saves: 01_raw_data/rainfall/rainfall_<date>.csv
       01_raw_data/temperature/temperature_<date>.csv

Other options (rarely needed):
    --date 2026-10-05   fetch just that one day (site only, no backup source)
    --no-fallback       never use the Open-Meteo backup for rainfall
If a run fails, see logs/scrape_error.png.

NEW (separate from everything above - the lag-model flow is untouched):
    python scripts/02_scrape_rainfall_temp.py --last24h --time "2026-10-13 08:00"
  Downloads the rainfall of the 24 hours BEFORE the given IST time for every farm in
  farm_inputs/farms.csv (farm_id, lat, lon) and writes farm_inputs/tp_last24h.csv.
  Source: rainfall.nic.in station files (default, --source stations) or Open-Meteo hourly
  (--source openmeteo, gridded model data, NOT rain-gauge data).
  See the section "LAST 24 HOURS OF RAINFALL" near the bottom of this file.
"""

import sys
import os
import re
import glob
import time
import argparse
from datetime import datetime, timedelta, date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import requests

SEARCH_BUTTON_ID = "ButtonSubmit"
RESULTS_TABLE_ID = "ctl00_ContentPlaceHolder1_grdMonthlyCompare"
# The date box has onkeydown="return false;" (no typing) and class "disable_past_dates",
# so the site may only serve the current day. The value is set with JavaScript.
DATE_INPUT_ID = "ctl00_ContentPlaceHolder1_txtDateOfTheft"

# remembers what happened during this run
_STATE = {"date_rejected": False, "site_rejects_past": False}


def _today():
    return date.today()


# ---------------------------------------------------------------------
# What is already on disk
# ---------------------------------------------------------------------
def get_latest_smap_date():
    """Most recent SMAP date (YYYY-MM-DD) among the files in 01_raw_data/smap, or None."""
    dates = []
    for f in glob.glob(os.path.join(config.SMAP_RAW_DIR, "*.h5")):
        m = re.search(r"(\d{8})T\d{6}", os.path.basename(f))
        if m:
            dates.append(datetime.strptime(m.group(1), "%Y%m%d").date())
    return max(dates).isoformat() if dates else None


def have_day(date_iso):
    """True if BOTH the rainfall and the temperature file exist for this date."""
    return (os.path.exists(os.path.join(config.RAINFALL_DIR, f"rainfall_{date_iso}.csv"))
            and os.path.exists(os.path.join(config.TEMPERATURE_DIR, f"temperature_{date_iso}.csv")))


# ---------------------------------------------------------------------
# Temperature (Open-Meteo)
# ---------------------------------------------------------------------
def get_daily_temperature(lat, lon, date_iso):
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_mean",
        "start_date": date_iso, "end_date": date_iso,
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()["daily"]["temperature_2m_mean"][0]


# ---------------------------------------------------------------------
# Rainfall (rainfall.nic.in, via Selenium)
# ---------------------------------------------------------------------
def set_target_date(driver, wait, target_date_display):
    """Today's date needs no change. For other dates, set the value WITHOUT
    firing change/blur events (those make the page reload -> stale element)."""
    if target_date_display == _today().strftime("%d/%m/%Y"):
        print(f"  Target is today ({target_date_display}); keeping the page's default date.")
        return True
    try:
        date_input = wait.until(EC.presence_of_element_located((By.ID, DATE_INPUT_ID)))
        driver.execute_script("arguments[0].value = arguments[1];", date_input, target_date_display)
        time.sleep(1)
        date_input = driver.find_element(By.ID, DATE_INPUT_ID)   # re-find in case the page refreshed
        actual_value = date_input.get_attribute("value")
        if actual_value == target_date_display:
            print(f"  Set date field to {target_date_display} (confirmed).")
            return True
        print(f"  WARNING: tried to set date to {target_date_display}, but field "
              f"shows '{actual_value}' instead. The site is rejecting this date.")
        _STATE["date_rejected"] = True
        return False
    except Exception as e:
        print(f"  WARNING: could not set the date field: {repr(e)[:200]}")
        return False


def scrape_odisha_rainfall(target_date_display):
    url = "https://rainfall.nic.in/rainfallnew/PublicRainFallDayWise.aspx"
    all_districts_data = []
    driver = None
    try:
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")

        if config.SELENIUM_URL:
            print(f"Connecting to remote Selenium grid at {config.SELENIUM_URL}...")
            driver = webdriver.Remote(command_executor=config.SELENIUM_URL, options=options)
        else:
            if config.HEADLESS:
                options.add_argument("--headless=new")
            driver = webdriver.Chrome(options=options)

        print("Loading the webpage...")
        driver.get(url)
        print("--> Pausing for 15 seconds for the page to load...")
        time.sleep(15)

        wait = WebDriverWait(driver, 40)

        print(f"Attempting to set target date to {target_date_display}...")
        date_ok = set_target_date(driver, wait, target_date_display)
        if not date_ok and target_date_display != _today().strftime("%d/%m/%Y"):
            print(f"  ERROR: the site did not accept {target_date_display}, so the page would show "
                  f"a different date. Not saving, to avoid a wrongly dated CSV.")
            return None

        district_dropdown_element = wait.until(
            EC.presence_of_element_located((By.ID, "ctl00_ContentPlaceHolder1_ddlDist"))
        )
        select = Select(district_dropdown_element)
        district_options = [o.text for o in select.options if o.get_attribute("value")]
        print(f"Found {len(district_options)} districts to scrape.")

        for district in district_options:
            print(f"Fetching data for: {district}...")
            district_dropdown = Select(driver.find_element(By.ID, "ctl00_ContentPlaceHolder1_ddlDist"))
            search_button = driver.find_element(By.ID, SEARCH_BUTTON_ID)
            district_dropdown.select_by_visible_text(district)

            try:
                old_table = driver.find_element(By.ID, RESULTS_TABLE_ID)
            except Exception:
                old_table = None

            time.sleep(2)
            search_button.click()

            try:
                if old_table:
                    wait.until(EC.staleness_of(old_table))
                results_table = wait.until(EC.visibility_of_element_located((By.ID, RESULTS_TABLE_ID)))
                rows = results_table.find_elements(By.TAG_NAME, "tr")[1:]
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) == 2:
                        all_districts_data.append(
                            {"RR Station": cells[0].text, "Rain (in mm.)": cells[1].text, "District": district}
                        )
                print(f"  -> Found {len(rows)} records for {district}.")
            except Exception:
                print(f"  -> No data table found for {district} on this date.")

        return pd.DataFrame(all_districts_data)

    except Exception as e:
        print(f"An error occurred: {repr(e)[:300]}")
        try:
            os.makedirs(config.LOGS_DIR, exist_ok=True)
            driver.save_screenshot(os.path.join(config.LOGS_DIR, "scrape_error.png"))
            with open(os.path.join(config.LOGS_DIR, "scrape_error.html"), "w", encoding="utf-8") as fh:
                fh.write(driver.page_source)
            print("Saved logs\\scrape_error.png and logs\\scrape_error.html")
        except Exception:
            pass
        return None
    finally:
        if driver:
            print("Scraping finished. Closing the browser in 5 seconds...")
            time.sleep(5)
            driver.quit()


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------
def save_day_csvs(merged_df, date_iso):
    """Writes the two per-date files: rainfall_<date>.csv and temperature_<date>.csv."""
    os.makedirs(config.RAINFALL_DIR, exist_ok=True)
    os.makedirs(config.TEMPERATURE_DIR, exist_ok=True)
    base = ["RR Station", "District", "Latitude", "Longitude"]
    extra = ["source"] if "source" in merged_df.columns else []
    rain_path = os.path.join(config.RAINFALL_DIR, f"rainfall_{date_iso}.csv")
    temp_path = os.path.join(config.TEMPERATURE_DIR, f"temperature_{date_iso}.csv")
    merged_df[base + ["Rain (in mm.)", "date"] + extra].to_csv(rain_path, index=False)
    merged_df[base + ["temperature", "date"] + extra].to_csv(temp_path, index=False)
    print(f"\nSaved {len(merged_df)} rows -> {rain_path}\n                      {temp_path}")
    return rain_path


def run(target_date=None):
    """Fetch ONE day from the rainfall site (+ Open-Meteo temperature) and save it.
    target_date: 'YYYY-MM-DD' (default today). Returns the saved path, or None."""
    if not os.path.exists(config.REFERENCE_COORDS_PATH):
        raise FileNotFoundError(
            f"Missing required file: {config.REFERENCE_COORDS_PATH}\n"
            f"Copy your RR_Reference_Coordinates.csv into that folder first."
        )
    rr_ref_cord = pd.read_csv(config.REFERENCE_COORDS_PATH)

    target_dt = datetime.strptime(target_date, "%Y-%m-%d") if target_date else datetime.combine(_today(), datetime.min.time())
    target_date_display = target_dt.strftime("%d/%m/%Y")
    date_iso = target_dt.strftime("%Y-%m-%d")
    print(f"Scraping rainfall for {target_date_display}...")

    rainfall_data = scrape_odisha_rainfall(target_date_display)
    if rainfall_data is None or rainfall_data.empty:
        print("No rainfall data collected — nothing to save.")
        return None

    merged_df = pd.merge(
        rainfall_data[["RR Station", "District", "Rain (in mm.)"]],
        rr_ref_cord[["RR Station", "District", "Latitude", "Longitude"]],
        on=["RR Station", "District"], how="left",
    )[["RR Station", "District", "Latitude", "Longitude", "Rain (in mm.)"]]

    print("Fetching temperature for each station from Open-Meteo...")
    temperatures = []
    for lat, lon in merged_df[["Latitude", "Longitude"]].itertuples(index=False):
        if pd.notna(lat) and pd.notna(lon):
            try:
                temperatures.append(get_daily_temperature(lat, lon, date_iso))
            except Exception as e:
                print(f"  Fetch failed for lat={lat}, lon={lon}: {e}")
                temperatures.append(None)
        else:
            temperatures.append(None)
    merged_df["temperature"] = temperatures
    merged_df["date"] = date_iso

    if merged_df["temperature"].notna().sum() == 0:
        print("No temperature could be fetched for any station — not saving this day.")
        return None
    return save_day_csvs(merged_df, date_iso)


# ---------------------------------------------------------------------
# Backup rainfall source (only for PAST days the site refuses)
# ---------------------------------------------------------------------
def openmeteo_day(date_iso):
    """Rainfall (precipitation_sum) + temperature for every station in the reference
    file, from Open-Meteo. This is NOT the station data your model was trained on."""
    stations = pd.read_csv(config.REFERENCE_COORDS_PATH)
    stations = stations.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    rows = []
    for i in range(0, len(stations), 50):
        chunk = stations.iloc[i:i + 50]
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in chunk["Latitude"]),
            "longitude": ",".join(f"{v:.4f}" for v in chunk["Longitude"]),
            "daily": "precipitation_sum,temperature_2m_mean",
            "start_date": date_iso, "end_date": date_iso, "timezone": "Asia/Kolkata",
        }
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            data = [data]
        for (_, st), d in zip(chunk.iterrows(), data):
            rows.append({
                "RR Station": st["RR Station"], "District": st["District"],
                "Latitude": st["Latitude"], "Longitude": st["Longitude"],
                "Rain (in mm.)": d["daily"]["precipitation_sum"][0],
                "temperature": d["daily"]["temperature_2m_mean"][0],
                "date": date_iso, "source": "openmeteo",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# The automatic mode
# ---------------------------------------------------------------------
def days_needed():
    """(latest SMAP date or None, list of 'YYYY-MM-DD' days to have on disk)."""
    today = _today()
    latest = get_latest_smap_date()
    if latest is None:
        print("No SMAP files found in 01_raw_data/smap. Run 01_download_smap.py first "
              "so the right days can be worked out.\nFor now: fetching the last 5 days up to today.")
        start, end = today - timedelta(days=5), today
    else:
        D0 = date.fromisoformat(latest)
        start = D0 - timedelta(days=1)             # keeps the previous anchor usable too
        end = min(D0 + timedelta(days=3), today)   # never ask for the future
    days, d = [], start
    while d <= end:
        days.append(d.isoformat())
        d += timedelta(days=1)
    return latest, days


def fetch_day(date_iso, allow_fallback):
    """Site first. If the site refuses a PAST day, use the Open-Meteo backup (if allowed).
    Returns 'site', 'openmeteo' or None."""
    is_today = date_iso == _today().isoformat()

    if not (_STATE["site_rejects_past"] and not is_today):
        _STATE["date_rejected"] = False
        if run(target_date=date_iso) is not None:
            return "site"
        if _STATE["date_rejected"] and not is_today:
            _STATE["site_rejects_past"] = True
            print("\n  The rainfall site only serves the current day, so the browser will not "
                  "be opened again for other past days in this run.")

    if allow_fallback and not is_today:
        print(f"  Using the Open-Meteo backup for {date_iso} ...")
        try:
            df = openmeteo_day(date_iso)
            if not df.empty:
                save_day_csvs(df, date_iso)
                return "openmeteo"
        except Exception as e:
            print(f"  Backup failed too: {repr(e)[:200]}")
    return None


def run_auto(allow_fallback=True):
    latest, days = days_needed()
    today_iso = _today().isoformat()
    todo = [d for d in days if d == today_iso or not have_day(d)]

    print(f"Today                : {today_iso}")
    print(f"Latest SMAP date D0  : {latest or '(none yet)'}")
    print(f"Days the model needs : {days}")
    print(f"Already on disk      : {[d for d in days if have_day(d) and d != today_iso] or '(none)'}")
    print(f"To fetch now         : {todo or '(nothing)'}")

    results = {}
    for d in todo:
        print(f"\n{'=' * 60}\nFetching {d}\n{'=' * 60}")
        results[d] = fetch_day(d, allow_fallback)

    print("\n" + "-" * 60 + "\nSUMMARY")
    for d, how in results.items():
        label = {"site": "from rainfall.nic.in", "openmeteo": "from Open-Meteo BACKUP (not station data)",
                 None: "NOT AVAILABLE"}[how]
        print(f"  {d}: {label}")

    if latest:
        D0 = date.fromisoformat(latest)
        print("\nReady for the lag model:")
        any_ready = False
        for anchor in (D0, D0 - timedelta(days=1)):
            window = [(anchor + timedelta(days=i)).isoformat() for i in range(4)]
            miss = [w for w in window if not have_day(w)]
            if miss:
                print(f"  anchor {anchor} -> target {anchor + timedelta(days=4)}: missing {miss}")
            else:
                any_ready = True
                print(f"  anchor {anchor} -> target {anchor + timedelta(days=4)}: READY")
        if not any_ready:
            print("\nNo anchor is complete yet. If days are missing because the site refuses past "
                  "dates, set RAIN_FALLBACK_OPENMETEO = True in config.py, or run this every day "
                  "so the days build up.")
    return results



# =====================================================================
# LAST 24 HOURS OF RAINFALL  (NEW - independent of the lag-model flow above)
# =====================================================================
# Window: (end - 24 h, end]  in Indian Standard Time.
# Open-Meteo "precipitation" is the SUM OF THE PRECEDING HOUR, so a stamp T means the
# hour [T-1h, T]. For a prediction at 08:00 the 24 stamps used are
# (previous day) 09:00, 10:00, ... 23:00, 00:00, ... 08:00 (today). Nothing after `end`
# is used, and no hour is counted twice.
from datetime import timezone

IST = timezone(timedelta(hours=5, minutes=30))     # India has no daylight saving: fixed offset is exact
RAIN24_MIN_HOURS = 24                              # all 24 hours must be present, else TP is left empty
_OM_FORECAST = "https://api.open-meteo.com/v1/forecast"
_OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def _now_ist():
    return datetime.now(IST)


def parse_ist_time(text):
    """'YYYY-MM-DD HH:MM' (read as IST) -> aware datetime. None -> the current time."""
    if not text:
        return _now_ist()
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=IST)


def window_stamps(end_ist):
    """The 24 hourly stamps (end-23h ... end) whose preceding-hour sums make up the window."""
    end_h = end_ist.replace(minute=0, second=0, microsecond=0)
    return [end_h - timedelta(hours=k) for k in range(23, -1, -1)]


def aggregate_window(times, values, end_ist):
    """times: list of 'YYYY-MM-DDTHH:MM' (IST); values: mm per preceding hour (None = missing).
    Returns (total_mm or None, hours_present). Missing hours are NEVER treated as 0."""
    by_time = dict(zip(times, values))
    stamps = window_stamps(end_ist)
    assert max(stamps) <= end_ist, "internal error: a stamp is after the prediction time"
    got = [by_time.get(t.strftime("%Y-%m-%dT%H:%M")) for t in stamps]
    present = [v for v in got if v is not None and v == v]
    total = round(float(sum(present)), 3) if len(present) >= RAIN24_MIN_HOURS else None
    return total, len(present)


def window_values(times, values, end_ist):
    """The 24 (stamp, mm) pairs of the window; mm is None where the source has no value."""
    by_time = dict(zip(times, values))
    return [(t.strftime("%Y-%m-%d %H:%M"), by_time.get(t.strftime("%Y-%m-%dT%H:%M")))
            for t in window_stamps(end_ist)]


def _hourly_precip(lats, lons, first_date, last_date):
    """Hourly precipitation (mm, preceding hour) for several points, IST timestamps."""
    age_days = (_now_ist().date() - first_date).days
    url = _OM_ARCHIVE if age_days > 80 else _OM_FORECAST      # forecast API keeps ~92 past days
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "hourly": "precipitation",
        "start_date": first_date.isoformat(), "end_date": last_date.isoformat(),
        "timezone": "Asia/Kolkata",
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise SystemExit(f"Open-Meteo could not be used ({type(e).__name__}); no rainfall value from it")
    if isinstance(data, dict):
        data = [data]
    return [(d["hourly"]["time"], d["hourly"]["precipitation"]) for d in data]


def fetch_last24h_rainfall(points, end_ist=None, hourly_log=None):
    """points: DataFrame with farm_id, lat, lon.  Returns a DataFrame, one row per point:
    farm_id, lat, lon, window_start_ist, window_end_ist, hours_covered, tp_mm, warnings."""
    end_ist = end_ist or _now_ist()
    end_h = end_ist.replace(minute=0, second=0, microsecond=0)
    notes = []
    if end_h != end_ist.replace(second=0, microsecond=0):
        notes.append(f"time floored to {end_h:%H:%M} (hourly data)")
    if end_h > _now_ist():
        raise SystemExit(f"Prediction time {end_h:%Y-%m-%d %H:%M} IST is in the future. "
                         f"Future rainfall would be a forecast, not an observation - refusing "
                         f"(no future-data leakage).")
    stamps = window_stamps(end_h)
    first_date, last_date = stamps[0].date(), stamps[-1].date()

    rows = []
    pts = points.reset_index(drop=True)
    for i in range(0, len(pts), 50):
        chunk = pts.iloc[i:i + 50]
        series = _hourly_precip(chunk["lat"].astype(float).tolist(),
                                chunk["lon"].astype(float).tolist(), first_date, last_date)
        for (_, p), (times, vals) in zip(chunk.iterrows(), series):
            total, n = aggregate_window(times, vals, end_h)
            if hourly_log is not None:
                hourly_log.append((p["farm_id"], window_values(times, vals, end_h)))
            w = list(notes)
            if total is None:
                w.append(f"only {n}/24 hours available -> TP left empty (not filled with 0)")
            rows.append({"farm_id": p["farm_id"], "lat": p["lat"], "lon": p["lon"],
                         "window_start_ist": (end_h - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M"),
                         "window_end_ist": end_h.strftime("%Y-%m-%d %H:%M"),
                         "hours_covered": n, "tp_mm": total, "warnings": "; ".join(w)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Source 1 (default): rainfall.nic.in STATION files saved by the normal daily flow
# ---------------------------------------------------------------------
# A station's daily value is taken to be the 24 hours ENDING 08:30 IST of its date (the IMD
# convention for daily rainfall). The app does NOT scrape the site per request (that takes
# minutes): the scheduled run saves rainfall_<date>.csv and the app reads the newest finished one.
import numpy as np

STATION_END_HHMM = (8, 30)


def station_window_end(day):
    return datetime(day.year, day.month, day.day, *STATION_END_HHMM, tzinfo=IST)


def pick_station_day(end_ist, publish_delay_min=60, rain_dir=None):
    """Newest rainfall_<date>.csv whose 24-h window (ending D 08:30 IST) is already over at
    `end_ist` AND whose file was written at least `publish_delay_min` after that moment
    (so it holds the finished day, not a half-day value). Looks at that day, then the day before.
    Returns (day, path, written_ist, notes)."""
    rain_dir = rain_dir or config.RAINFALL_DIR
    newest = (end_ist - timedelta(hours=STATION_END_HHMM[0], minutes=STATION_END_HHMM[1])).date()
    notes = []
    for back in (0, 1):
        day = newest - timedelta(days=back)
        f = os.path.join(rain_dir, f"rainfall_{day.isoformat()}.csv")
        if not os.path.exists(f):
            notes.append(f"{day}: no station file")
            continue
        written = datetime.fromtimestamp(os.path.getmtime(f), IST)
        ready = station_window_end(day) + timedelta(minutes=publish_delay_min)
        if written < ready:
            notes.append(f"{day}: file written {written:%H:%M}, before {ready:%H:%M} -> may be incomplete")
            continue
        if back:
            notes.append(f"newest day {newest} not available -> using {day}")
        return day, f, written, notes
    return None, None, None, notes


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def tp_from_stations(points, end_ist=None, max_km=25.0, method="nearest",
                     publish_delay_min=60, rain_dir=None):
    """Rainfall (mm, 24 h ending 08:30 IST) at each point from the nearest station(s)."""
    end_ist = end_ist or _now_ist()
    if end_ist > _now_ist():
        raise SystemExit("Prediction time is in the future - refusing (no future-data leakage).")
    day, path, written, notes = pick_station_day(end_ist, publish_delay_min, rain_dir)
    if day is None:
        raise SystemExit("No finished station file available: " + "; ".join(notes) +
                         "\nRun the daily scrape after the data is published, or use --source openmeteo.")
    st = pd.read_csv(path)
    st["rain"] = pd.to_numeric(st["Rain (in mm.)"], errors="coerce")
    n_all = len(st)
    st = st.dropna(subset=["Latitude", "Longitude", "rain"]).reset_index(drop=True)
    from_backup = "source" in st.columns and (st["source"] == "openmeteo").any()
    if from_backup:
        notes.append("this day's file was filled from the Open-Meteo backup, not station data")
    if len(st) == 0:
        raise SystemExit(f"{path} has no station with coordinates and a rainfall value.")
    n_zero, n_pos = int((st["rain"] == 0).sum()), int((st["rain"] > 0).sum())
    print(f"Stations with a value: {len(st)} | exactly 0.0 mm: {n_zero} | above 0 mm: {n_pos} | "
          f"max {st['rain'].max():.1f} mm")
    if n_pos == 0:
        notes.append("ALL stations report 0.0 mm: either it did not rain, or the site had not published "
                     "yet / shows a running total. Compare with --source openmeteo before trusting it")
    w_end = station_window_end(day)
    pts = points.reset_index(drop=True)
    rows = []
    for _, p in pts.iterrows():
        d = _haversine_km(float(p["lat"]), float(p["lon"]),
                          st["Latitude"].to_numpy(float), st["Longitude"].to_numpy(float))
        order = np.argsort(d)
        w = list(notes)
        tp, name, dist = None, "", None
        if method == "idw":
            sel = [i for i in order[:3] if d[i] <= max_km]
            if sel:
                wt = np.array([1.0 / max(d[i], 0.5) ** 2 for i in sel])
                tp = round(float(np.dot(wt, st.loc[sel, "rain"].to_numpy(float)) / wt.sum()), 3)
                name = "; ".join(st.loc[sel, "RR Station"].astype(str))
                dist = round(float(d[sel[0]]), 1)
        else:
            i = order[0]
            if d[i] <= max_km:
                tp, name, dist = round(float(st.at[i, "rain"]), 3), str(st.at[i, "RR Station"]), round(float(d[i]), 1)
        if tp is None:
            w.append(f"no station within {max_km:g} km (nearest {d[order[0]]:.0f} km) -> TP left empty")
        rows.append({"farm_id": p["farm_id"], "lat": p["lat"], "lon": p["lon"],
                     "window_start_ist": (w_end - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M"),
                     "window_end_ist": w_end.strftime("%Y-%m-%d %H:%M"),
                     "station": name, "distance_km": dist, "tp_mm": tp,
                     "station_file_written_ist": f"{written:%Y-%m-%d %H:%M}",
                     "warnings": "; ".join(w)})
    print(f"Station file used: {path}  ({len(st)}/{n_all} stations with a value)")
    return pd.DataFrame(rows)


def run_last24h(farms_path, time_text, out_path, tp_unit="mm", source="stations",
                max_km=25.0, method="nearest", publish_delay_min=60, show_hourly=False):
    farms = pd.read_csv(farms_path)
    for c in ("farm_id", "lat", "lon"):
        if c not in farms.columns:
            raise SystemExit(f"{farms_path} needs a '{c}' column")
    end_ist = parse_ist_time(time_text)
    if source == "openmeteo":
        eh = end_ist.replace(minute=0, second=0, microsecond=0)
        print(f"Rainfall window: {eh - timedelta(hours=24):%Y-%m-%d %H:%M}  ->  {eh:%Y-%m-%d %H:%M} IST (Open-Meteo, hourly)")
        hourly_log = [] if show_hourly else None
        res = fetch_last24h_rainfall(farms, end_ist, hourly_log)
        if show_hourly:
            for fid, hv in hourly_log[:1]:
                print(f"\nHourly rainfall for farm {fid} (stamp = END of the hour, mm in that hour):")
                for stamp, v in hv:
                    print(f"  {stamp}  {'-' if v is None else f'{v:.2f}'}")
        res["source"] = "open-meteo hourly precipitation (gridded model data, not gauge data)"
        show = ["farm_id", "hours_covered", "tp_mm", "TP", "warnings"]
    else:
        res = tp_from_stations(farms, end_ist, max_km, method, publish_delay_min)
        res["source"] = f"rainfall.nic.in stations ({method}, max {max_km:g} km)"
        show = ["farm_id", "station", "distance_km", "tp_mm", "TP", "warnings"]
    factor = 1.0 if tp_unit == "mm" else 0.001
    res["TP"] = (res["tp_mm"].astype(float) * factor).round(6)
    res["tp_unit"] = tp_unit
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    res.to_csv(out_path, index=False)
    print(res[show].to_string(index=False))
    print(f"\nSaved {out_path}")
    print(f"NOTE: TP column is in {tp_unit}. It must match the unit and the 24-hour definition used "
          f"when the groundwater model was trained (use --tp-unit m for metres).")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch the rainfall + temperature days the lag model needs.")
    parser.add_argument("--date", type=str, default=None,
                        help="Fetch only this one day (YYYY-MM-DD), site only.")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Never use the Open-Meteo backup for rainfall.")
    parser.add_argument("--last24h", action="store_true",
                        help="NEW: download the rainfall of the 24 hours before --time for each farm.")
    parser.add_argument("--time", type=str, default=None,
                        help="with --last24h: prediction time 'YYYY-MM-DD HH:MM' in IST (default: now).")
    parser.add_argument("--farms", type=str,
                        default=os.path.join(config.BASE_DIR, "farm_inputs", "farms.csv"),
                        help="with --last24h: CSV with farm_id, lat, lon.")
    parser.add_argument("--out", type=str,
                        default=os.path.join(config.BASE_DIR, "farm_inputs", "tp_last24h.csv"),
                        help="with --last24h: output CSV.")
    parser.add_argument("--tp-unit", choices=["mm", "m"], default="mm",
                        help="with --last24h: unit of the TP column (default mm).")
    parser.add_argument("--show-hourly", action="store_true",
                        help="with --source openmeteo: print the 24 hourly values of the first farm.")
    parser.add_argument("--source", choices=["stations", "openmeteo"], default="stations",
                        help="with --last24h: rainfall.nic.in station files (default) or Open-Meteo hourly.")
    parser.add_argument("--max-km", type=float, default=25.0,
                        help="with --source stations: farthest acceptable station distance (km).")
    parser.add_argument("--method", choices=["nearest", "idw"], default="nearest",
                        help="with --source stations: nearest station, or inverse-distance mean of up to 3.")
    parser.add_argument("--publish-delay-min", type=int, default=60,
                        help="with --source stations: a day's file counts as finished only if written this "
                             "many minutes after 08:30 IST (default 60).")
    args = parser.parse_args()

    if args.last24h:
        run_last24h(args.farms, args.time, args.out, args.tp_unit, args.source,
                    args.max_km, args.method, args.publish_delay_min, args.show_hourly)
        sys.exit(0)

    if args.date:
        run(target_date=args.date)
    else:
        run_auto(allow_fallback=getattr(config, "RAIN_FALLBACK_OPENMETEO", True) and not args.no_fallback)

    print("\nNext step: run scripts/03_build_lag_features.py")