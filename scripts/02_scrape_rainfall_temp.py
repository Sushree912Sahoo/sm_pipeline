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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch the rainfall + temperature days the lag model needs.")
    parser.add_argument("--date", type=str, default=None,
                        help="Fetch only this one day (YYYY-MM-DD), site only.")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Never use the Open-Meteo backup for rainfall.")
    args = parser.parse_args()

    if args.date:
        run(target_date=args.date)
    else:
        run_auto(allow_fallback=getattr(config, "RAIN_FALLBACK_OPENMETEO", True) and not args.no_fallback)

    print("\nNext step: run scripts/03_build_lag_features.py")