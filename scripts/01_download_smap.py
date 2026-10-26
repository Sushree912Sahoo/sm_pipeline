"""
01_download_smap.py
---------------------
Downloads SMAP L4 soil moisture granules for Odisha over the last
config.LOOKBACK_DAYS days, using the virtual-directory scraping approach
you've already confirmed works (adapted from your own working script) —
NOT the CMR JSON search API, which returned a 401 on the public search
endpoint when a Basic Auth session was attached to it.

Differences from your original script:
  - Credentials come from config.py (env vars), not a direct dotenv call —
    Docker Compose already injects EARTHDATA_USERNAME/PASSWORD from .env
    into the container's environment.
  - Saves flat into config.SMAP_RAW_DIR (no per-date subfolder) so
    03_update_history_and_build_features.py's glob can find the files.
  - Loops across the WHOLE lookback window and downloads every day it
    finds, instead of stopping at the first match — the lag features need
    several consecutive days of history, not just the single latest file.

Run from the project root:
    python scripts/01_download_smap.py
"""

import sys
import os
import re
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import h5_to_tiff

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        if "Authorization" in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(prepared_request.url)
            if (
                original_parsed.hostname != redirect_parsed.hostname
                and redirect_parsed.hostname != self.AUTH_HOST
                and original_parsed.hostname != self.AUTH_HOST
            ):
                del headers["Authorization"]
        return


def get_session():
    if not config.EARTHDATA_USERNAME or not config.EARTHDATA_PASSWORD:
        info = getattr(config, "CREDENTIALS_INFO", None)
        if info is None:
            detail = ("Your config.py is the OLD version (it cannot read credentials.env). "
                      "Replace it with the new config.py.")
        elif not info["exists"]:
            detail = f"There is no credentials.env at: {info['path']}"
        else:
            detail = (f"Found {info['path']} but the keys inside it are: {info['keys_found'] or 'none'}. "
                      f"Expected EARTHDATA_USERNAME and EARTHDATA_PASSWORD, each as KEY=VALUE on its own line."
                      + (f" Problem: {info['error']}." if info["error"] else ""))
        raise RuntimeError(
            "Earthdata login not found.\n" + detail + "\n"
            "credentials.env must be in the sm_pipeline folder with two lines:\n"
            "EARTHDATA_USERNAME=your_username\nEARTHDATA_PASSWORD=your_password"
        )
    return SessionWithHeaderRedirection(config.EARTHDATA_USERNAME, config.EARTHDATA_PASSWORD)


def _is_junk_href(href: str) -> bool:
    if not href:
        return True
    href = href.strip()
    return (
        href in ("../", "./", "/")
        or href.startswith("#")
        or href.startswith("?")
        or "C=" in href
    )


def find_target_h5_recursive(session, base_url, target_time, retry_count=3, max_depth=20, _depth=0):
    """Recursively searches an Earthdata virtual-directory listing for a
    .h5 file matching target_time, entering a same-named subfolder first
    if one exists. Unchanged from your working version."""
    if _depth > max_depth:
        return ""

    response = None
    for attempt in range(retry_count):
        try:
            response = session.get(base_url, timeout=20, allow_redirects=True)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == retry_count - 1:
                print(f"    Failed to access {base_url}: {e}")
                return ""

    soup = BeautifulSoup(response.text, "html.parser")
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    links = [h for h in links if not _is_junk_href(h)]

    time_dir_candidates = [h for h in links if h.rstrip("/").endswith(target_time) and h.endswith("/")]
    if time_dir_candidates:
        time_dir_url = urljoin(base_url, time_dir_candidates[0])
        result = find_target_h5_recursive(session, time_dir_url, target_time, retry_count, max_depth, _depth + 1)
        if result:
            return result

    for href in links:
        if href.endswith(".h5") and target_time in href:
            return urljoin(base_url, href)

    for href in links:
        if href.endswith("/"):
            next_url = urljoin(base_url, href)
            if urlparse(next_url).netloc and urlparse(next_url).netloc != urlparse(base_url).netloc:
                continue
            result = find_target_h5_recursive(session, next_url, target_time, retry_count, max_depth, _depth + 1)
            if result:
                return result

    return ""


def download_file(session, url, out_dir):
    """Downloads url into out_dir (flat — no date subfolder), skipping
    files already present."""
    os.makedirs(out_dir, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path):
        print(f"    Already have {filename}, skipping.")
        return out_path

    print(f"    Downloading {filename} ...")
    try:
        with session.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as fd:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fd.write(chunk)
        print(f"    Saved -> {out_path}")
        return out_path
    except requests.exceptions.RequestException as e:
        print(f"    Download failed: {e}")
        return None


def run():
    session = get_session()

    # Collection + target time from your working script:
    # SMAP L4 Global 3-hourly 9km Analysis Update (SPL4SMAU),
    # concept-id C3480440639-NSIDC_CPRD, 09:00 UTC snapshot.
    base = "https://cmr.earthdata.nasa.gov/virtual-directory/collections/C3480440639-NSIDC_CPRD/temporal"
    target_time = "T090000"

    # Step 1: find the LATEST available date, however far back that is
    # (up to config.MAX_LOOKBACK_DAYS), instead of giving up after a
    # fixed short window.
    print(f"Searching backward for the latest available {target_time} file "
          f"(up to {config.MAX_LOOKBACK_DAYS} days back)...")
    latest_date = None
    latest_link = None
    for i in range(config.MAX_LOOKBACK_DAYS):
        d = datetime.now() - timedelta(days=i)
        day_url = f"{base}/{d.year}/{d.month:02d}/{d.day:02d}/"
        print(f"  Checking {d.date()}...")
        link = find_target_h5_recursive(session, day_url, target_time)
        if link:
            print(f"  Found latest available: {d.date()} -> {link.rsplit('/', 1)[-1]}")
            latest_date = d
            latest_link = link
            break

    if latest_link is None:
        print(f"\nNo {target_time} file found in the last {config.MAX_LOOKBACK_DAYS} "
              f"days at all. This is unusual — worth checking the collection "
              f"directly on Earthdata Search for an outage or reprocessing notice.")
        return []

    downloaded = []
    path = download_file(session, latest_link, config.SMAP_RAW_DIR)
    if path:
        downloaded.append(path)

    # Step 2: also grab the LAG_DAYS days immediately before the latest
    # date found, so there's enough consecutive history for lag features
    # anchored at that date (rather than just one isolated day).
    print(f"\nFetching the {config.LAG_DAYS} day(s) before {latest_date.date()} "
          f"for lag-feature history...")
    for offset in range(1, config.LAG_DAYS + 1):
        d = latest_date - timedelta(days=offset)
        day_url = f"{base}/{d.year}/{d.month:02d}/{d.day:02d}/"
        print(f"  Checking {d.date()}...")
        link = find_target_h5_recursive(session, day_url, target_time)
        if link:
            path = download_file(session, link, config.SMAP_RAW_DIR)
            if path:
                downloaded.append(path)
        else:
            print(f"    No granule found for {d.date()} (gap in history).")

    print(f"\nDownloaded {len(downloaded)} file(s) total.")

    if latest_link is not None:
        print("\nConverting the latest downloaded file to GeoTIFF...")
        latest_local_path = os.path.join(config.SMAP_RAW_DIR, latest_link.rsplit("/", 1)[-1])
        h5_to_tiff.convert_h5_to_tiff(latest_local_path)

    return downloaded


if __name__ == "__main__":
    run()
    print("\nNext step: run scripts/02_scrape_rainfall_temp.py")
