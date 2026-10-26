"""
config.py - one place for paths and settings. Every script imports this.

All paths are built from BASE_DIR (the project root). Set the environment
variable SM_PIPELINE_ROOT to move the whole project somewhere else.
"""
import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("SM_PIPELINE_ROOT", Path(__file__).resolve().parent))


def P(*parts):
    return str(BASE_DIR.joinpath(*parts))


CREDENTIALS_INFO = {"path": None, "exists": False, "keys_found": [], "error": None}


def _load_credentials_file():
    """Reads KEY=VALUE lines from credentials.env (in the project folder) so you never have
    to set the Earthdata login by hand. Tolerates spaces, quotes, 'set '/'export ' prefixes,
    UTF-16 files and empty pre-set variables. A real, non-empty environment variable wins."""
    # accept the usual Notepad mistake (credentials.env.txt) and a plain .env too
    candidates = [BASE_DIR / n for n in ("credentials.env", "credentials.env.txt", ".env")]
    path = next((c for c in candidates if c.exists()), candidates[0])
    CREDENTIALS_INFO.update(path=str(path), exists=path.exists())
    if not path.exists():
        return
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeError:
            continue
    if text is None:
        CREDENTIALS_INFO["error"] = "could not read the file's text encoding"
        return
    for line in text.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith(("export ", "set ")):
            line = line.split(None, 1)[1]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().upper(), value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        CREDENTIALS_INFO["keys_found"].append(key)
        if not os.environ.get(key):
            os.environ[key] = value


_load_credentials_file()


# --------------------------------------------------------------------
# EARTHDATA CREDENTIALS (read from credentials.env, or from environment variables)
# --------------------------------------------------------------------
EARTHDATA_USERNAME = os.environ.get("EARTHDATA_USERNAME")
EARTHDATA_PASSWORD = os.environ.get("EARTHDATA_PASSWORD")

# --------------------------------------------------------------------
# FOLDER 1 - RAW DATA
# --------------------------------------------------------------------
SMAP_RAW_DIR = P("01_raw_data", "smap")            # SMAP L4 .h5 files
SMAP_TIFF_DIR = P("01_raw_data", "smap", "tif")    # float32 GeoTIFF copies
ECOSTRESS_RAW_DIR = P("01_raw_data", "ecostress")  # only for one-off grid/reference use
RAINFALL_DIR = P("01_raw_data", "rainfall")        # rainfall_<date>.csv
TEMPERATURE_DIR = P("01_raw_data", "temperature")  # temperature_<date>.csv

# --------------------------------------------------------------------
# FOLDER 2 / FOLDER 3 - OUTPUTS
# --------------------------------------------------------------------
LAG_DIR = P("02_lag_adjusted_sm")        # lag features, 9 km predicted SMAP
DOWNSCALED_DIR = P("03_downscaled_sm")   # 70 m soil moisture GeoTIFFs

# --------------------------------------------------------------------
# MODELS, STATIC LAYER, LOGS
# --------------------------------------------------------------------
LAG_MODEL_PATH = P("models", "lag", "XGBoost.pkl")
DOWNSCALE_MODEL_PATH = P("models", "downscale", "RF.pkl")

STATIC_DIR = P("static")
# Parquet file (or folder of parquet files) that holds your 70 m training table.
# Only read ONCE by scripts/00_build_static_layer.py.
STATIC_SOURCE = os.environ.get("STATIC_SOURCE", P("static", "source"))
STATIC_LAT_COL = "lat"
STATIC_LON_COL = "lon"
STATIC_LULC_COL = "LULC_2025"        # latest year available in the table
STATIC_SOIL_COL = "soil_tex_0cm"     # null -> all-zero one-hot (see README)

LOGS_DIR = P("logs")
REFERENCE_COORDS_PATH = P("reference", "RR_Reference_Coordinates.csv")

# --------------------------------------------------------------------
# SMAP PRODUCT (SPL4SMAU, confirmed from your earlier config)
# --------------------------------------------------------------------
SHORT_NAME = "SPL4SMAU"
SM_DATASET = "Analysis_Data/sm_surface_analysis"
LAT_DATASET = "cell_lat"
LON_DATASET = "cell_lon"
SM_FILL_VALUE = -9999.0
SMAP_CRS = "EPSG:6933"       # EASE-Grid 2.0 global cylindrical equal-area

# How many days back to search for the latest SMAP file, and how many
# days before it to also download (used by 01_download_smap.py)
LOOKBACK_DAYS = 5
MAX_LOOKBACK_DAYS = 60
LAG_DAYS = 4

# --------------------------------------------------------------------
# ODISHA BOX + CROP MARGIN
# --------------------------------------------------------------------
LAT_MIN, LAT_MAX = 17.5, 23.0
LON_MIN, LON_MAX = 81.0, 87.5
# Extra margin so 70 m points near the box edge still find their 9 km cell
CROP_MARGIN_DEG = 0.15

# --------------------------------------------------------------------
# RAINFALL / TEMPERATURE SCRAPER
# --------------------------------------------------------------------
# Visible browser when you run by hand; the scheduled launcher (run_daily.bat) sets
# SM_HEADLESS=1 so the browser runs hidden and nobody has to be watching.
HEADLESS = os.environ.get("SM_HEADLESS", "0") == "1"
SELENIUM_URL = os.environ.get("SELENIUM_URL")   # e.g. http://selenium:4444/wd/hub
NN_TOLERANCE_DEG = 0.18

# --- Filling gaps instead of dropping them (so the 70 m map has no holes) ---
# Rainfall/temperature: if no station is within NN_TOLERANCE_DEG, use the nearest
# station up to this far away (degrees, ~111 km each). 0 = never use a farther one.
NN_FALLBACK_DEG = 0.6
# SMAP: an invalid 9 km cell on the anchor day is filled from the nearest valid cell
# up to this many cells away (0 = off).
SMAP_FILL_MAX_CELLS = 3
# Step 05: a 9 km cell with no lag prediction gets the nearest predicted cell's value,
# up to this many cells away (0 = off).
LAG_FILL_MAX_CELLS = 4

# If rainfall.nic.in refuses a PAST day (it may only serve the current day), fill
# that day from Open-Meteo instead. This is a different rainfall source than your
# training data (marked source=openmeteo in the CSV). Set False to never do this.
RAIN_FALLBACK_OPENMETEO = True

# --------------------------------------------------------------------
# LAG MODEL FEATURES (exact order of XGBoost.pkl's 11 inputs)
# --------------------------------------------------------------------
FEATURE_COLUMNS = [
    "sm_4_prior",
    "sum_rainfall_4", "sum_rainfall_3", "sum_rainfall_2", "sum_rainfall_1",
    "mean_temp_4", "mean_temp_3", "mean_temp_2", "mean_temp_1",
    "doy_sin", "doy_cos",
]
DOY_PERIOD = 365.25   # same formula as your lag-feature script
NODATA = -9999.0


# --------------------------------------------------------------------
# IRRIGATION ADVICE (step 08 and the dashboard)
# --------------------------------------------------------------------
# The ONE big parquet (about 27 million rows). Read by step 06 (soil, slope, geology)
# and step 07 (groundwater inputs). Change it here only.
INPUT_PARQUET = os.environ.get("INPUT_PARQUET", P("static", "source", "input.parquet"))

POINT_LOOKUP_PATH = P("static", "point_lookup.parquet")      # built by step 06
GW_LOOKUP_PATH = P("static", "gw_point_lookup.parquet")      # built by step 07

CROP_TABLE_PATH = P("reference", "crop_table.csv")
EFFICIENCY_TABLE_PATH = P("reference", "efficiency_table.csv")   # Ea (application rows)
GEOLOGY_K_XLSX = P("reference", "GeologyClass_Ksat.xlsx")        # column "Ksat (m/day)"
IRRIGATION_OUT_DIR = P("05_irrigation")

# Maximum pumping time per day (hr/day). Used by Decision 3 (flood, furrow).
T_P_MAX_HR = 6.0

# run_pipeline.py warns when the newest 70 m map is older than this many days.
STALE_MAP_DAYS = 3


# --------------------------------------------------------------------
# DASHBOARD
# --------------------------------------------------------------------
DASHBOARD_DB_PATH = P("dashboard", "data", "farms.db")     # saved farms (SQLite)
FARMS_CSV_PATH = P("farm_inputs", "farms.csv")             # the daily pipeline's farm list
