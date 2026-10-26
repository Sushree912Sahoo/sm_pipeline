# Soil moisture pipeline: SMAP 9 km -> lag model -> 70 m downscaling (Odisha)

```
01_raw_data/    smap/ (+tif/)   ecostress/   rainfall/   temperature/
02_lag_adjusted_sm/   lag features, predicted 9 km SMAP (parquet + GeoTIFF)
03_downscaled_sm/     sm_70m_<date>.tif  <- final output
models/lag/XGBoost.pkl        models/downscale/RF.pkl
static/   one-time 70 m layer      scripts/   logs/   reference/
```

## One-time setup
1. `pip install -r requirements.txt` (scikit-learn is pinned to 1.7.2, the version RF.pkl was trained with).
2. Put `XGBoost.pkl` in `models/lag/` and `RF.pkl` in `models/downscale/`.
3. Put `RR_Reference_Coordinates.csv` in `reference/`.
4. Set `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` as environment variables.
5. Put your training parquet (file or folder) in `static/source/`, or set `STATIC_SOURCE`.
6. Download one SMAP file (`python scripts/01_download_smap.py`), then build the static layer **once**:
   `python scripts/00_build_static_layer.py`

## Daily run
`python run_pipeline.py` (options: `--skip-download`, `--skip-scrape`, `--anchor YYYY-MM-DD`)

| Step | Script | Result |
|---|---|---|
| 1 | 01_download_smap.py | SMAP .h5 + float32 GeoTIFF (real m3/m3 values, nodata -9999) |
| 2 | 02_scrape_rainfall_temp.py | rainfall_<date>.csv and temperature_<date>.csv |
| 3 | 03_build_lag_features.py | the 11 lag features per 9 km cell |
| 4 | 04_lag_predict.py | predicted 9 km SMAP for the target date |
| 5 | 05_downscale_to_70m.py | 70 m soil moisture GeoTIFF |

## How the date works
The lag model predicts D+4 from the SMAP value on day D and rainfall/temperature on D..D+3.
Step 03 picks the latest D that has a SMAP file AND all four rainfall+temperature days.
With daily scraping that is today minus 4, so the output is for **today**.

## Things to know
- **Rainfall needs daily scraping.** rainfall.nic.in may reject past dates (its date field is
  `disable_past_dates`). The scraper now refuses to save a CSV when the site ignores the requested
  date, instead of saving it under the wrong date. If backfill is refused, the first prediction
  appears after 4 consecutive daily runs.
- **Day-of-year** uses `2*pi*doy/365.25` for both models (the lag formula from your script). If the
  downscaler's training used another formula, change `DOY_PERIOD` / `build_model_inputs`.
- **Missing soil texture** (about 2.5% of points) gives all-zero soil inputs, which is what a
  one-hot built from the class list does. Confirm this matches your training.
- **LULC** uses `LULC_2025` (latest year in the table), set in `config.py`.
- **Nodata at 70 m** appears where a 9 km cell has no valid SMAP on day D or no rainfall/temperature
  match within `NN_TOLERANCE_DEG`. Steps 03 and 05 print these counts.
- **ECOSTRESS** is not needed in the daily run. The 70 m grid comes from your training table.
- Tested end to end on synthetic data with your real model files. Steps 01 and 02 (network,
  Selenium) were not run here.



# SOIL MOISTURE DATA PREDICTION

## 1. Project Objective

The objective of this project is to generate a **70 m soil-moisture prediction map for Odisha, India** using:

* SMAP satellite soil-moisture data
* Rainfall data
* Temperature data
* XGBoost for lag adjustment / future soil-moisture prediction
* Random Forest for downscaling from approximately 9 km to 70 m
* LULC (Land Use/Land Cover)
* Soil texture information

The final output is a **70 m soil-moisture GeoTIFF**.

---

# 2. Project Location

The complete project is located at:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline
```

The project directory contains:

```text
sm_pipeline/
│
├── 01_raw_data/
│   ├── smap/
│   │   └── tif/
│   ├── rainfall/
│   └── temperature/
│
├── 02_lag_adjusted_sm/
│
├── 03_downscaled_sm/
│
├── models/
│   ├── lag/
│   │   └── XGBoost.pkl
│   └── downscale/
│       └── RF.pkl
│
├── reference/
│   └── RR_Reference_Coordinates.csv
│
├── static/
│   └── source/
│       └── ECOSTRESS_SMAP_SM_Odisha_2018_2026_with_LULC2_SOILTEX.parquet
│
├── scripts/
│   ├── 00_build_static_layer.py
│   ├── 01_download_smap.py
│   ├── 02_scrape_rainfall_temp.py
│   ├── 03_build_lag_features.py
│   ├── 04_lag_predict.py
│   ├── 05_downscale_to_70m.py
│   └── h5_to_tiff.py
│
├── logs/
├── config.py
├── requirements.txt
├── README.md
└── run_pipeline.py
```

---

# 3. Conda Environment Setup

The project uses **Anaconda Prompt** on Windows.

Conda version:

```text
conda 25.11.1
```

A dedicated Python environment was created.

### Create environment

```cmd
conda create -n sm_pipeline python=3.11
```

### Activate environment

```cmd
conda activate sm_pipeline
```

Python version used:

```text
Python 3.11.16
```

The command prompt should show:

```text
(sm_pipeline) D:\IIT BBSR\Suraj Sir work\sm_pipeline>
```

---

# 4. Install Required Packages

From the project directory:

```cmd
python -m pip install -r requirements.txt
```

Important packages installed include:

```text
scikit-learn
numpy
pandas
scipy
h5py
rasterio
pyarrow
joblib
xgboost
requests
beautifulsoup4
selenium
```

The installed scikit-learn version was:

```text
scikit-learn 1.7.2
```

---

# 5. Important Configuration

The main configuration file is:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\config.py
```

Important configuration values include:

```text
SMAP product:
SPL4SMAU

SMAP dataset:
Analysis_Data/sm_surface_analysis

SMAP latitude:
cell_lat

SMAP longitude:
cell_lon

SMAP CRS:
EPSG:6933

SMAP fill value:
-9999

LOOKBACK_DAYS:
5

MAX_LOOKBACK_DAYS:
60

LAG_DAYS:
4

NODATA:
-9999
```

Odisha bounding box:

```text
Latitude:
17.5 to 23.0

Longitude:
81.0 to 87.5
```

---

# 6. Earthdata Credentials

The SMAP data are downloaded using NASA Earthdata credentials.

The credentials are stored as environment variables.

In Anaconda Prompt:

```cmd
set EARTHDATA_USERNAME=YOUR_USERNAME
```

```cmd
set EARTHDATA_PASSWORD=YOUR_PASSWORD
```

The password should **never be written in the README or shared publicly**.

To safely verify that the credentials are available:

```cmd
python -c "import os; print('Username:', os.environ.get('EARTHDATA_USERNAME')); print('Password set:', bool(os.environ.get('EARTHDATA_PASSWORD')))"
```

Expected output:

```text
Username: YOUR_USERNAME
Password set: True
```

Important:

The `set` command applies to the **current Anaconda Prompt session**. If the terminal is closed and reopened, the variables need to be set again unless a persistent credential method is configured.

---

# 7. Models Used

Two pretrained models are already present.

## XGBoost Lag Model

Location:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\models\lag\XGBoost.pkl
```

Size:

```text
79,693,190 bytes
```

Purpose:

Predict future SMAP soil moisture using the latest available SMAP soil moisture and weather information.

---

## Random Forest Downscaling Model

Location:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\models\downscale\RF.pkl
```

Size:

```text
157,207,800 bytes
```

Purpose:

Downscale the approximately 9 km predicted soil moisture to approximately 70 m resolution.

---

# 8. Static Training Dataset

The large source Parquet file is located at:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\static\source\ECOSTRESS_SMAP_SM_Odisha_2018_2026_with_LULC2_SOILTEX.parquet
```

It contains approximately:

```text
27,501,463 points
```

The columns used to construct the static layer are:

```text
lat
lon
LULC_2025
soil_tex_0cm
```

---

# 9. One-Time Static Layer Creation

The static layer is required by the 70 m downscaling stage.

Run:

```cmd
python scripts\00_build_static_layer.py
```

The script:

1. Reads the required columns from the large Parquet dataset.
2. Derives the 70 m grid.
3. Reads the SMAP grid.
4. Groups points into unique combinations of:

   * SMAP cell
   * LULC
   * Soil texture
5. Creates the static layer.

Actual result:

```text
27,501,463 points
```

70 m grid:

```text
6,801 rows x 8,714 columns
```

Grid spacing:

```text
0.0007000 degrees
```

Unique combinations:

```text
30,953
```

Points with no soil texture:

```text
684,906
```

Percentage:

```text
2.49%
```

Static layer saved to:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\static
```

This step is normally required **only once**.

---

# 10. Daily Pipeline — Step 1: Download SMAP

Command:

```cmd
python scripts\01_download_smap.py
```

The script searches backward for the latest available SMAP `T090000` file.

It downloads:

* Latest available SMAP file
* Previous SMAP files required for lag/history processing

Example files obtained during testing:

```text
SMAP_L4_SM_aup_20260930T090000_Vv8011_001.h5
SMAP_L4_SM_aup_20261001T090000_Vv8011_001.h5
SMAP_L4_SM_aup_20261002T090000_Vv8011_001.h5
SMAP_L4_SM_aup_20261003T090000_Vv8011_001.h5
SMAP_L4_SM_aup_20261004T090000_Vv8011_001.h5
```

Stored in:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\01_raw_data\smap
```

The latest successful anchor SMAP date in this run was:

```text
2026-10-04
```

The script also converts the latest SMAP file to GeoTIFF.

Example:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\01_raw_data\smap\tif
```

---

# 11. Daily Pipeline — Step 2: Rainfall and Temperature

Command:

```cmd
python scripts\02_scrape_rainfall_temp.py
```

This collects the rainfall and temperature information required by the lag model.

Output directories:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\01_raw_data\rainfall
```

and:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\01_raw_data\temperature
```

The lag model requires weather data for:

```text
D
D+1
D+2
D+3
```

For the successful run:

```text
D = 2026-10-04
```

Therefore the required weather dates were:

```text
2026-10-04
2026-10-05
2026-10-06
2026-10-07
```

---

# 12. Daily Pipeline — Step 3: Build Lag Features

Command:

```cmd
python scripts\03_build_lag_features.py
```

The successful run produced:

```text
Anchor SMAP date D : 2026-10-04
Target date T=D+4  : 2026-10-08
```

Rainfall and temperature window:

```text
2026-10-04
2026-10-05
2026-10-06
2026-10-07
```

SMAP cells with valid value:

```text
4,241
```

Weather matching:

```text
2026-10-04:
rainfall 1,911/4,241
temperature 1,911/4,241

2026-10-05:
rainfall 1,911/4,241
temperature 1,911/4,241

2026-10-06:
rainfall 1,911/4,241
temperature 1,909/4,241

2026-10-07:
rainfall 1,911/4,241
temperature 1,911/4,241
```

Final cells containing all 11 required features:

```text
1,909 / 4,241
```

Output:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\02_lag_adjusted_sm\lag_features_2026-10-08.parquet
```

---

# 13. XGBoost Lag Prediction

Command:

```cmd
python scripts\04_lag_predict.py
```

The model predicted soil moisture for:

```text
1,909 SMAP cells
```

Target date:

```text
2026-10-08
```

Predicted SMAP statistics:

```text
Minimum = 0.169
Mean    = 0.263
Maximum = 0.467
```

Output Parquet:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\02_lag_adjusted_sm\lag_smap_pred_2026-10-08.parquet
```

Output GeoTIFF:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\02_lag_adjusted_sm\lag_smap_pred_2026-10-08.tif
```

---

# 14. Random Forest Downscaling to 70 m

Command:

```cmd
python scripts\05_downscale_to_70m.py
```

The script uses:

* Predicted 9 km SMAP soil moisture
* LULC
* Soil texture
* Day-of-year features
* Pretrained Random Forest model

The successful run produced:

```text
Target date:
2026-10-08
```

Predicted 9 km combinations:

```text
28,251 / 30,953
```

70 m pixels predicted:

```text
25,188,107 / 27,501,463
```

NoData pixels:

```text
2,313,356
```

Final soil-moisture statistics:

```text
Minimum = 0.281
Mean    = 0.333
Maximum = 0.392
```

Final GeoTIFF:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\03_downscaled_sm\sm_70m_2026-10-08.tif
```

---

# 15. Complete Processing Flow

The complete processing chain is:

```text
SMAP Satellite Data
        │
        ▼
01_download_smap.py
        │
        ▼
Raw SMAP H5
        │
        ├───────────────┐
        │               │
        ▼               ▼
Rainfall            Temperature
        │               │
        └───────┬───────┘
                ▼
03_build_lag_features.py
                │
                ▼
11 Lag Features
                │
                ▼
04_lag_predict.py
                │
                ▼
Future 9 km Soil Moisture
                │
                ▼
05_downscale_to_70m.py
                │
       ┌────────┴────────┐
       │                 │
      LULC          Soil Texture
       │                 │
       └────────┬────────┘
                ▼
        Random Forest
                │
                ▼
        70 m Soil Moisture
                │
                ▼
        Final GeoTIFF
```

---

# 16. Daily Execution Order

After the one-time static layer has been created, the normal sequence is:

```cmd
conda activate sm_pipeline
```

Then:

```cmd
python scripts\01_download_smap.py
```

```cmd
python scripts\02_scrape_rainfall_temp.py
```

```cmd
python scripts\03_build_lag_features.py
```

```cmd
python scripts\04_lag_predict.py
```

```cmd
python scripts\05_downscale_to_70m.py
```

The final output will be saved under:

```text
03_downscaled_sm\
```

with a date-based filename such as:

```text
sm_70m_YYYY-MM-DD.tif
```

---

# 17. Automatic Pipeline

The project also contains:

```text
run_pipeline.py
```

After verifying the individual scripts, the entire workflow can be executed using:

```cmd
python run_pipeline.py
```

This is intended to automate the daily sequence.

---

# 18. Final Output

The main final product is a:

```text
70 m Soil Moisture GeoTIFF
```

Example:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\03_downscaled_sm\sm_70m_2026-10-08.tif
```

The final GeoTIFF uses:

```text
NoData = -9999
```

---

# 19. Important Notes

### Earthdata credentials

Earthdata credentials must be available before running:

```cmd
python scripts\01_download_smap.py
```

If the terminal is closed, the temporary Windows `set` environment variables are lost.

---

### Static layer

The following command is normally a **one-time operation**:

```cmd
python scripts\00_build_static_layer.py
```

It should not be rebuilt every day unless the source static dataset or grid needs to be changed.

---

### Rainfall and temperature

The lag model requires four consecutive weather dates:

```text
D, D+1, D+2, D+3
```

The predicted target date is:

```text
T = D + 4
```

Example:

```text
Latest SMAP:
2026-10-04

Weather:
2026-10-04
2026-10-05
2026-10-06
2026-10-07

Prediction:
2026-10-08
```

---

# 20. Successful Test Run Summary

The complete pipeline was successfully tested.

```text
Anchor SMAP date:
2026-10-04

Prediction date:
2026-10-08

Valid SMAP cells:
4,241

Cells with all lag features:
1,909

9 km predicted cells:
1,909

70 m pixels predicted:
25,188,107

70 m NoData pixels:
2,313,356

Final SM minimum:
0.281

Final SM mean:
0.333

Final SM maximum:
0.392
```

Final output:

```text
D:\IIT BBSR\Suraj Sir work\sm_pipeline\03_downscaled_sm\sm_70m_2026-10-08.tif
```

---

# 21. Quick Command Reference

For future runs:

```cmd
cd /d D:\IIT BBSR\Suraj Sir work\sm_pipeline
```

```cmd
conda activate sm_pipeline
```

```cmd
python scripts\01_download_smap.py
```

```cmd
python scripts\02_scrape_rainfall_temp.py
```

```cmd
python scripts\03_build_lag_features.py
```

```cmd
python scripts\04_lag_predict.py
```

```cmd
python scripts\05_downscale_to_70m.py
```

Or, after confirming the automated runner:

```cmd
python run_pipeline.py
```
