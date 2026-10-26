"""
h5_to_tiff.py
--------------
Reads a SMAP L4 (SPL4SMAU) .h5 granule and writes a float32 GeoTIFF with
the REAL soil moisture values (m3/m3), nodata = -9999, on the true
EASE-Grid 2.0 (EPSG:6933) 9 km grid, cropped to Odisha + a small margin.

Replaces the earlier 0-255 normalised converter: the models need real
values, and 0 must not mean both "dry" and "missing".

The georeferencing is derived from the file's own cell_lat / cell_lon
arrays (projected to EPSG:6933), so nothing about the grid is assumed.

Also provides read_smap_grid(), used by the lag and downscale steps so
every script crops and numbers the grid cells in exactly the same way.

Standalone use:
    python scripts/h5_to_tiff.py path/to/file.h5
    python scripts/h5_to_tiff.py            (latest file in 01_raw_data/smap)
"""

import os
import re
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import h5py
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform


def extract_date(path):
    """'YYYY-MM-DD' from the SMAP filename (YYYYMMDDThhmmss block), or None."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})T\d{6}", os.path.basename(path))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def get_latest_h5_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".h5")]
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {directory}")
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
    return os.path.join(directory, files[0])


def read_smap_grid(h5_path, margin=None):
    """
    Reads one SMAP granule and crops it to Odisha (+ margin).

    Returns a dict:
      sm          2D float32 soil moisture of the crop (NaN = missing)
      lat, lon    2D cell-centre coordinates of the crop
      r0, c0      row/col of the crop's top-left cell in the FULL grid
      nrows_full, ncols_full   shape of the full grid
      transform, crs           georeferencing of the crop (EPSG:6933)
      xs, ys      1D projected cell-centre coordinates (columns / rows)
      res_x, res_y  cell size in metres
    A cell's global id is  (r0 + i) * ncols_full + (c0 + j).
    """
    if margin is None:
        margin = config.CROP_MARGIN_DEG

    with h5py.File(h5_path, "r") as f:
        sm = f[config.SM_DATASET][:].astype(np.float32)
        lat = f[config.LAT_DATASET][:]
        lon = f[config.LON_DATASET][:]

    sm[sm == config.SM_FILL_VALUE] = np.nan
    nrows_full, ncols_full = sm.shape

    mask = (
        (lat >= config.LAT_MIN - margin) & (lat <= config.LAT_MAX + margin)
        & (lon >= config.LON_MIN - margin) & (lon <= config.LON_MAX + margin)
    )
    if not mask.any():
        raise ValueError("No SMAP cells fall inside the Odisha box; check config bounds / file.")
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()

    sm_c = sm[r0:r1 + 1, c0:c1 + 1]
    lat_c = lat[r0:r1 + 1, c0:c1 + 1]
    lon_c = lon[r0:r1 + 1, c0:c1 + 1]

    # EPSG:6933 is cylindrical: x depends only on longitude, y only on latitude,
    # so one row gives all column x's and one column gives all row y's.
    xs, _ = warp_transform("EPSG:4326", config.SMAP_CRS, lon_c[0, :].tolist(), lat_c[0, :].tolist())
    _, ys = warp_transform("EPSG:4326", config.SMAP_CRS, lon_c[:, 0].tolist(), lat_c[:, 0].tolist())
    xs, ys = np.asarray(xs), np.asarray(ys)

    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Crop is too small to derive the grid spacing.")
    res_x = float(np.median(np.diff(xs)))
    res_y = float(np.median(np.abs(np.diff(ys))))
    if ys[0] < ys[-1]:
        raise NotImplementedError("SMAP rows run south->north; expected north->south.")
    if np.ptp(np.diff(xs)) > 1e-3 * res_x or np.ptp(np.diff(ys)) > 1e-3 * res_y:
        warnings.warn("SMAP cell spacing is not uniform in EPSG:6933; georeferencing may be off.")

    transform = Affine(res_x, 0.0, xs[0] - res_x / 2, 0.0, -res_y, ys[0] + res_y / 2)

    return dict(sm=sm_c, lat=lat_c, lon=lon_c, r0=int(r0), c0=int(c0),
                nrows_full=int(nrows_full), ncols_full=int(ncols_full),
                transform=transform, crs=config.SMAP_CRS,
                xs=xs, ys=ys, res_x=res_x, res_y=res_y)


def write_float_geotiff(path, arr, transform, crs, nodata=None):
    """Writes a 2D array as a compressed float32 GeoTIFF (NaN -> nodata)."""
    if nodata is None:
        nodata = config.NODATA
    out = np.where(np.isnan(arr), nodata, arr).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=out.shape[0], width=out.shape[1], count=1,
        dtype="float32", crs=crs, transform=transform, nodata=nodata,
        compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(out, 1)
    return path


def convert_h5_to_tiff(h5_path):
    """h5 -> float32 GeoTIFF in config.SMAP_TIFF_DIR. Returns the path, or None on failure."""
    if not os.path.exists(h5_path):
        print(f"  [tiff] File does not exist: {h5_path}")
        return None
    try:
        g = read_smap_grid(h5_path)
        date = extract_date(h5_path) or "unknown_date"
        out = os.path.join(config.SMAP_TIFF_DIR, f"smap_sm_surface_analysis_{date}.tif")
        write_float_geotiff(out, g["sm"], g["transform"], g["crs"])
        valid = int(np.isfinite(g["sm"]).sum())
        print(f"  [tiff] Saved {out}  shape={g['sm'].shape}, valid cells={valid:,}")
        return out
    except Exception as e:
        print(f"  [tiff] Failed to convert {os.path.basename(h5_path)}: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_h5_to_tiff(sys.argv[1])
    else:
        convert_h5_to_tiff(get_latest_h5_file(config.SMAP_RAW_DIR))
