"""
================================================================================
irrigation_data.py  -  everything that READS data for the irrigation decisions

  tables      crop table, Ea table, geology K table (read once and cached)
  PointLookup nearest 70 m database point -> sand, silt, clay, slope_pct, geology
  rasters     today's 70 m soil moisture at a point
  advice_for_farm(...)   ONE farm: snap to the nearest point, read soil / SM / geology / slope /
                         groundwater, run decision1, decision2, decision3 (irrigation_core.py) and
                         return one result dictionary.  The batch script (08) and the dashboard
                         both call THIS function, so their numbers are identical.

No formulas in this file; they all live in irrigation_core.py.

DASHBOARD EXAMPLE
    from irrigation_data import Context, advice_for_farm
    ctx = Context()                                   # loads tables + lookup once
    res = advice_for_farm({"lat": 20.3, "lon": 85.8, "crop": "Tomato", "irrigation_type": "furrow",
                           "area_ha": 0.5, "well_depth_m": 40, "well_radius_m": 0.075,
                           "furrow_width_m": 1.0, "furrow_length_m": 50},
                          day=datetime.date(2026, 10, 22), ctx=ctx, tp_mm=0.0)
    # res["decision1"], res["decision2"], res["decision3"], res["warnings"]
================================================================================
"""
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config                      # noqa: E402
import irrigation_core as core     # noqa: E402

MAX_SNAP_M = 200.0            # warn if the nearest database point is farther than this
MAX_SNAP_REFUSE_M = 1000.0    # beyond this the farm is outside coverage -> no advice at all
_AUTO = object()


# ----------------------------------------------------------------------------- #
# TABLES
# ----------------------------------------------------------------------------- #
def load_crop_table(path=config.CROP_TABLE_PATH):
    df = pd.read_csv(path)
    df["_key"] = df["crop_name"].str.strip().str.lower()
    alias_map = {}
    for _, r in df.iterrows():
        alias_map[r["_key"]] = r["_key"]
        if isinstance(r.get("aliases"), str):
            for a in r["aliases"].split("|"):
                alias_map[a.strip().lower()] = r["_key"]
    return df.set_index("_key"), alias_map


def find_crop(crop_name, crops, alias_map):
    key = alias_map.get(str(crop_name).strip().lower())
    if key is None:
        raise KeyError(f"Crop '{crop_name}' not in crop_table.csv (add it or add an alias). "
                       f"Known: {sorted(crops.index)}")
    return crops.loc[key]


def load_efficiency(path=config.EFFICIENCY_TABLE_PATH):
    """{'flood': 0.5, 'furrow': 0.6, 'sprinkler': 0.75}  - only the rows with kind = application."""
    df = pd.read_csv(path)
    df = df[df["kind"].str.strip().str.lower() == "application"]
    return {str(n).strip().lower(): float(e) for n, e in zip(df["name"], df["efficiency"])}


@lru_cache(maxsize=4)
def load_geology_k(path=config.GEOLOGY_K_XLSX):
    """{class number: (geology name, K in m/day or None)} from the column 'Ksat (m/day)'.
    The 'Alt Ksat (m/day)' column is NOT used. Classes without a value (water, ice) give K = None."""
    df = pd.read_excel(path, sheet_name="Sheet1", usecols=["Class", "Geology", "Ksat (m/day)"])
    out = {}
    for c, g, k in zip(df["Class"], df["Geology"], df["Ksat (m/day)"]):
        if pd.isna(c):
            continue
        out[int(c)] = (str(g), None if pd.isna(k) else float(k))
    return out


# ----------------------------------------------------------------------------- #
# NEAREST-POINT LOOKUP (grid is regular, file is lat-sorted -> no tree needed)
# ----------------------------------------------------------------------------- #
class PointLookup:
    M_PER_DEG = 111_320.0

    def __init__(self, path=config.POINT_LOOKUP_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found - run scripts/06_build_point_lookup.py first")
        self.df = pd.read_parquet(path)
        self.lat = self.df["lat"].to_numpy()
        self.lon = self.df["lon"].to_numpy()
        self.has_slope = "slope_pct" in self.df.columns
        self.has_geology = "geology" in self.df.columns

    def nearest(self, lat, lon):
        """Exact match is returned as-is (distance 0). Returns (row, distance_m)."""
        cos_lat = math.cos(math.radians(lat))
        for band in (0.005, 0.02, 0.1, None):
            if band is None:
                lo, hi = 0, len(self.lat)
            else:
                lo = np.searchsorted(self.lat, lat - band, "left")
                hi = np.searchsorted(self.lat, lat + band, "right")
            if hi <= lo:
                continue
            dy = (self.lat[lo:hi] - lat) * self.M_PER_DEG
            dx = (self.lon[lo:hi] - lon) * self.M_PER_DEG * cos_lat
            d = np.hypot(dx, dy)
            i = int(np.argmin(d))
            if band is None or d[i] <= band * self.M_PER_DEG:
                return self.df.iloc[lo + i], float(d[i])
        raise RuntimeError("lookup table is empty")


# ----------------------------------------------------------------------------- #
# 70 m SOIL MOISTURE
# ----------------------------------------------------------------------------- #
def find_sm_file(day):
    sm_dir = Path(config.DOWNSCALED_DIR)
    for pat in (f"sm_70m_{day:%Y%m%d}.tif", f"sm_70m_{day:%Y-%m-%d}.tif"):
        if (sm_dir / pat).exists():
            return sm_dir / pat
    hits = [p for p in sm_dir.glob(f"sm_70m_*{day:%Y%m%d}*.tif") if "QA" not in p.name]
    if hits:
        return hits[0]
    avail = sorted(p.name for p in sm_dir.glob("sm_70m_*.tif") if "QA" not in p.name)[-5:]
    raise FileNotFoundError(f"No 70 m soil-moisture map for {day:%Y-%m-%d} in {sm_dir}. "
                            f"Latest available: {avail}")


def latest_map_date(today=None):
    """Date (datetime.date) of the newest 70 m map that is NOT after today (IST). None if there is none."""
    import re
    from datetime import datetime, timedelta, timezone
    today = today or datetime.now(timezone(timedelta(hours=5, minutes=30))).date()
    best = None
    for f in Path(config.DOWNSCALED_DIR).glob("sm_70m_*.tif"):
        m = re.search(r"sm_70m_(\d{4})-?(\d{2})-?(\d{2})\.tif$", f.name)
        if m:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
            if d <= today and (best is None or d > best):
                best = d
    return best


def sample_raster(tif_path, lons, lats):
    import rasterio
    from rasterio.warp import transform as warp_transform
    with rasterio.open(tif_path) as src:
        xs, ys = list(lons), list(lats)
        if src.crs and src.crs.to_epsg() != 4326:
            xs, ys = warp_transform("EPSG:4326", src.crs, xs, ys)
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype="float64")
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    return vals


# ----------------------------------------------------------------------------- #
# CONTEXT = everything loaded once
# ----------------------------------------------------------------------------- #
def _load_script(filename, modname):
    spec = importlib.util.spec_from_file_location(modname, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Context:
    def __init__(self):
        self.crops, self.alias_map = load_crop_table()
        self.ea = load_efficiency()
        self.k_table = load_geology_k()
        self.lookup = PointLookup()
        self._gw = None

    def predict_gw(self, lat, lon, day, tp_mm):
        """Groundwater for ONE farm (dashboard): uses GwPredictor from 07_gw_predict.py."""
        if self._gw is None:
            self._gw = _load_script("07_gw_predict.py", "gw_predict_07").GwPredictor()
        r = self._gw.predict(lat, lon, day, tp_mm)
        return dict(gwl_depth_m=r["gwl_depth_m"], gwe_pred=r["gwe_pred"],
                    ground_elevation_amsl=r.get("ground_elevation_amsl"),
                    warnings="; ".join(r["warnings"]))


# ----------------------------------------------------------------------------- #
# ONE FARM  -> one result dictionary
# ----------------------------------------------------------------------------- #
def _get(farm, key):
    v = farm.get(key)
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v


def advice_for_farm(farm, day, ctx, sm_value=_AUTO, sm_qa=None, point=None, snap_m=None,
                    gw=None, tp_mm=None):
    """Run the three decisions for one farm.

    farm      dict: farm_id (optional), lat, lon, crop, irrigation_type, area_ha, well_depth_m,
              well_radius_m, furrow_width_m, furrow_length_m, sprinkler_discharge_ls,
              sprinkler_spacing_m, lateral_spacing_m   (only the ones the chosen type needs)
    day       datetime.date or datetime (the data date)
    ctx       Context()
    sm_value  70 m soil moisture (m3/m3). Leave out to read it from the raster for `day`.
    point, snap_m   optional pre-computed nearest point (the batch script does this once per farm)
    gw        dict with gwl_depth_m, gwe_pred, ground_elevation_amsl (from 07). Or give tp_mm
              (rainfall of the last 24 h) and the groundwater model runs for this farm.
    Raises KeyError for an unknown crop and ValueError for a bad lat/lon (the caller turns that into
    status ERROR for that one farm).
    """
    farm_id = _get(farm, "farm_id") or "farm"
    date_s = f"{day:%Y-%m-%d}"
    crop_in, type_in = _get(farm, "crop"), _get(farm, "irrigation_type")
    type_key = core.normalize_irrigation_type(type_in)
    type_out = type_key or type_in
    try:
        lat, lon = float(farm["lat"]), float(farm["lon"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("lat and lon are missing or not numbers")
    if point is None:
        point, snap_m = ctx.lookup.nearest(lat, lon)

    meta = dict(lookup_lat=round(float(point["lat"]), 6), lookup_lon=round(float(point["lon"]), 6),
                snap_distance_m=round(float(snap_m), 1),
                sm_qa=None if sm_qa is None or pd.isna(sm_qa) else float(sm_qa))
    warns = []
    if snap_m > MAX_SNAP_REFUSE_M:
        return core.assemble_result(
            farm_id, date_s, crop_in, type_out, status="OUT_OF_COVERAGE", meta=meta,
            extra_warnings=[f"nearest database point is {snap_m:.0f} m away (more than "
                            f"{MAX_SNAP_REFUSE_M:.0f} m): the farm is outside the soil-moisture grid, "
                            f"no advice given"])
    if snap_m > MAX_SNAP_M:
        warns.append(f"nearest database point is {snap_m:.0f} m away (more than {MAX_SNAP_M:.0f} m)")

    # --- soil ------------------------------------------------------------------
    sand, silt, clay = (core.num(point.get(k)) for k in ("sand", "silt", "clay"))
    if None not in (sand, silt, clay) and not 90 <= sand + silt + clay <= 110:
        warns.append(f"sand + silt + clay = {sand + silt + clay:.0f}% (expected about 100)")

    # --- 70 m soil moisture --------------------------------------------------------
    if sm_value is _AUTO:
        sm_value = float(sample_raster(find_sm_file(day), [point["lon"]], [point["lat"]])[0])
    sm = core.num(sm_value)
    if sm is None:
        warns.append("soil moisture is nodata at this point")
    elif not 0.0 <= sm <= 1.0:
        warns.append(f"soil moisture value {sm:.3g} is outside 0 to 1 m3/m3 and is not used")
        sm = None

    # --- crop (KeyError -> status ERROR for this farm) ---------------------------------
    crop = find_crop(crop_in, ctx.crops, ctx.alias_map)
    zr_m, mad = float(crop["zr_max_m"]), float(crop["p"])

    d1 = core.decision1(sm, zr_m, mad, sand=sand, silt=silt, clay=clay)
    d2 = core.decision2(sm, d1["theta_fc_pct"], zr_m, type_in, ctx.ea)

    # --- geology -> K, slope --------------------------------------------------------------
    k = None
    code = point.get("geology") if ctx.lookup.has_geology else None
    if not ctx.lookup.has_geology:
        warns.append("point_lookup.parquet has no geology column: re-run scripts/06_build_point_lookup.py")
    elif code is None or pd.isna(code):
        warns.append("no geology code at this point, so no K")
    else:
        code = int(code)
        if code not in ctx.k_table:
            warns.append(f"geology code {code} is not in GeologyClass_Ksat.xlsx, so no K")
        else:
            name, k = ctx.k_table[code]
            if k is None:
                warns.append(f"geology class {code} ({name}) has no K value in GeologyClass_Ksat.xlsx")
    slope = None
    if not ctx.lookup.has_slope:
        warns.append("point_lookup.parquet has no slope_pct column: re-run scripts/06_build_point_lookup.py")
    else:
        slope = core.num(point.get("slope_pct"))

    # --- groundwater (from step 07, or predicted now for one farm) -------------------------
    if gw is None and tp_mm is not None:
        gw = ctx.predict_gw(lat, lon, day, tp_mm)
    gwe = gse = None
    if gw is None:
        warns.append("no groundwater prediction for this farm, so well discharge is not calculated")
    else:
        gse = core.num(gw.get("ground_elevation_amsl"))
        if core.num(gw.get("gwl_depth_m")) is None:
            gwe = None
            warns.append("groundwater prediction is not usable for this farm (see the groundwater "
                         "step's message), so well discharge is not calculated")
        else:
            gwe = core.num(gw.get("gwe_pred"))
        gw_note = gw.get("warnings")
        if isinstance(gw_note, str) and gw_note.strip():
            warns.append("groundwater step: " + gw_note.strip())

    d3 = core.decision3(
        gwe, gse, k, _get(farm, "well_depth_m"), _get(farm, "well_radius_m"), type_in,
        d2["d_gross_mm"], area_ha=_get(farm, "area_ha"),
        furrow_width_m=_get(farm, "furrow_width_m"), furrow_length_m=_get(farm, "furrow_length_m"),
        slope_pct=slope, sprinkler_discharge_ls=_get(farm, "sprinkler_discharge_ls"),
        sprinkler_spacing_m=_get(farm, "sprinkler_spacing_m"),
        lateral_spacing_m=_get(farm, "lateral_spacing_m"), t_p_max_hr=config.T_P_MAX_HR)

    return core.assemble_result(farm_id, date_s, crop_in, type_out, d1, d2, d3,
                                status="OK" if sm is not None else "NO_SM_DATA",
                                extra_warnings=warns, meta=meta)
