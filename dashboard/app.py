"""
================================================================================
Irrigation advice dashboard (Phase 2)          start it from the project folder:

        streamlit run dashboard\\app.py

A thin layer over the pipeline: it collects the farmer inputs, calls services.run_advice(...)
(which calls the same advice_for_farm(...) as 08_irrigation_advice.py) and SHOWS the three decisions.
There are NO formulas in this file: only layout, number formatting and saving of farms.
================================================================================
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT / "scripts"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
import streamlit as st              # noqa: E402

import config                       # noqa: E402
import farm_store                   # noqa: E402
import irrigation_core as core      # noqa: E402
import irrigation_data as data      # noqa: E402
import mapview                      # noqa: E402
import services                     # noqa: E402

st.set_page_config(page_title="Irrigation advice", page_icon="💧", layout="wide")

TYPES = {"Surface irrigation (flood)": "flood", "Furrow irrigation": "furrow",
         "Sprinkler irrigation": "sprinkler"}
TYPE_LABEL = {v: k for k, v in TYPES.items()}

# session key -> farm column
KEY_TO_COL = {"farm_id": "farm_id", "lat": "lat", "lon": "lon", "crop": "crop", "area_ha": "area_ha",
              "well_depth_m": "well_depth_m", "well_radius_m": "well_radius_m",
              "furrow_width_m": "furrow_width_m", "furrow_length_m": "furrow_length_m",
              "spr_q": "sprinkler_discharge_ls", "spr_sp": "sprinkler_spacing_m",
              "lat_sp": "lateral_spacing_m"}

# every value shown on the cards: key -> (label, unit)
LABELS = {
    "sm_pct": ("Soil moisture today (70 m)", "% vol"),
    "theta_trigger_pct": ("Pump-on trigger", "% vol"),
    "theta_fc_pct": ("Field capacity", "% vol"), "theta_wp_pct": ("Wilting point", "% vol"),
    "zr_m": ("Root depth (Zr)", "m"), "mad": ("Depletion factor (MAD)", ""),
    "awc_mm": ("Available water capacity", "mm"),
    "taw_mm": ("Total available water (TAW)", "mm"), "raw_mm": ("Readily available water (RAW)", "mm"),
    "d_net_mm": ("Net irrigation depth", "mm"), "d_gross_mm": ("Gross irrigation depth", "mm"),
    "ea": ("Application efficiency (Ea) used", ""),
    "gwe_m_amsl": ("Groundwater level (GWE)", "m above sea level"),
    "gse_m_amsl": ("Ground surface (GSE)", "m above sea level"),
    "well_bottom_m_amsl": ("Well bottom", "m above sea level"),
    "h_m": ("Water column in well (H)", "m"), "hw_m": ("Water depth at well when pumping (hw)", "m"),
    "ro_m": ("Radius of influence (Ro)", "m"), "k_m_day": ("Hydraulic conductivity (K)", "m/day"),
    "slope_pct": ("Land slope", "%"),
    "q_m3_day": ("Well discharge (Q)", "m³/day"), "q_m3_hr": ("Well discharge (Q)", "m³/hr"),
    "command_area_m2": ("Command area", "m²"), "t_p_hr": ("Pump time (t_p)", "hr"),
    "n_furrows_at_a_time": ("Furrows irrigated at a time (N)", "furrows"),
    "furrows_in_6hr": ("Furrows irrigated in 6 hr", "furrows"),
    "area_irrigable_m2": ("Total area irrigable", "m²"), "area_max_m2": ("Maximum area (area max)", "m²"),
}
D3_BY_TYPE = {"flood": ["command_area_m2", "t_p_hr"],
              "furrow": ["n_furrows_at_a_time", "furrows_in_6hr", "area_irrigable_m2", "t_p_hr"],
              "sprinkler": ["t_p_hr", "area_max_m2"]}


# ----------------------------------------------------------------------------- #
# cached resources
# ----------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading soil, slope and geology tables (first time only) ...")
def get_ctx():
    return services.load_context()


@st.cache_resource
def get_store():
    return farm_store.FarmStore(config.DASHBOARD_DB_PATH)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_rain(lat, lon, day_iso):
    return services.rainfall_24h(lat, lon)


# ----------------------------------------------------------------------------- #
# helpers (formatting only)
# ----------------------------------------------------------------------------- #
def fmt(v, digits=4):
    """4 significant digits, never hides small values (0.000535 stays 0.000535)."""
    if v is None:
        return None
    return np.format_float_positional(float(v), precision=digits, unique=True, fractional=False, trim="-")


def show(label, v, unit=""):
    s = fmt(v)
    if s is not None:                                  # null -> show nothing (the warning explains why)
        st.metric(label, f"{s} {unit}".strip())


def init_state(crops):
    defaults = dict(farm_id="my_farm", lat=20.296, lon=85.824, crop=crops[0], itype=list(TYPES)[0],
                    zoom=11, area_ha=None, well_depth_m=None, well_radius_m=None, furrow_width_m=None,
                    furrow_length_m=None, spr_q=None, spr_sp=None, lat_sp=None)
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def current_farm():
    s = st.session_state
    itype = TYPES[s["itype"]]
    farm = {"farm_id": (s["farm_id"] or "").strip(), "irrigation_type": itype}
    for key, col in KEY_TO_COL.items():
        if key != "farm_id":
            farm[col] = s.get(key)
    if itype != "furrow":                              # fields that are not shown are not used
        farm["furrow_width_m"] = farm["furrow_length_m"] = None
    if itype != "sprinkler":
        farm["sprinkler_discharge_ls"] = farm["sprinkler_spacing_m"] = farm["lateral_spacing_m"] = None
    return farm


def load_selected():
    fid = st.session_state.get("sel_farm")
    f = get_store().get(fid) if fid else None
    if not f:
        return
    for key, col in KEY_TO_COL.items():
        st.session_state[key] = f[col]
    st.session_state["itype"] = TYPE_LABEL.get(core.normalize_irrigation_type(f["irrigation_type"]),
                                               list(TYPES)[0])


# ----------------------------------------------------------------------------- #
# result cards
# ----------------------------------------------------------------------------- #
def card_decision1(d1):
    st.subheader("1 · Pump")
    state = d1.get("pump_state")
    if state:
        colour = "#c0392b" if state == "ON" else "#1e8449"
        st.markdown(f"<div style='font-size:3.2rem;font-weight:800;color:{colour};line-height:1.1'>"
                    f"PUMP {state}</div>", unsafe_allow_html=True)
    else:
        st.info("Pump decision not available (see warnings).")
    for k in ("sm_pct", "theta_trigger_pct", "taw_mm", "raw_mm"):
        show(LABELS[k][0], d1.get(k), LABELS[k][1])


def card_decision2(d2):
    st.subheader("2 · Irrigation depth")
    for k in ("d_net_mm", "d_gross_mm"):
        show(LABELS[k][0], d2.get(k), LABELS[k][1])
    show(LABELS["ea"][0], d2.get("ea"), "")
    if all(d2.get(k) is None for k in ("d_net_mm", "d_gross_mm")):
        st.info("Irrigation depth not available (see warnings).")


def card_decision3(d3, itype):
    st.subheader("3 · Well and pump time")
    show(LABELS["q_m3_day"][0], d3.get("q_m3_day"), "m³/day")
    show("Well discharge (Q), per hour", d3.get("q_m3_hr"), "m³/hr")
    for k in D3_BY_TYPE.get(itype, []):
        show(LABELS[k][0], d3.get(k), LABELS[k][1])
    if all(d3.get(k) is None for k in ["q_m3_day"] + D3_BY_TYPE.get(itype, [])):
        st.info("Pump capacity not available (see warnings).")


def details_table(res):
    rows = []
    for block, name in (("decision1", "Decision 1"), ("decision2", "Decision 2"), ("decision3", "Decision 3")):
        for k, v in res.get(block, {}).items():
            if v is None or isinstance(v, str):
                continue
            label, unit = LABELS.get(k, (k, ""))
            rows.append({"Decision": name, "Quantity": label,
                         "Value": ("yes" if v else "no") if isinstance(v, bool) else fmt(v), "Unit": unit})
    return pd.DataFrame(rows)


def show_results(res, ctx):
    itype = core.normalize_irrigation_type(res.get("irrigation_type"))
    st.divider()
    st.markdown(f"### Advice for **{res.get('farm_id')}** · {res.get('crop')} · "
                f"{TYPE_LABEL.get(itype, res.get('irrigation_type'))}")
    rain = res.get("rain") or {}
    bits = [f"data date {res.get('map_date')}"]
    if rain.get("tp_mm") is not None:
        bits.append(f"rain last 24 h {fmt(rain['tp_mm'])} mm ({rain.get('source')})")
    if res.get("snap_distance_m") is not None:
        bits.append(f"nearest database point {fmt(res['snap_distance_m'])} m away")
    st.caption(" · ".join(bits))

    st.markdown("##### Warnings")
    if res.get("status") not in ("OK",):
        st.error(f"Status: {res.get('status')}")
    for w in res.get("warnings", []):
        st.warning(w)
    if not res.get("warnings"):
        st.success("No warnings.")

    c1, c2, c3 = st.columns(3)
    with c1:
        card_decision1(res.get("decision1", {}))
    with c2:
        card_decision2(res.get("decision2", {}))
    with c3:
        card_decision3(res.get("decision3", {}), itype)

    with st.expander("All calculated values (with units)"):
        st.dataframe(details_table(res), hide_index=True, width="stretch")

    d1, d2 = st.columns(2)
    d1.download_button("Download result (JSON)", json.dumps(res, indent=2), f"advice_{res.get('farm_id')}.json",
                       "application/json")
    d2.download_button("Download result (CSV)", pd.DataFrame([core.flatten_result(res)]).to_csv(index=False),
                       f"advice_{res.get('farm_id')}.csv", "text/csv")

    # 70 m soil-moisture map around the farm
    try:
        lat, lon = float(st.session_state["lat"]), float(st.session_state["lon"])
        tif = data.find_sm_file(services.map_date())
        m, rng = mapview.sm_overlay_map(tif, lat, lon)
        with st.expander("70 m soil moisture around the farm (blue = wetter, red = drier)", expanded=True):
            from streamlit_folium import st_folium
            st_folium(m, height=420, use_container_width=True, returned_objects=[], key="sm_map")
            if rng:
                st.caption(f"Soil moisture in this view: {fmt(rng[0] * 100)} to {fmt(rng[1] * 100)} % vol")
    except Exception as e:
        st.caption(f"(soil-moisture map not shown: {e})")


# ----------------------------------------------------------------------------- #
# page
# ----------------------------------------------------------------------------- #
st.title("💧 Irrigation advice")

try:
    ctx = get_ctx()
except Exception as e:
    st.error(f"The pipeline files are not ready: {e}")
    st.info("Run  python run_pipeline.py  (and  python scripts\\06_build_point_lookup.py  once) first.")
    st.stop()
store = get_store()
crops = sorted(ctx.crops["crop_name"].tolist())
init_state(crops)

day = services.map_date()
if day is None:
    st.error("No 70 m soil-moisture map found in 03_downscaled_sm. Run run_pipeline.py first.")
else:
    age = (services.today_ist() - day).days
    st.caption(f"Soil-moisture data date: **{day:%d %b %Y}** (70 m map) · today (IST): "
               f"{services.today_ist():%d %b %Y}")
    if age > config.STALE_MAP_DAYS:
        st.warning(f"The newest 70 m map is {age} days old (limit {config.STALE_MAP_DAYS} days). "
                   f"Run run_pipeline.py to refresh it.")

# ---- sidebar: saved farms ----
with st.sidebar:
    st.header("Saved farms")
    ids = store.list_ids()
    st.selectbox("Farm", ids, key="sel_farm", index=None, placeholder="choose a saved farm" if ids else
                 "none saved yet")
    b1, b2 = st.columns(2)
    b1.button("Load", on_click=load_selected, disabled=not ids)
    if b2.button("Delete", disabled=not ids):
        if st.session_state.get("sel_farm"):
            store.delete(st.session_state["sel_farm"])
            st.rerun()
    if st.button("Save current farm", type="primary"):
        try:
            row = store.upsert(current_farm())
            st.success(f"Saved {row['farm_id']}")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    st.divider()
    st.caption("Share with the daily pipeline (farm_inputs/farms.csv)")
    if st.button("Import farms from farms.csv"):
        try:
            st.success(f"{store.import_csv(config.FARMS_CSV_PATH)} farm(s) imported")
            st.rerun()
        except OSError as e:
            st.error(f"Could not read farms.csv: {e}")
    ok = st.checkbox("Replace farms.csv (a backup copy is made first)")
    if st.button("Export saved farms to farms.csv", disabled=not (ok and ids)):
        try:
            bk = store.export_csv(config.FARMS_CSV_PATH)
            st.success("farms.csv written." + (f" Old file saved as {Path(bk).name}" if bk else ""))
        except PermissionError:
            st.error("farms.csv is open in another program (Excel?). Close it and try again.")

# ---- main: location map (code first so a click can set lat/lon) + inputs ----
left, right = st.columns([5, 6])
with right:
    st.subheader("Farm location")
    from streamlit_folium import st_folium
    out = st_folium(mapview.pick_map(float(st.session_state["lat"]), float(st.session_state["lon"]),
                                     int(st.session_state["zoom"])),
                    height=380, use_container_width=True, returned_objects=["last_clicked"], key="pick_map")
    st.slider("Map zoom", 8, 17, key="zoom")
    clk = (out or {}).get("last_clicked")
    if clk and clk != st.session_state.get("_last_click"):
        st.session_state["_last_click"] = clk
        st.session_state["lat"], st.session_state["lon"] = round(clk["lat"], 6), round(clk["lng"], 6)
        st.rerun()
    st.caption("Click the map to set the farm point (free OpenStreetMap / Esri satellite maps).")

with left:
    st.subheader("Farm inputs")
    st.text_input("Farm ID (name)", key="farm_id")
    c1, c2 = st.columns(2)
    c1.number_input("Latitude (°N)", min_value=-90.0, max_value=90.0, format="%.6f", step=0.0005, key="lat")
    c2.number_input("Longitude (°E)", min_value=-180.0, max_value=180.0, format="%.6f", step=0.0005, key="lon")
    st.selectbox("Crop", crops, key="crop")
    st.selectbox("Irrigation type", list(TYPES), key="itype")
    ph = "enter value"
    st.number_input("Farm area (ha)", min_value=0.0, format="%.4f", step=0.05, value=None, key="area_ha",
                    placeholder=ph)
    w1, w2 = st.columns(2)
    w1.number_input("Well depth (m)", min_value=0.0, format="%.2f", step=1.0, value=None,
                    key="well_depth_m", placeholder=ph)
    w2.number_input("Well radius (m)", min_value=0.0, format="%.4f", step=0.005, value=None,
                    key="well_radius_m", placeholder=ph)
    itype_key = TYPES[st.session_state["itype"]]
    if itype_key == "furrow":
        f1, f2 = st.columns(2)
        f1.number_input("Furrow width (m)", min_value=0.0, format="%.3f", step=0.1, value=None,
                        key="furrow_width_m", placeholder=ph)
        f2.number_input("Furrow length (m)", min_value=0.0, format="%.2f", step=1.0, value=None,
                        key="furrow_length_m", placeholder=ph)
    if itype_key == "sprinkler":
        s1, s2, s3 = st.columns(3)
        s1.number_input("Sprinkler discharge (L/s)", min_value=0.0, format="%.4f", step=0.05, value=None,
                        key="spr_q", placeholder=ph)
        s2.number_input("Sprinkler spacing (m)", min_value=0.0, format="%.2f", step=0.5, value=None,
                        key="spr_sp", placeholder=ph)
        s3.number_input("Lateral spacing (m)", min_value=0.0, format="%.2f", step=0.5, value=None,
                        key="lat_sp", placeholder=ph)
    go = st.button("Get irrigation advice", type="primary", disabled=day is None, width="stretch")

if go:
    farm = current_farm()
    if not farm["farm_id"]:
        farm["farm_id"] = "my_farm"
    with st.spinner("Reading the soil-moisture map, rainfall and groundwater ..."):
        try:
            st.session_state["result"] = services.run_advice(
                farm, ctx, rain_fn=lambda la, lo: cached_rain(round(la, 3), round(lo, 3),
                                                              services.today_ist().isoformat()))
        except (KeyError, ValueError, FileNotFoundError) as e:
            st.session_state.pop("result", None)
            st.error(str(e.args[0]) if e.args else str(e))

if st.session_state.get("result"):
    show_results(st.session_state["result"], ctx)
