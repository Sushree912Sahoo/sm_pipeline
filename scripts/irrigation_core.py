"""
================================================================================
irrigation_core.py  -  the three irrigation decisions as PLAIN FUNCTIONS

No file reading, no pandas, no rasters in this module. Every function takes plain numbers and
text and returns a dictionary. The batch script (08_irrigation_advice.py) and the dashboard both
call these functions, so the numbers are identical everywhere. All lookups (soil, geology, SM,
groundwater) live in irrigation_data.py.

    decision1(...)  pump ON/OFF                                 (formulas: Irrigation_Terms docx, eq. 2-4)
    decision2(...)  irrigation depth (net and gross)            (docx eq. 8, 9)
    decision3(...)  well discharge Q, pump time, area           (Sichardt Ro; Q = pi*K*(H^2-hw^2)/ln(Ro/rw))

RULES USED EVERYWHERE
  * Missing input  ->  the output is None (never 0, never a default) and a plain-words warning is added.
  * Units: theta_FC / theta_WP / SM are shown in % by volume, but every formula uses m3/m3 (divide by 100).
           TAW, RAW, d_net, d_gross in mm. H, hw, Ro, rw, GSE, GWE in m. Q in m3/day, Q_hr in m3/hr.
           K in m/day (Ro only: K / 86400 in m/s).
  * Values are returned UNROUNDED. Rounding happens only when results are written (round_sig).
  * Decisions 2 and 3 are always calculated, whether Decision 1 says ON or OFF.

SELF-TEST (the Word reference example + hand-checked Decision 3 numbers):
    python scripts/irrigation_core.py --selftest
================================================================================
"""
import math

T_P_MAX_HR_DEFAULT = 6.0

# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
_TYPE_ALIASES = {
    "flood": "flood", "surface": "flood", "surface irrigation": "flood",
    "surface irrigation (flood)": "flood", "flood irrigation": "flood",
    "furrow": "furrow", "furrow irrigation": "furrow",
    "sprinkler": "sprinkler", "sprinkler irrigation": "sprinkler",
}
SUPPORTED_TYPES = ("flood", "furrow", "sprinkler")


def num(x):
    """float(x), or None when x is missing / not a number / not finite."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def normalize_irrigation_type(x):
    """'Surface irrigation (flood)' / 'Furrow irrigation' / 'Sprinkler irrigation' (or the short
    names) -> 'flood' | 'furrow' | 'sprinkler'. Anything else -> None."""
    if x is None:
        return None
    return _TYPE_ALIASES.get(str(x).strip().lower())


def round_sig(x, digits=6):
    """Round to significant digits (keeps tiny values such as K = 0.000535 intact)."""
    v = num(x)
    if v is None:
        return None
    if v == 0:
        return 0.0
    return round(v, digits - 1 - int(math.floor(math.log10(abs(v)))))


def theta_fc_wp_pct(sand, silt, clay):
    """Field capacity and wilting point in % by volume (BD removed, as decided):
        theta_FC = 56.37 - 0.51*sand - 0.27*silt        theta_WP = 0.71 + 0.44*clay
    sand, silt, clay in %."""
    return 56.37 - 0.51 * sand - 0.27 * silt, 0.71 + 0.44 * clay


# ----------------------------------------------------------------------------- #
# DECISION 1  -  pump ON / OFF
# ----------------------------------------------------------------------------- #
def decision1(sm_m3m3, zr_m, mad, sand=None, silt=None, clay=None,
              theta_fc_pct=None, theta_wp_pct=None):
    """Pump ON/OFF.

    sm_m3m3   70 m soil moisture at the farm point, m3/m3 (the raster value, a fraction)
    zr_m      root depth, m (zr_max_m of the crop)
    mad       depletion factor p of the crop (fraction)
    sand, silt, clay   % at the farm point; used unless theta_fc_pct and theta_wp_pct are given
    theta_fc_pct, theta_wp_pct   optional: give them directly (for tests)

    Returns: theta_fc_pct, theta_wp_pct, theta_trigger_pct, sm_pct, zr_m, mad, awc_mm, taw_mm,
             raw_mm, pump_on (True/False/None), pump_state ('ON'/'OFF'/None), warnings
    """
    warns = []
    out = dict(theta_fc_pct=None, theta_wp_pct=None, theta_trigger_pct=None, sm_pct=None,
               zr_m=num(zr_m), mad=num(mad), awc_mm=None, taw_mm=None, raw_mm=None,
               pump_on=None, pump_state=None)

    fc, wp = num(theta_fc_pct), num(theta_wp_pct)
    if fc is None or wp is None:
        s, si, c = num(sand), num(silt), num(clay)
        if None in (s, si, c):
            warns.append("no sand/silt/clay at this point, so field capacity and wilting point "
                         "cannot be calculated")
        else:
            fc, wp = theta_fc_wp_pct(s, si, c)
    out["theta_fc_pct"], out["theta_wp_pct"] = fc, wp

    sm = num(sm_m3m3)
    if sm is not None:
        out["sm_pct"] = sm * 100.0
    if out["zr_m"] is None:
        warns.append("no root depth (Zr) for this crop")
    if out["mad"] is None:
        warns.append("no depletion factor (MAD) for this crop")

    if fc is not None and wp is not None:
        taw_frac = (fc - wp) / 100.0                                  # m3/m3
        if out["zr_m"] is not None:
            out["awc_mm"] = 1000.0 * taw_frac * out["zr_m"]           # no BD, Zr kept
            out["taw_mm"] = 1000.0 * taw_frac * out["zr_m"]           # docx eq. 2
            if out["mad"] is not None:
                out["raw_mm"] = out["mad"] * out["taw_mm"]            # docx eq. 3
        if out["mad"] is not None:
            out["theta_trigger_pct"] = fc - out["mad"] * (fc - wp)    # docx eq. 4
    if out["theta_trigger_pct"] is not None and out["sm_pct"] is not None:
        out["pump_on"] = out["sm_pct"] <= out["theta_trigger_pct"]
        out["pump_state"] = "ON" if out["pump_on"] else "OFF"
    elif sm is None:
        warns.append("no soil moisture value, so pump ON/OFF cannot be decided")
    out["warnings"] = warns
    return out


# ----------------------------------------------------------------------------- #
# DECISION 2  -  irrigation depth
# ----------------------------------------------------------------------------- #
def decision2(sm_m3m3, theta_fc_pct, zr_m, irrigation_type, ea_table):
    """Net and gross irrigation depth. Always calculated (pump ON or OFF).

    ea_table   dict like {'flood': 0.50, 'furrow': 0.60, 'sprinkler': 0.75} (application efficiency)

    d_net   = 1000 * (theta_FC - theta) * Zr        (mm; theta = SM, both in m3/m3)   docx eq. 8
    d_gross = d_net / Ea                            (mm)                              docx eq. 9
    If SM is at or above field capacity nothing can be added to the root zone: d_net = 0.
    Returns: d_net_mm, ea, d_gross_mm, warnings
    """
    warns = []
    out = dict(d_net_mm=None, ea=None, d_gross_mm=None)
    key = normalize_irrigation_type(irrigation_type)
    if key is None:
        warns.append(f"irrigation type '{irrigation_type}' is not supported "
                     f"(only flood, furrow and sprinkler), so no irrigation depth is given")
    else:
        ea = num((ea_table or {}).get(key))
        if ea is None or ea <= 0:
            warns.append(f"no application efficiency (Ea) for '{key}' in the efficiency table")
        else:
            out["ea"] = ea
    sm, fc, zr = num(sm_m3m3), num(theta_fc_pct), num(zr_m)
    if sm is None or fc is None or zr is None:
        warns.append("net irrigation depth needs soil moisture, field capacity and root depth; "
                     "one of them is missing")
    else:
        d_net = 1000.0 * (fc / 100.0 - sm) * zr
        if d_net < 0:
            d_net = 0.0
            warns.append("soil moisture is above field capacity: no net irrigation depth needed")
        out["d_net_mm"] = d_net
        if out["ea"] is not None:
            out["d_gross_mm"] = d_net / out["ea"]
    out["warnings"] = warns
    return out


# ----------------------------------------------------------------------------- #
# DECISION 3  -  pump capacity and running time
# ----------------------------------------------------------------------------- #
def well_discharge(gwe_m, gse_m, k_m_day, well_depth_m, well_radius_m):
    """Section 7.1. Returns a dict with well_bottom_m_amsl, h_m, hw_m, ro_m, q_m3_day, q_m3_hr, warnings.
        well bottom = GSE - well_depth          H  = GWE - well bottom
        hw = H - H*2/3                          Ro = 3000 * (H*2/3) * sqrt(K/86400)     (Sichardt)
        Q  = pi * K * (H^2 - hw^2) / ln(Ro / rw)    [m3/day, K in m/day]          Q_hr = Q / 24
    """
    warns = []
    out = dict(well_bottom_m_amsl=None, h_m=None, hw_m=None, ro_m=None, q_m3_day=None, q_m3_hr=None)
    gwe, gse, k = num(gwe_m), num(gse_m), num(k_m_day)
    wd, rw = num(well_depth_m), num(well_radius_m)
    missing = []
    if gwe is None:
        missing.append("groundwater level (GWE)")
    if gse is None:
        missing.append("ground surface elevation (GSE)")
    if wd is None:
        missing.append("well depth")
    if rw is None:
        missing.append("well radius")
    if missing:
        warns.append("well discharge not calculated: missing " + ", ".join(missing))
    if k is None:
        warns.append("well discharge not calculated: no K (hydraulic conductivity) for this geology")
    if missing or k is None:
        out["warnings"] = warns
        return out

    bottom = gse - wd
    h = gwe - bottom
    out["well_bottom_m_amsl"], out["h_m"] = bottom, h
    if h <= 0:
        warns.append(f"groundwater level is at or below the well bottom (H = {h:.2f} m), "
                     f"so well discharge is not calculated")
        out["warnings"] = warns
        return out
    hw = h - h * 2.0 / 3.0
    ro = 3000.0 * (h * 2.0 / 3.0) * math.sqrt(k / 86400.0)
    out["hw_m"], out["ro_m"] = hw, ro
    if rw <= 0:
        warns.append("well radius must be greater than 0, so well discharge is not calculated")
    elif ro <= rw:
        warns.append(f"radius of influence ({ro:.3g} m) is not larger than the well radius "
                     f"({rw:g} m), so well discharge is not calculated")
    else:
        q = math.pi * k * (h ** 2 - hw ** 2) / math.log(ro / rw)
        out["q_m3_day"], out["q_m3_hr"] = q, q / 24.0
    out["warnings"] = warns
    return out


def decision3(gwe_m, gse_m, k_m_day, well_depth_m, well_radius_m, irrigation_type, d_gross_mm,
              area_ha=None, furrow_width_m=None, furrow_length_m=None, slope_pct=None,
              sprinkler_discharge_ls=None, sprinkler_spacing_m=None, lateral_spacing_m=None,
              t_p_max_hr=T_P_MAX_HR_DEFAULT):
    """Pump capacity and running time. Always calculated (pump ON or OFF).

    d_gross_mm comes from decision2. area_ha is the farm area in hectares (x 10000 -> m2).
    sprinkler_discharge_ls is litres per second (x 3.6 -> m3/hr).

    Returned columns: gwe_m_amsl, gse_m_amsl, well_bottom_m_amsl, h_m, hw_m, ro_m, k_m_day,
    slope_pct, q_m3_day, q_m3_hr, plus the outputs of the chosen type:
        flood      command_area_m2, t_p_hr
        furrow     n_furrows_at_a_time, furrows_in_6hr, area_irrigable_m2, t_p_hr
        sprinkler  t_p_hr, area_max_m2
    """
    gse, gwe = num(gse_m), num(gwe_m)
    out = dict(gwe_m_amsl=gwe, gse_m_amsl=gse, well_bottom_m_amsl=None, h_m=None, hw_m=None,
               ro_m=None, k_m_day=num(k_m_day), slope_pct=num(slope_pct), q_m3_day=None, q_m3_hr=None)
    wd = well_discharge(gwe, gse, k_m_day, well_depth_m, well_radius_m)
    warns = wd.pop("warnings")
    out.update(wd)
    q_hr = out["q_m3_hr"]
    tmax = num(t_p_max_hr) or T_P_MAX_HR_DEFAULT
    v_total = q_hr * tmax if q_hr is not None else None            # m3 pumped in t_p_max hours

    key = normalize_irrigation_type(irrigation_type)
    dg_mm = num(d_gross_mm)
    dg = dg_mm / 1000.0 if dg_mm is not None else None              # m
    if key is None:
        warns.append(f"irrigation type '{irrigation_type}' is not supported "
                     f"(only flood, furrow and sprinkler), so no pump time or area is given")
        out["warnings"] = warns
        return out
    if dg is None:
        warns.append("pump time and area need the gross irrigation depth (Decision 2), which is missing")

    if key == "flood":
        out.update(command_area_m2=None, t_p_hr=None)
        area_ha_v = num(area_ha)
        if area_ha_v is None or area_ha_v <= 0:
            warns.append("flood: farm area (area_ha) is missing")
        elif dg is not None and q_hr is not None:
            v_gross = dg * (area_ha_v * 10000.0)                    # m3
            out["t_p_hr"] = v_gross / q_hr if v_gross <= v_total else tmax
        if dg is not None and q_hr is not None:
            if dg > 0:
                out["command_area_m2"] = v_total / dg
            else:
                warns.append("no irrigation needed (gross depth is 0), so command area is not calculated")

    elif key == "furrow":
        out.update(n_furrows_at_a_time=None, furrows_in_6hr=None, area_irrigable_m2=None, t_p_hr=None)
        w, l = num(furrow_width_m), num(furrow_length_m)
        s = num(slope_pct)
        if w is None or l is None or w <= 0 or l <= 0:
            warns.append("furrow: furrow width or furrow length is missing")
        else:
            req = dg * w * l if dg is not None else None             # m3 needed per furrow
            # these two do NOT depend on the slope
            if req is not None and q_hr is not None:
                if req > 0:
                    n = v_total / req
                    out["furrows_in_6hr"] = n
                    out["area_irrigable_m2"] = n * (w * l)
                else:
                    warns.append("no irrigation needed (gross depth is 0), so furrows and area "
                                 "irrigable are not calculated")
            # these two need q, and q needs the slope
            if s is None or s <= 0:
                warns.append("furrow: slope is missing or 0, so the water released in one furrow (q), "
                             "furrows irrigated at a time (N) and furrow pump time are not calculated")
            else:
                q_ls = (0.6 / s) * 0.80                              # L/s  (slope in %)
                q_m3hr = q_ls * 3.6                                  # m3/hr
                if q_hr is not None:
                    out["n_furrows_at_a_time"] = q_hr / q_m3hr
                if req is not None:
                    out["t_p_hr"] = req / q_m3hr

    else:  # sprinkler
        out.update(t_p_hr=None, area_max_m2=None)
        qs, sl, sm_ = num(sprinkler_discharge_ls), num(sprinkler_spacing_m), num(lateral_spacing_m)
        if None in (qs, sl, sm_) or qs <= 0 or sl <= 0 or sm_ <= 0:
            warns.append("sprinkler: sprinkler discharge, sprinkler spacing or lateral spacing is missing")
        elif dg is not None:
            i_rate = (qs * 3.6) / (sl * sm_)                          # m/hr
            t_p = dg / i_rate
            out["t_p_hr"] = t_p
            if dg > 0:
                if q_hr is not None:
                    out["area_max_m2"] = (q_hr * t_p) / dg            # V_gross_t_p = Q_hr * t_p
            else:
                warns.append("no irrigation needed (gross depth is 0), so area max is not calculated")
    out["warnings"] = warns
    return out


# ----------------------------------------------------------------------------- #
# RESULT ASSEMBLY (JSON block per farm, flat CSV row)
# ----------------------------------------------------------------------------- #
D1_KEYS = ["theta_fc_pct", "theta_wp_pct", "theta_trigger_pct", "sm_pct", "zr_m", "mad", "awc_mm",
           "taw_mm", "raw_mm", "pump_on", "pump_state"]
D2_KEYS = ["d_net_mm", "ea", "d_gross_mm"]
D3_KEYS = ["gwe_m_amsl", "gse_m_amsl", "well_bottom_m_amsl", "h_m", "hw_m", "ro_m", "k_m_day",
           "slope_pct", "q_m3_day", "q_m3_hr", "command_area_m2", "n_furrows_at_a_time",
           "furrows_in_6hr", "area_irrigable_m2", "area_max_m2", "t_p_hr"]
ID_KEYS = ["farm_id", "date", "crop", "irrigation_type", "lookup_lat", "lookup_lon", "snap_distance_m",
           "sm_qa"]
FLAT_COLUMNS = (ID_KEYS + D1_KEYS[:4] + D1_KEYS[4:] + D2_KEYS + D3_KEYS + ["status", "warnings"])


def _clean(d, keys):
    out = {}
    for k in keys:
        if k in d:
            v = d[k]
            out[k] = v if isinstance(v, (bool, str)) or v is None else round_sig(v)
    return out


def assemble_result(farm_id, date, crop, irrigation_type, d1=None, d2=None, d3=None,
                    status="OK", extra_warnings=(), meta=None):
    """One farm -> one nested dict (the JSON object). Unavailable values are None (JSON null)."""
    d1, d2, d3 = d1 or {}, d2 or {}, d3 or {}
    warns = list(extra_warnings)
    for d in (d1, d2, d3):
        for w in d.get("warnings", []):
            if w not in warns:
                warns.append(w)
    res = dict(farm_id=farm_id, date=date, crop=crop, irrigation_type=irrigation_type, status=status)
    if meta:
        res.update(meta)
    res["decision1"] = _clean(d1, D1_KEYS)
    res["decision2"] = _clean(d2, D2_KEYS)
    res["decision3"] = _clean(d3, D3_KEYS)
    res["warnings"] = warns
    return res


def flatten_result(res):
    """Nested result -> flat dict with the Section 9 column names (for the CSV)."""
    flat = {k: res.get(k) for k in ID_KEYS}
    flat["irrigation_type"] = res.get("irrigation_type")
    for block in ("decision1", "decision2", "decision3"):
        flat.update(res.get(block, {}))
    flat["status"] = res.get("status")
    flat["warnings"] = "; ".join(res.get("warnings", []))
    return {c: flat.get(c) for c in FLAT_COLUMNS}


# ----------------------------------------------------------------------------- #
# SELF-TEST
# ----------------------------------------------------------------------------- #
def selftest(verbose=True):
    """Returns the number of failed checks (0 = all passed)."""
    bad = []

    def chk(name, got, want, tol=1e-6):
        ok = (got == want) if isinstance(want, (str, bool)) or want is None else \
            (got is not None and abs(got - want) <= tol * max(1.0, abs(want)))
        if verbose:
            print(f"  {'ok  ' if ok else 'FAIL'} {name:46s} got {got!r:>24}   expected {want!r}")
        if not ok:
            bad.append(name)

    print("Section 11 (Word reference): theta_FC 27, theta_WP 12, Zr 1.0, MAD 0.50, SM 21, Ea 0.60")
    d1 = decision1(0.21, 1.0, 0.50, theta_fc_pct=27, theta_wp_pct=12)
    chk("TAW (mm)", d1["taw_mm"], 150.0)
    chk("AWC (mm)", d1["awc_mm"], 150.0)
    chk("RAW (mm)", d1["raw_mm"], 75.0)
    chk("theta_trigger (% vol)", d1["theta_trigger_pct"], 19.5)
    chk("Decision 1", d1["pump_state"], "OFF")
    d2 = decision2(0.21, 27, 1.0, "furrow", {"furrow": 0.60})
    chk("d_net (mm)", d2["d_net_mm"], 60.0)
    chk("d_gross (mm)", d2["d_gross_mm"], 100.0)
    chk("pump ON when SM = 19.5 (<= trigger)", decision1(0.195, 1.0, 0.5, theta_fc_pct=27,
                                                          theta_wp_pct=12)["pump_state"], "ON")

    print("Decision 3: K 0.0847584 m/day, H 10 m, rw 0.5 m  (GSE 100, well depth 30 -> bottom 70, GWE 80 -> H = 10)")
    w = well_discharge(80, 100, 0.0847584, 30, 0.5)
    chk("well bottom (m amsl)", w["well_bottom_m_amsl"], 70.0)
    chk("H (m)", w["h_m"], 10.0)
    chk("hw (m)", w["hw_m"], 10.0 / 3.0)
    chk("Ro (m) ~ 19.8", w["ro_m"], 19.8, tol=0.01)
    chk("Q (m3/day) ~ 6.4", w["q_m3_day"], 6.4, tol=0.01)
    chk("Q_hr (m3/hr) ~ 0.27", w["q_m3_hr"], 0.27, tol=0.02)
    q_hr = w["q_m3_hr"]

    print("Flood (d_gross 100 mm): area 0.002 ha = 20 m2 -> V_gross = 2.0 m3 > V_total = Q_hr * 6 = 1.61 m3")
    f = decision3(80, 100, 0.0847584, 30, 0.5, "flood", 100.0, area_ha=0.002)
    chk("V_gross > V_total -> t_p = 6 hr", f["t_p_hr"], 6.0)
    chk("command area (m2) = V_total / d_gross", f["command_area_m2"], q_hr * 6 / 0.1)
    f2 = decision3(80, 100, 0.0847584, 30, 0.5, "flood", 100.0, area_ha=0.0001)
    chk("V_gross <= V_total -> t_p = V_gross / Q_hr", f2["t_p_hr"], (0.1 * 1.0) / q_hr)

    print("Furrow (d_gross 100 mm, w 1.0 m, l 50 m, slope 0.5 %): q = (0.6/0.5)*0.8 = 0.96 L/s = 3.456 m3/hr")
    fu = decision3(80, 100, 0.0847584, 30, 0.5, "furrow", 100.0, furrow_width_m=1.0,
                   furrow_length_m=50.0, slope_pct=0.5)
    req = 0.1 * 1.0 * 50.0
    chk("N = Q_hr / q", fu["n_furrows_at_a_time"], q_hr / 3.456)
    chk("t_p one furrow = req / q (hr)", fu["t_p_hr"], req / 3.456)
    chk("n furrows in 6 hr", fu["furrows_in_6hr"], q_hr * 6 / req)
    chk("area irrigable (m2)", fu["area_irrigable_m2"], q_hr * 6 / req * 50.0)
    fn = decision3(80, 100, 0.0847584, 30, 0.5, "furrow", 100.0, furrow_width_m=1.0, furrow_length_m=50.0)
    chk("no slope: area still calculated", fn["area_irrigable_m2"], q_hr * 6 / req * 50.0)
    chk("no slope: N left empty", fn["n_furrows_at_a_time"], None)

    print("Sprinkler (d_gross 100 mm, q_s 0.5 L/s = 1.8 m3/hr, S_L 12 m, S_m 12 m): I = 1.8/144 = 0.0125 m/hr")
    sp = decision3(80, 100, 0.0847584, 30, 0.5, "sprinkler", 100.0, sprinkler_discharge_ls=0.5,
                   sprinkler_spacing_m=12, lateral_spacing_m=12)
    chk("t_p = d_gross / I (hr)", sp["t_p_hr"], 0.1 / 0.0125)
    chk("area max = Q_hr * t_p / d_gross (m2)", sp["area_max_m2"], q_hr * 8.0 / 0.1)

    print("Missing / unsupported inputs give None + warning, never 0")
    nk = decision3(80, 100, None, 30, 0.5, "flood", 100.0, area_ha=0.5)
    chk("no K -> Q empty", nk["q_m3_day"], None)
    chk("no K -> warning given", any("K" in x for x in nk["warnings"]), True)
    dr = decision2(0.21, 27, 1.0, "drip", {"furrow": 0.6})
    chk("drip not supported -> d_gross empty", dr["d_gross_mm"], None)
    chk("drip -> d_net still calculated", dr["d_net_mm"], 60.0)
    wet = decision2(0.30, 27, 1.0, "furrow", {"furrow": 0.6})
    chk("SM above FC -> d_net 0", wet["d_net_mm"], 0.0)
    dry = decision3(65, 100, 0.0847584, 30, 0.5, "flood", 100.0, area_ha=0.5)
    chk("GWE below well bottom (H<=0) -> Q empty", dry["q_m3_day"], None)

    print(f"\n{'ALL CHECKS PASSED' if not bad else 'FAILED: ' + ', '.join(bad)}")
    return len(bad)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
    print(__doc__)
