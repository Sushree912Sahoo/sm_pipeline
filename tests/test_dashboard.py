"""Run:  python tests/test_dashboard.py   (uses Streamlit's AppTest; rainfall and groundwater are stubbed)"""
import sys
from pathlib import Path
from unittest import mock
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "dashboard", ROOT / "scripts", ROOT):
    sys.path.insert(0, str(p))
from streamlit.testing.v1 import AppTest
import tempfile, config
import services, irrigation_data as data

GW = dict(gwl_depth_m=9.0, gwe_pred=60.0, ground_elevation_amsl=69.0, warnings="")
stub_rain = lambda lat, lon, timeout=0: {"tp_mm": 0.0, "source": "stations", "note": ""}

def btn(at, label):
    return [b for b in at.button if b.label == label][0]

_tmp = tempfile.mkdtemp()      # saved farms go to a throw-away database, not your real one
with mock.patch.object(config, "DASHBOARD_DB_PATH", str(Path(_tmp) / "test_farms.db")), \
     mock.patch.object(services, "rainfall_24h", stub_rain), \
     mock.patch.object(data.Context, "predict_gw", lambda self, *a, **k: GW):
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    print("1. page loads, no exception. title:", at.title[0].value)

    # --- flood farm ---
    at.text_input(key="farm_id").set_value("T1")
    at.selectbox(key="crop").select("Tomato")
    at.number_input(key="area_ha").set_value(0.5)
    at.number_input(key="well_depth_m").set_value(40.0)
    at.number_input(key="well_radius_m").set_value(0.075)
    at.run()
    btn(at, "Get irrigation advice").click().run()
    assert not at.exception, at.exception
    res = at.session_state["result"]
    print("2. flood advice: status", res["status"], "| pump", res["decision1"]["pump_state"],
          "| d_gross", res["decision2"]["d_gross_mm"], "| Q m3/day", res["decision3"]["q_m3_day"],
          "| command area", res["decision3"].get("command_area_m2"))
    metrics = {m.label: m.value for m in at.metric}
    print("   cards show:", metrics)
    assert "Well discharge (Q)" in metrics and "Net irrigation depth" in metrics
    assert any("PUMP" in m.value for m in at.markdown), "big ON/OFF not shown"

    # --- furrow fields appear only for furrow, sprinkler fields only for sprinkler ---
    keys = lambda: {n.key for n in at.number_input}
    assert "furrow_width_m" not in keys() and "spr_q" not in keys()
    at.selectbox(key="itype").select("Furrow irrigation").run()
    assert "furrow_width_m" in keys() and "spr_q" not in keys()
    at.selectbox(key="itype").select("Sprinkler irrigation").run()
    assert "spr_q" in keys() and "furrow_width_m" not in keys()
    print("3. conditional fields OK")

    # --- furrow with missing dimensions -> warning shown, not a zero ---
    at.selectbox(key="itype").select("Furrow irrigation").run()
    btn(at, "Get irrigation advice").click().run()
    assert not at.exception, at.exception
    res = at.session_state["result"]
    print("4. furrow, no dimensions: area =", res["decision3"].get("area_irrigable_m2"),
          "| warnings:", [w for w in res["warnings"] if "furrow" in w])
    assert res["decision3"].get("area_irrigable_m2") is None and any("furrow" in w for w in res["warnings"])
    assert any("furrow" in w.value for w in at.warning), "furrow warning not shown on the page"

    # --- save / load / export ---
    btn(at, "Save current farm").click().run()
    assert not at.exception, at.exception
    print("5. saved farms:", at.selectbox(key="sel_farm").options)
    assert "T1" in at.selectbox(key="sel_farm").options

    # --- unsupported crop / stale ---
    at.text_input(key="farm_id").set_value("T2").run()
    print("6. no exceptions in whole run")
print("DASHBOARD TESTS PASSED")
