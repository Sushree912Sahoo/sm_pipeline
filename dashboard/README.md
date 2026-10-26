# Irrigation dashboard (Phase 2, Windows first)

Thin layer over the pipeline: farmer inputs -> advice_for_farm(...) -> three decision cards.
No formulas in the dashboard; every number comes from scripts/irrigation_core.py.

Files
  dashboard/app.py          the page (Streamlit)
  dashboard/services.py     rainfall (same code as the pipeline), groundwater, advice call
  dashboard/farm_store.py   saved farms (SQLite: dashboard/data/farms.db)
  dashboard/mapview.py      free maps (OpenStreetMap + Esri satellite). Switch to Google Maps here later.

Start (Anaconda Prompt, project folder, environment sm_pipeline):
    pip install -r requirements-dashboard.txt        (once)
    streamlit run dashboard\app.py                    (or run_dashboard.bat)
  -> opens http://localhost:8501

Needs the daily pipeline outputs: a 70 m map for today in 03_downscaled_sm, static\point_lookup.parquet
(step 06), static\gw_point_lookup.parquet + models\groundwater (step 07).

Saved farms: "Save current farm" in the sidebar. "Import" reads farm_inputs\farms.csv; "Export" writes it
(the old file is copied to farm_inputs\_backup first), so the daily pipeline can use the saved farms.

Other PCs on your network: start with --server.address 0.0.0.0 and allow port 8501 in Windows Firewall.
There is no login, so anyone who can reach the PC can use it.

Test:  python tests\test_dashboard.py     (rainfall and groundwater are stubbed; uses a temporary farm database)
