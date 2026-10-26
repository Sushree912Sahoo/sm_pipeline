@echo off
REM Start the irrigation dashboard. Run this from the ANACONDA PROMPT (environment sm_pipeline active):
REM     run_dashboard.bat
REM It opens at http://localhost:8501   (press Ctrl+C in this window to stop it)
cd /d "%~dp0"
streamlit run dashboard\app.py --server.address 127.0.0.1 --server.port 8501
