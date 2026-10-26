@echo off
rem ------------------------------------------------------------------
rem run_daily.bat  -  started every day by Windows Task Scheduler.
rem Runs the whole pipeline with no one watching:
rem   SMAP download, rainfall/temperature, lag model, 70 m downscaling,
rem   groundwater prediction, irrigation advice.
rem
rem STEP 1 (the only required edit): put the full path of the python.exe
rem         that belongs to your conda environment "sm_pipeline".
rem         In Anaconda Prompt run:
rem             conda activate sm_pipeline
rem             where python
rem         and copy the FIRST line (it must contain \envs\sm_pipeline\).
rem ------------------------------------------------------------------
set "PYTHON=C:\Users\RUTUDHWAJA\.conda\envs\sm_pipeline\python.exe"

rem hidden browser for the rainfall scraper
set SM_HEADLESS=1
rem avoid text-encoding crashes when output goes to a log file
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem always work from the folder this .bat sits in (the sm_pipeline folder)
cd /d "%~dp0"
if not exist logs mkdir logs

rem stop early with a clear message if the python path is wrong
if not exist "%PYTHON%" (
    echo %date% %time%  ERROR: python not found at %PYTHON% >> logs\scheduler.log
    exit /b 1
)

echo ================================================ >> logs\scheduler.log
echo Started  %date% %time% >> logs\scheduler.log

"%PYTHON%" run_pipeline.py >> logs\scheduler_output.log 2>&1
set RC=%errorlevel%

echo Finished %date% %time%  exit code %RC% >> logs\scheduler.log
exit /b %RC%
