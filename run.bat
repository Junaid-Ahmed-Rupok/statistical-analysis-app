@echo off
REM One-command launcher for Windows. Creates a local .venv on first run.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv\Scripts\python.exe (
  %PY% -c "import sys; sys.exit(sys.version_info < (3,11))" || (echo Python 3.11+ is required: https://python.org/downloads & exit /b 1)
  %PY% -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
streamlit run app.py
