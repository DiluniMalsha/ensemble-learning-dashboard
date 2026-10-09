@echo off
REM One-click launcher for Windows: creates a virtual environment (first time only), installs packages, starts the dashboard.
cd /d "%~dp0"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)
streamlit run app.py
pause
