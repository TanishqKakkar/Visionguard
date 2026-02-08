@echo off
echo ========================================
echo Starting CCTV Detection API Server
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
)

REM Set model path (change if needed)
if "%MODEL_PATH%"=="" set MODEL_PATH=models\vit_convlstm_best.pt

REM Check if model exists
if not exist "%MODEL_PATH%" (
    echo WARNING: Model file not found at %MODEL_PATH%
    echo Please ensure the model file exists or set MODEL_PATH environment variable
    echo.
)

echo Starting API server on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python api.py

pause

