@echo off
REM Swimming Pool Detection System - Windows Setup Script
REM Author: Swimming Pool Detection Team
REM Date: 2026-01-02

echo ============================================================
echo   Swimming Pool Detection System - Setup
echo ============================================================
echo.

REM Step 1: Check Python installation
echo [Step 1/8] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and add it to PATH.
    pause
    exit /b 1
)
python --version
echo.

REM Step 2: Create virtual environment
echo [Step 2/8] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)
echo.

REM Step 3: Activate virtual environment
echo [Step 3/8] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Step 4: Upgrade pip
echo [Step 4/8] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip. Continuing anyway...
)
echo.

REM Step 5: Install dependencies
echo [Step 5/8] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

REM Step 6: Create directory structure
echo [Step 6/8] Creating directory structure...
python scripts\create_directories.py
if errorlevel 1 (
    echo [ERROR] Failed to create directory structure.
    pause
    exit /b 1
)
echo.

REM Step 7: Download datasets (optional)
echo [Step 7/8] Dataset download...
echo To download datasets, you need to configure API credentials:
echo   - Kaggle: Place kaggle.json in %%USERPROFILE%%\.kaggle\
echo   - Roboflow: Set ROBOFLOW_API_KEY environment variable
echo.
echo Skipping automatic download. Run manually:
echo   python datasets\download_dataset.py
echo.

REM Step 8: Verify installation
echo [Step 8/8] Verifying installation...
python scripts\verify_setup.py
if errorlevel 1 (
    echo [WARNING] Some components may not be properly configured.
    echo Please review the verification output above.
)
echo.

echo ============================================================
echo   Setup Complete
echo ============================================================
echo.
echo To activate the environment in future sessions:
echo   call venv\Scripts\activate.bat
echo.
echo To start training:
echo   python training\train.py --config config\training_config.yaml
echo.
echo To run inference:
echo   python inference\detect_pools.py --input path\to\image.jpg --model weights\best.pt
echo.
pause
