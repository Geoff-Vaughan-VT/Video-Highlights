@echo off
REM Video Highlights Generator - GUI Launcher for Windows
REM This script launches the Tkinter GUI on Windows with dependency checking

echo ============================================
echo Video Highlights Generator - GUI Launcher
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if numpy is installed (quick dependency check)
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Required dependencies not found!
    echo.
    echo The Video Highlights Generator requires several Python packages.
    echo Would you like to install them now? This may take a few minutes.
    echo.
    choice /C YN /M "Install dependencies"
    if errorlevel 2 goto :skip_install
    if errorlevel 1 goto :install_deps
)

:check_torch
REM Check if PyTorch with CUDA is installed for GPU acceleration
python -c "import torch; print('GPU Available' if torch.cuda.is_available() else 'CPU Only')" 2>nul
if errorlevel 1 (
    echo.
    echo NOTE: PyTorch not installed - GPU acceleration unavailable
    echo For better performance, install PyTorch with CUDA:
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    echo.
    echo Press any key to continue with CPU mode...
    pause >nul
)

goto :launch_gui

:install_deps
echo.
echo Installing dependencies from requirements.txt...
echo This may take several minutes. Please wait...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please try manually: pip install -r requirements.txt
    pause
    exit /b 1
)
echo.
echo Dependencies installed successfully!
echo.

REM Offer to install PyTorch with CUDA for GPU acceleration
echo Would you like to install GPU acceleration support (CUDA/PyTorch)?
echo This is optional but provides 2-3x faster processing.
echo.
choice /C YN /M "Install GPU support"
if errorlevel 2 goto :launch_gui
if errorlevel 1 goto :install_gpu

:install_gpu
echo.
echo Installing PyTorch with CUDA support...
echo This download is large (~2GB). Please wait...
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo.
    echo WARNING: Failed to install GPU support
    echo You can still use CPU mode, or install manually later:
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    echo.
)
goto :launch_gui

:skip_install
echo.
echo Skipping dependency installation.
echo If the GUI fails to start, run: pip install -r requirements.txt
echo.

:launch_gui
echo Starting GUI...
echo.

REM Launch the GUI
python VideoHighlightsGUI.py

if errorlevel 1 (
    echo.
    echo ============================================
    echo ERROR: Failed to start GUI
    echo ============================================
    echo.
    echo If you see "ModuleNotFoundError", install dependencies:
    echo   pip install -r requirements.txt
    echo.
    echo For GPU support (2-3x faster):
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    echo.
    pause
)

REM Keep window open if launched by double-clicking
echo.
echo GUI closed.
pause
