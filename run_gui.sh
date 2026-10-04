#!/bin/bash
# Video Highlights Generator - GUI Launcher for Linux/Mac
# This script launches the Tkinter GUI with dependency checking

echo "============================================"
echo "Video Highlights Generator - GUI Launcher"
echo "============================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from python.org or your package manager"
    exit 1
fi

echo "Python found:"
python3 --version
echo

# Check if numpy is installed (quick dependency check)
if ! python3 -c "import numpy" &> /dev/null; then
    echo "WARNING: Required dependencies not found!"
    echo
    echo "The Video Highlights Generator requires several Python packages."
    echo "Would you like to install them now? This may take a few minutes."
    echo
    read -p "Install dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo
        echo "Installing dependencies from requirements.txt..."
        echo "This may take several minutes. Please wait..."
        echo
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo
            echo "ERROR: Failed to install dependencies"
            echo "Please try manually: pip install -r requirements.txt"
            exit 1
        fi
        echo
        echo "Dependencies installed successfully!"
        echo

        # Offer to install PyTorch with CUDA for GPU acceleration
        echo "Would you like to install GPU acceleration support (CUDA/PyTorch)?"
        echo "This is optional but provides 2-3x faster processing."
        echo
        read -p "Install GPU support? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo
            echo "Installing PyTorch with CUDA support..."
            echo "This download is large (~2GB). Please wait..."
            echo
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
            if [ $? -ne 0 ]; then
                echo
                echo "WARNING: Failed to install GPU support"
                echo "You can still use CPU mode, or install manually later:"
                echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
                echo
            fi
        fi
    else
        echo
        echo "Skipping dependency installation."
        echo "If the GUI fails to start, run: pip install -r requirements.txt"
        echo
    fi
fi

# Check if PyTorch with CUDA is installed for GPU acceleration
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_STATUS=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "GPU Available: $GPU_STATUS"
elif python3 -c "import torch" 2>/dev/null; then
    echo "NOTE: PyTorch installed but GPU not available (CPU mode)"
else
    echo "NOTE: PyTorch not installed - GPU acceleration unavailable"
    echo "For better performance, install PyTorch with CUDA:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    echo
fi

echo
echo "Starting GUI..."
echo

# Launch the GUI
python3 VideoHighlightsGUI.py

if [ $? -ne 0 ]; then
    echo
    echo "============================================"
    echo "ERROR: Failed to start GUI"
    echo "============================================"
    echo
    echo "If you see 'ModuleNotFoundError', install dependencies:"
    echo "  pip install -r requirements.txt"
    echo
    echo "For GPU support (2-3x faster):"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    echo
    exit 1
fi

echo
echo "GUI closed."
