#!/bin/bash

echo "========================================"
echo "Starting CCTV Detection API Server"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

# Check if requirements are installed
if ! python3 -c "import flask" &> /dev/null; then
    echo "Installing required packages..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install requirements"
        exit 1
    fi
fi

# Set model path (change if needed)
if [ -z "$MODEL_PATH" ]; then
    export MODEL_PATH="models/vit_convlstm_best.pt"
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "WARNING: Model file not found at $MODEL_PATH"
    echo "Please ensure the model file exists or set MODEL_PATH environment variable"
    echo ""
fi

echo "Starting API server on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python3 api.py

