#!/bin/bash
# This script sets up a Python virtual environment and installs the required packages.

python3 -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



