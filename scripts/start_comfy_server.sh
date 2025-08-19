#!/bin/bash

echo "changing directory"

echo "changing to venv"
source venv/comfy/bin/activate

cd ComfyUI

echo "running model"
python3.12 main.py

echo "deactivating venv"
deactivate