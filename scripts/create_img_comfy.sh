#!/bin/bash

echo "changing directory"

echo "changing to venv"
source venv/comfy/bin/activate

echo "running model"
cd code
python3.12 create_img_comfy.py $1

echo "deactivating venv"
deactivate