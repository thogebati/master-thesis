#!/bin/bash

echo "changing directory"

cd MiDaS/

echo "activating mini conda"
source ~/miniconda3/bin/activate

echo "activating venv"
conda activate midas-py310

echo "running midas"
python midas_run.py --input_path "$1" --output_path "$2"

echo "deactivating venv"
deactivate