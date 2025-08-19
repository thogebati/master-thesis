#!/bin/bash


echo "changing to venv"
source venv/deepface/bin/activate

echo "changing directory"
cd code

echo "analyzing faces"
python3.12 extract_facial_features.py

echo "deactivating venv"
deactivate