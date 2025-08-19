#!/bin/bash

dataset_dir='datasets'

open_images_dir=$dataset_dir/'open-images'

csv_dir=$open_images_dir/'csv'
json_dir=$open_images_dir/'json'


mkdir $dataset_dir
mkdir $open_images_dir
mkdir $csv_dir
mkdir $json_dir
mkdir venv

# initialize .venv venv
echo "creating .venv for all jupyter notebooks"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --quiet
deactivate

echo "downloading Open Image files"

# download Open Image V6 files
while read url; do
    echo "downloading $url"
    if [[ $url == *"csv"* ]]; then
        wget $url -P $csv_dir
    else
        wget $url -P $json_dir
    fi
done < open_images_urls.txt

# get comfy UI for specific commit
echo "cloning comfy UI"
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
git checkout 7ebd8087ffb9c713d308ff74f1bd14f07d569bed
cd ..

# echo "downloading SD3.5 Large safetensor"
wget https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/sd3.5_large_fp8_scaled.safetensors -P ComfyUI/models/checkpoints

# create venv for comfy
echo "creating venv for comfy and downloading requirements"
python3.12 -m venv venv/comfy
source venv/comfy/bin/activate
/home/thogebati/Master/master-thesis/venv/comfy/bin/python -m pip install -r ComfyUI/requirements.txt --quiet
/home/thogebati/Master/master-thesis/venv/comfy/bin/python -m pip install pandas requests_toolbelt websocket-client --quiet
deactivate


# #get DeepFace specific commit
echo "cloning DeepFace"
git clone https://github.com/serengil/deepface.git
cd deepface
git checkout c7c6739c5f36c401da2bc0e25d5eca0b2e2fecd9
cd ..

# # create venv for deepface
echo "creating venv for deepface and downloading requirements"
python3.12 -m venv venv/deepface
source venv/deepface/bin/activate
cd deepface
pip install --quiet -e . 
pip install tf-keras --quiet
cd ..
deactivate


# #get MiDaS specific commit
echo "cloning MiDaS"
git clone https://github.com/isl-org/MiDaS.git
cd MiDaS
git checkout 454597711a62eabcbf7d1e89f3fb9f569051ac9b
cd ..

# create venv for MiDaS
echo "creating venv for MiDaS and downloading requirements"
source ~/miniconda3/bin/activate
conda env create -f MiDaS/environment.yaml
conda deactivate

# download MiDaS model
echo "downloading MiDaS model"
wget -P MiDaS/weights https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt

# copy run file to midas
echo "copying run file to midas"
cp code/midas_run.py MiDaS