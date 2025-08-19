# master-thesis
This is the repository for the master thesis "Uncovering Group Image Representations in Text-conditional Image Generation"

# Initialization

To initialize the workspace and download everything that is necessary to run the code, run the following:
```console
bash init.sh
```

In the following run all Jupyter Nottebooks witht he .venv environment.


# Running the code to get the final results

## 1. Downloading and curating prompts

To download and curated the necessary prompts run all cells in the code/prompt_curation.ipynb jupyter notebook.

## 2. Create the images

To create the images run the following scripts in two different terminals:

First start the ComfyUI server that runs the models:
```console
bash scripts/start_comfy_server.sh
```
Wait until the server is up and running.  
Now you can run the ComfyUI client script:

```console
bash scripts/create_img_comfy.sh $1
```

This scripts requires the path to the result csv file that contains the prompts.  
This will run for a very long time. To stop it just press CTRL+C in the terminal.

## 3. Create the depth images and pfms

With the image created now the depth images and pfms can be created.  
To create the images run the following script in a terminal:

```console
bash scripts/run_midas.sh $1 $2
```

The first parameter should be the path to where the generated images are stored and the second parameter should be the path to where the depth images should be stored.

## 4. Extract demographic attributes with DeepFace

Now the demographic attributes need to be extracted. To achieve this goal run the following script in a terminal. Before running this script you need to update the path to the images in the code/extract_facial_features.py file.


```console
bash scripts/extract_demographic_attributes.sh
```

## 5. Generating raw statistic data

With both the pfms created and the demographic attributes extracted now the raw statistic data can be generated. To achieve this goal run all cells in the code/create_statistic_data.ipynb notebook. Here again a few variables pointing to the previous results need to be filled.

## 6. Create Plots

With the raw statistical data generated the plots can be created. To achieve this goal run all cells in the code/create_plots.ipynb notebook. Here again a variable pointing to the previous results needs to be filled.

## 7. Create LaTeX tables and table figures

With the raw statistical data generated the LaTeX tables and table figures can be created. To achieve this goal run all cells in the code/create_latex_tables.ipynb notebook. Here again a variable pointing to the previous results needs to be filled.

