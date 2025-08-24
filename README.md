# master-thesis
This is the repository for the master thesis "Uncovering Group Image Representations in Text-Conditional Image Generation."

# Initialization

To initialize the workspace and download everything that is necessary to run the code, run the following:
```console
bash init.sh
```

To make sure all necessary packages are installed, use the created virtual environment in the .venv folder when running the Jupyter Notebooks.


# Running the code to get the final results

## 1. Downloading and curating prompts

To download and curate the necessary prompts, run all cells in the code/prompt_curation.ipynb Jupyter Notebook.

## 2. Create the images

To create the images, run the following scripts in two different terminals:

First start the ComfyUI server that runs the models:
```console
bash scripts/start_comfy_server.sh
```
Wait until the server is up and running.  
Now you can run the ComfyUI client script:

```console
bash scripts/create_img_comfy.sh $1
```

This script requires the path to the result CSV file that contains the prompts.  
This will run for a very long time. To stop it, just press CTRL+C in the terminal.

## 3. Create the depth images and pfms

With the image created, now the depth images and PFMs can be created.  
To create the images, run the following script in a terminal:

```console
bash scripts/run_midas.sh $1 $2
```

The first parameter should be the path to where the generated images are stored, and the second parameter should be the path to where the depth images should be stored.

## 4. Extract demographic attributes with DeepFace

Now the demographic attributes need to be extracted. To achieve this goal, run the following script in a terminal. Before running this script, you need to update the path to the images in the code/extract_facial_features.py file.


```console
bash scripts/extract_demographic_attributes.sh
```

## 5. Generating raw statistical data

With both the PFMs created and the demographic attributes extracted, now the raw statistical data can be generated. To achieve this goal, run all cells in the code/create_statistic_data.ipynb notebook. Here again a few variables pointing to the previous results must be filled.

## 6. Create Plots

With the raw statistical data generated, the plots can be created. To achieve this goal, run all cells in the code/create_plots.ipynb notebook. Here again a variable pointing to the previous results needs to be filled.

## 7. Create LaTeX tables and table figures

With the raw statistical data generated, the LaTeX tables and table figures can be created. To achieve this goal, run all cells in the code/create_latex_tables.ipynb notebook. Here again a variable pointing to the previous results needs to be filled.


## The resulting pipeline

```mermaid

graph TD
    subgraph "Step 0: Initialization"
        A(Start) --> B["<b>Initialize Workspace</b><br><em>Sets up the environment, clones repositories, and downloads models</em>"];
    end

    subgraph "Step 1: Prompt Curation"
        B --> C["<b>Curate Prompts</b><br><em>Downloads and filters text prompts from public datasets like MS Coco and Open Images</em>"];
        C --> D([Curated & Filtered Prompts]);
    end

    subgraph "Step 2: Image Generation"
        D --> E["<b>Generate Images</b><br><em>Creates a large set of images from the curated prompts using a text-to-image model</em>"];
        E --> F([Generated Image Dataset]);
    end

    subgraph "Step 3 & 4: Parallel Image Analysis"
        F --> G["<b>Create Depth Maps</b><br><em>Analyzes each generated image to estimate its depth information</em>"];
        G --> H([Depth Images & PFM Files]);

        F --> I["<b>Extract Demographic Attributes</b><br><em>Analyzes each generated image to identify facial features like age and gender</em>"];
        I --> J([Demographic Attribute Data]);
    end

    subgraph "Step 5: Data Aggregation"
        H --> K;
        J --> K["<b>Generate Statistical Data</b><br><em>Combines the depth maps and demographic attributes into a unified dataset for analysis</em>"];
        K --> L([Raw Statistical Data]);
    end

    subgraph "Step 6 & 7: Final Output Generation"
        L --> M["<b>Create Plots</b><br><em>Visualizes the aggregated data to create plots for the thesis</em>"];
        M --> N([Final Plots]);

        L --> O["<b>Create LaTeX Tables</b><br><em>Formats the aggregated data into tables suitable for the thesis paper</em>"];
        O --> P([Final LaTeX Tables]);
    end

    subgraph "Finish"
        N --> Q(End);
        P --> Q(End);
    end

```


# Results

The extracted raw statistical data is supplied in this repo and can be found in the results folder.