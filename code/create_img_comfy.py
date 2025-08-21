# Import necessary libraries
from tqdm import tqdm
import comfy_api_helper as capi
import datetime
import os
import pandas as pd
import warnings
import sys
import signal

# Suppress FutureWarnings from pandas
warnings.filterwarnings('ignore', category=FutureWarning)

# Global variables to control stopping the batch process
stop: bool = False
n: int = 0

def set_stop(val: bool) -> None:
    """
    Set the global stop flag to the given value and increment the counter.

    Args:
        val (bool): Value to set the stop flag.
    """
    global stop, n
    stop = val
    n += 1

def signal_handler(sig: int, frame) -> None:
    """
    Signal handler for SIGINT (Ctrl+C). Sets the stop flag.

    Args:
        sig (int): Signal number.
        frame: Current stack frame.
    """
    print("CTRL + C pressed. Batch will be finished and then program will stop")
    set_stop(True)
    
# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)

# File paths for workflow, negative prompts, and output directories
workflow_dir = '../comfy_prompts/sd3.5_simple_example.json'
negative_prompt_dir = '../comfy_prompts/negative_prompts.txt'

final_dir = '../results/'
out_dir = '../results/images'
batches_dir = '../results/batches'

# If a log file is provided as a command-line argument, resume from it
if len(sys.argv) > 1:
    log = pd.read_csv(sys.argv[1])
    runtime = sys.argv[1].split('/')[-1].replace(".csv", '')
    tmp = pd.read_csv(os.path.join(final_dir, f'image_dirs_{runtime}.csv'))
else:
    # Otherwise, start a new run with a timestamp
    runtime = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    tmp = pd.DataFrame(columns=['image_id', 'path', 'with_negative_prompt'])
    log = pd.DataFrame(columns=['file'])

# Path for logging progress
log_dir = os.path.join('../results/logs', f'{runtime}.csv')

# Ensure output directories exist
os.makedirs(out_dir, exist_ok=True)
out_dir = os.path.join(out_dir, runtime)
os.makedirs(out_dir, exist_ok=True)
out_dir = os.path.join(out_dir, 'face_img')

# Read negative prompts from file
fs = open(negative_prompt_dir)
negative_prompts = fs.read()
fs.close()

# Configuration for image generation
cfg: float = 5.0
steps: int = 60

# Load the workflow for image generation
workflow = capi.load_workflow(workflow_dir)

# Iterate over all CSV files in the batches directory
for csv in os.listdir(batches_dir): 
    # Skip directories
    if os.path.isdir(os.path.join(batches_dir, csv)):
        continue
    # Skip files already processed (logged)
    if log['file'].str.contains(csv).sum() > 0:
        continue
    # Read prompts from the batch CSV, skipping the first row
    prompts = pd.read_csv(os.path.join(batches_dir, csv), skiprows=1)
    print(f"starting to generate images for batch: {csv}")
    # Progress bar for the batch
    pbar = tqdm(prompts.iterrows(), total=len(prompts), position=0, leave=True, desc="batches")
    for index, row in pbar:
        # Prepare file names for images with and without negative prompts
        file_name_with = f"{row[0]}_{runtime}_with_negative.png"
        file_name_with_out = f"{row[0]}_{runtime}_without_negative.png"
        save_path_with = os.path.join(out_dir, file_name_with)
        save_path_without = os.path.join(out_dir, file_name_with_out)
        # Generate image without negative prompt
        capi.prompt_to_image(
            workflow=workflow,
            positve_prompt=row[1],
            output_path=out_dir,
            file_name=save_path_without,
            negative_prompt='',
            steps=steps,
            cfg=cfg
        )
        # Generate image with negative prompt
        capi.prompt_to_image(
            workflow=workflow,
            positve_prompt=row[1],
            output_path=out_dir,
            file_name=save_path_with,
            negative_prompt=negative_prompts,
            steps=steps,
            cfg=cfg
        )
        # Log the generated image paths
        tmp.loc[len(tmp)] = [row[0], save_path_with, True]
        tmp.loc[len(tmp)] = [row[0], save_path_without, False]
    
    # Log the processed batch file
    log.loc[len(log)] = [csv]
    log.to_csv(log_dir, index=False)
    tmp.to_csv(os.path.join(final_dir, f'path_comfy_images_{runtime}.csv'), index=False)
    # If stop flag is set, exit after finishing the current batch
    if n > 0:
        sys.exit(0)



