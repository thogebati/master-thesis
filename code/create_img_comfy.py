from tqdm import tqdm
import comfy_api_helper as capi
import datetime
import os
import pandas as pd
import warnings
import sys
import signal

warnings.filterwarnings('ignore', category=FutureWarning)

stop = False
n = 0

def set_stop(val: bool):
    stop = val
    n += 1

def signal_handler(sig, frame):
    print("CTRL + C pressed. Batch will be finished and then program will stop")
    set_stop(True)
    
signal.signal(signal.SIGINT, signal_handler)



workflow_dir = '../comfy_prompts/sd3.5_simple_example.json'
negative_prompt_dir = '../comfy_prompts/negative_prompts.txt'

final_dir = '../results/'
out_dir = '../results/images'
batches_dir = '../results/batches'

if len(sys.argv) > 1:
    log = pd.read_csv(sys.argv[1])
    runtime = sys.argv[1].split('/')[-1].replace(".csv", '')
    tmp = pd.read_csv(os.path.join(final_dir, f'image_dirs_{runtime}.csv'))
else:
    runtime = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    tmp = pd.DataFrame(columns=['image_id', 'path', 'with_negative_prompt'])
    log = pd.DataFrame(columns=['file'])

log_dir = os.path.join('../results/logs', f'{runtime}.csv')

os.makedirs(out_dir, exist_ok=True)
out_dir = os.path.join(out_dir, runtime)
os.makedirs(out_dir, exist_ok=True)
out_dir = os.path.join(out_dir, 'face_img')


fs = open(negative_prompt_dir)
negative_prompts= fs.read()
fs.close()
cfg = 5.0
steps = 60

workflow = capi.load_workflow(workflow_dir)

for csv in os.listdir(batches_dir): 
    if os.path.isdir(os.path.join(batches_dir, csv)):
        continue
    if log['file'].str.contains(csv).sum() > 0:
        continue
    prompts = pd.read_csv(os.path.join(batches_dir, csv), skiprows=1)
    print(f"starting to generate images for batch: {csv}")
    pbar = tqdm(prompts.iterrows(), total=len(prompts), position=0, leave=True, desc="batches")
    for index, row in pbar:
        file_name_with = f"{row[0]}_{runtime}_with_negative.png"
        file_name_with_out = f"{row[0]}_{runtime}_without_negative.png"
        save_path_with = os.path.join(out_dir, file_name_with)
        save_path_without = os.path.join(out_dir, file_name_with_out)
        capi.prompt_to_image(workflow=workflow, positve_prompt=row[1], output_path=out_dir, file_name=save_path_without, negative_prompt='', steps=steps, cfg=cfg)
        capi.prompt_to_image(workflow=workflow, positve_prompt=row[1], output_path=out_dir, file_name=save_path_with, negative_prompt=negative_prompts, steps=steps, cfg=cfg)
        tmp.loc[len(tmp)] = [row[0], save_path_with, True]
        tmp.loc[len(tmp)] = [row[0], save_path_without, False]
    
    
    
    log.loc[len(log)] = [csv]
    log.to_csv(log_dir, index=False)
    tmp.to_csv(os.path.join(final_dir, f'path_comfy_images_{runtime}.csv'), index=False)
    if n > 0:
        sys.exit(0)



