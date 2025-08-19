from deepface import DeepFace
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer
from enum import Enum
import json
import os
import numpy as np
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# change before running
face_log_path = ''
if face_log_path == '':
    print("please set face_log_path")
    sys.exit(1)

final_dir = 'results'


imdir_ind = face_log_path.find('image_dirs_')
dot_ind = face_log_path.find('.')
date = face_log_path[imdir_ind + len('image_dirs_'): dot_ind]

img_df = pd.read_csv(face_log_path)

# Create a custom JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)  # Convert np.float32 to Python's native float
        if isinstance(obj, np.int64):
            return int(obj)

        return json.JSONEncoder.default(self, obj)


models = [ 'ssd', 'retinaface', 'mtcnn',  'opencv']


for model in models:
    demographies = pd.DataFrame(columns=['image_id', 'demography', 'path', 'with_negative_prompt'])

    pbar = tqdm(img_df.iterrows(), total=len(img_df), position=0, leave=True, desc="images")
    for index, row in pbar:
        print(row[1])
        try:
            demography = DeepFace.analyze(row[1], detector_backend=model)
            # print(demography)
            demographies.loc[len(demographies)] = [row[0], json.dumps(demography, cls=NumpyEncoder), row[1], row[2]]
            
            
        except ValueError:
            demographies.loc[len(demographies)] = [row[0], None, row[1], row[2]]
            # print("bad image found " + row[0])
            
            
    demographies.to_csv(os.path.join(final_dir, f'facial_features_{model}_{date}.csv'), index=False)
