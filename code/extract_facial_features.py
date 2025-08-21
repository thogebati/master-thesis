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

# Set the path to the CSV file containing image paths and metadata
face_log_path = ''
if face_log_path == '':
    print("please set face_log_path")
    sys.exit(1)

final_dir = 'results'

# Extract the date string from the filename for output naming
imdir_ind = face_log_path.find('image_dirs_')
dot_ind = face_log_path.find('.')
date = face_log_path[imdir_ind + len('image_dirs_'): dot_ind]

# Load the CSV file into a DataFrame
img_df = pd.read_csv(face_log_path)

# Custom JSON encoder to handle numpy types for serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)  # Convert np.float32 to Python's native float
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

# List of face detector backends to use for analysis
models = [ 'ssd', 'retinaface', 'mtcnn',  'opencv']

# Loop over each face detector model
for model in models:
    # Create an empty DataFrame to store results for this model
    demographies = pd.DataFrame(columns=['image_id', 'demography', 'path', 'with_negative_prompt'])

    # Progress bar for processing images
    pbar = tqdm(img_df.iterrows(), total=len(img_df), position=0, leave=True, desc="images")
    for index, row in pbar:
        print(row[1])  # Print the image path for debugging
        try:
            # Analyze the image using DeepFace with the specified detector backend
            demography = DeepFace.analyze(row[1], detector_backend=model)
            # Store the results in the DataFrame (demography is serialized as JSON)
            demographies.loc[len(demographies)] = [row[0], json.dumps(demography, cls=NumpyEncoder), row[1], row[2]]
        except ValueError:
            # If DeepFace fails (e.g., no face found), store None for demography
            demographies.loc[len(demographies)] = [row[0], None, row[1], row[2]]
            # print("bad image found " + row[0])

    # Save the results for this model to a CSV file
    demographies.to_csv(os.path.join(final_dir, f'facial_features_{model}_{date}.csv'), index=False)
