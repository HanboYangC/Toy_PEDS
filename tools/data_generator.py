from src.DataGenerator import DataGenerator as DG
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import tools.config as config
from src.DiffusionModel import DiffusionModel as DM
from tqdm import tqdm
import src.utils as ut
SEED=42

params=config.params
data_dir=config.data_dir
w_list=config.w_list

device = 'cuda' if torch.cuda.is_available() else 'cpu'
grid_width = params['width'] / params['num_wells']
params['grid_width'] = grid_width
anchors = np.linspace(0, params['width'], params['num_wells'] + 1)[:params['num_wells']] + (
            params['grid_width'] / 2)
params['anchors'] = anchors
lengths_dir = os.path.join(data_dir, 'lengths')
y_dir = os.path.join(data_dir, 'y')
k_dir = os.path.join(data_dir, 'k')
D_dir = os.path.join(data_dir, 'D')
print(f'device:{device}')

dg = DG(params)
geo_list = dg.generate_geo(seed=SEED)

# Initialize DG and other variables
dg = DG(params)
geo_list = dg.generate_geo(seed=SEED)
lengths_array = np.zeros((params['num_samples'], params['num_wells']))
y_array = np.zeros((params['num_samples'], params['HF_N']))
D_array = np.zeros((params['num_samples'], params['HF_N']))

# label and save the data
log_interval = 100
log_file_path = './data_labeling_log.txt'
# Check if the log file exists, delete it if it does
if os.path.exists(log_file_path):
    os.remove(log_file_path)

# Create the log file
with open(log_file_path, 'w') as log_file:
    log_file.write("Log file created\n")

# Processing loop
for i, geo in enumerate(tqdm(geo_list, desc='Labeling data')):
    lengths = geo.lengths
    lengths_array[i] = lengths
    y = ut.label_lengths(lengths, params)
    y_array[i] = y
    D_array[i]=geo.get_D(params['HF_N'])
    if (i + 1) % log_interval == 0:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Processed {i + 1} samples\n")

np.save(y_dir,y_array)
np.save(lengths_dir,lengths_array)
np.save(D_dir,D_array)


#%%
'''Label k'''
y_array = np.load(y_dir + '.npy')
k_list=[]
for i, geo in enumerate(tqdm(geo_list, desc='Labeling k')):
    D=geo.get_D(params['HF_N'])
    k=ut.give_K(D,y_array[i],params)
    k_list.append(k)
k_array=np.array(k_list)
np.save(k_dir + '.npy',k_array)