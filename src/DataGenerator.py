import numpy as np
from Geometry_1D import Geometry_1D as G1D
from tqdm import tqdm
import os
import torch
import utils as ut
SEED=42
#%%
class DataGenerator:
    def __init__(self,params):
        self.params = params
        self.num_sample=self.params['num_samples']
        self.num_wells=self.params['num_wells']
        self.width=self.params['width']
        self.grid_width = self.width / self.num_wells
        # self.anchors=np.linspace(0,self.width,self.num_wells+1)[:self.num_wells]+(self.grid_width/2)
        self.anchors = self.params['anchors']
        return

    def generate_geo(self,seed=None):
        if seed is not None:
            np.random.seed(seed)
        geo_list=[]
        for i in tqdm(range(self.num_sample),desc='Generating data'):
            lengths=np.random.rand(self.num_wells)*self.grid_width
            geo=G1D(lengths,self.params)
            geo_list.append(geo)
        return geo_list


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {'num_samples': 10000,
              'num_wells': 4,
              'width': 10,
              'd_hole': 0.1,
              'd_med': 1,
              'HF_N': 128,
              'LF_N': 4,
              'num_train': 700,
              'num_test': 200,
              'num_val': 100,
              'epochs': 75,
              'w': 0.99
              }
    grid_width = params['width'] / params['num_wells']
    params['grid_width'] = grid_width
    anchors = np.linspace(0, params['width'], params['num_wells'] + 1)[:params['num_wells']] + (
                params['grid_width'] / 2)
    params['anchors'] = anchors
    data_dir = './data'
    lengths_dir = os.path.join(data_dir, 'lengths')
    y_dir = os.path.join(data_dir, 'y')
    k_dir = os.path.join(data_dir, 'k')
    print(f'device:{device}')

    dg = DataGenerator(params)
    geo_list = dg.generate_geo(seed=SEED)
    lengths_array = np.zeros((params['num_samples'], params['num_wells']))
    y_array = np.zeros((params['num_samples'], params['HF_N']))

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
        if (i + 1) % log_interval == 0:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Processed {i + 1} samples\n")

    np.save(y_dir,y_array)
    np.save(lengths_dir,lengths_array)

    '''Label k'''
    y_array = np.load(y_dir + '.npy')
    k_list=[]
    for i, geo in enumerate(tqdm(geo_list, desc='Labeling k')):
        D=geo.get_D(params['HF_N'])
        k=ut.give_K(D,y_array[i],params)
        k_list.append(k)
    k_array=np.array(k_list)
    np.save(k_dir + '.npy',k_array)