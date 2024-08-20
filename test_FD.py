from DataGenerator import DataGenerator as DG
from tqdm import tqdm
import utils as ut
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from LF_1D import LF_Layer as LF
import torch
from DiffusionModel import DiffusionModel as DM
SEED=42
from FD_1D import Diffusion_FD_1D as FD
#%%
# Set the Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
params={'num_samples':6000,
        'num_wells':4,
        'width':10,
        'd_hole':0.1,
        'd_med':1,
        'HF_N':128,
        'LF_N':16,
        'num_train':4200,
        'num_test':1200,
        'num_val':600,
        'w':0.7
        }
grid_width = params['width'] / params['num_wells']
params['grid_width'] = grid_width
anchors=np.linspace(0,params['width'],params['num_wells']+1)[:params['num_wells']]+(params['grid_width']/2)
params['anchors']=anchors
# data_dir='./data'
# lengths_dir=os.path.join(data_dir,'lengths')
# y_dir=os.path.join(data_dir,'y')
# k_dir=os.path.join(data_dir,'k')
print(f'device:{device}')

dg = DG(params)
geo_list = dg.generate_geo(seed=SEED)

num=5
for i in tqdm(range(num)):
    geo=geo_list[i]
    d=geo.get_D(params['LF_N'])
    fd=FD()
    fd.add_geometry(geo)
    y=fd.solve(params['LF_N'],d,plot=True)
    