import numpy as np
import os
import torch
import src.utils
from src.DataGenerator import DataGenerator as DG
import matplotlib.pyplot as plt
from src.LF_1D import LF_Layer as LF
SEED=42
#%%
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

dg = DG(params)
geo_list = dg.generate_geo(seed=SEED)
lengths_array=np.load(lengths_dir+'.npy')
y_array=np.load(y_dir+'.npy')
k_array=np.load(k_dir+'.npy')
#%%
check_num=5
# # lengths_tensor=torch.from_numpy(lengths_array)
lf=LF(params=params)
# result_K,result_U=lf.forward(lengths_tensor[:check_num])

for i in range(check_num):
    geo=geo_list[i]
    D=torch.from_numpy(geo.get_D(params['HF_N'])).view((1,-1))
    # signal=result_U[i]
    result_K, result_U = lf.forward(D)
    plt.plot(result_U[0])
    plt.plot(D[0])
    plt.show()