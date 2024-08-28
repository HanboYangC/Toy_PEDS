import numpy as np
import os
import torch

import utils
from DataGenerator import DataGenerator as DG
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
SEED=42
#%%
# Set the Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
params={'num_samples':10000,
        'num_wells':4,
        'width':10,
        'd_hole':0.1,
        'd_med':1,
        'HF_N':128,
        'LF_N':4,
        'num_train':700,
        'num_test':200,
        'num_val':100,
        'epochs':75,
        'w':0.99
        }
grid_width = params['width'] / params['num_wells']
params['grid_width'] = grid_width
anchors=np.linspace(0,params['width'],params['num_wells']+1)[:params['num_wells']]+(params['grid_width']/2)
params['anchors']=anchors
data_dir='./data'
lengths_dir=os.path.join(data_dir,'lengths')
y_dir=os.path.join(data_dir,'y')
k_dir=os.path.join(data_dir,'k')
print(f'device:{device}')
#%%
# Simple Check
lengths_array=np.load(lengths_dir+'.npy')
y_array=np.load(y_dir+'.npy')
k_array=np.load(k_dir+'.npy')
check_num=5
# Initialize DG and other variables
dg = DG(params)
geo_list = dg.generate_geo(seed=SEED)
#%%
for i in range(check_num):
    d_HF = geo_list[i].get_D(N=params['HF_N'])
    y_hf=y_array[i]
    plt.plot(y_hf,label='Temperature')
    plt.plot(d_HF,label='Diffusion Coefficients')
    plt.legend(loc="lower right")
    plt.show()
#%%
'check if k is constant'
check_id=185
N=y_array.shape[1]
mid_list=range(1,N-1)
k_test=[]
y=y_array[check_id]
for mid in mid_list:
    d_HF=geo_list[check_id].get_D(N=params['HF_N'])
    k_test.append(utils.give_K(d_HF,y,params,mid))
plt.plot(k_test)
plt.ylim(0,0.04)
plt.show()