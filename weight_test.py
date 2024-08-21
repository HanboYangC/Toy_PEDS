import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from DiffusionModel import DiffusionModel as DM
from DataGenerator import DataGenerator as DG
from sklearn.model_selection import train_test_split
#%%
LF_N=4
ws_s=np.linspace(0, 1, 11)
weight_loss_ds_path= f'weights/weights_LF={LF_N}_dense/weights_LF={LF_N}_dense.npy'
weight_loss_sp_path= f'weights/weights_LF={LF_N}_sparse/weights_LF={LF_N}_sparse.npy'
#%%
w_4_loss_sp=np.load(weight_loss_sp_path)
w_4_loss_sp_avg=np.nanmean(w_4_loss_sp,axis=1)
plt.plot(ws_s, w_4_loss_sp_avg)
plt.show()
#%%
min_index = np.argmin(w_4_loss_sp_avg)
w_d=ws_s[min_index]
ws_d = np.linspace(max(0, w_d - 0.1), min(0.99, w_d + 0.1),11)
w_4_loss_ds=np.load(weight_loss_ds_path)
w_4_loss_ds_avg=np.nanmean(w_4_loss_ds,axis=1)
plt.plot(ws_d,w_4_loss_ds_avg)
plt.show()
#%%
'''Now Test Performance'''
'''Define some params'''
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
        'epochs':100,
        'w':0
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
SEED=42
#%%
dg = DG(params)
geo_list = dg.generate_geo(seed=SEED)
# lengths_array = np.zeros((params['num_samples'], params['num_wells']))
# y_array = np.zeros((params['num_samples'], params['HF_N']))

y_array=np.load(y_dir+'.npy')
lengths_array=np.load(lengths_dir+'.npy')
k_array=np.load(k_dir+'.npy')

y_train, y_temp, lengths_train, lengths_temp, k_train, k_temp = train_test_split(
    y_array, lengths_array, k_array, train_size=params['num_train'], random_state=SEED
)

y_test, y_val, lengths_test, lengths_val, k_test, k_val = train_test_split(
    y_temp, lengths_temp, k_temp, train_size=params['num_test'], test_size=params['num_val'], random_state=SEED
)
#%%
input_dim=lengths_array.shape[1]
dm=DM(input_dim=input_dim,geometry_dim=params['LF_N'])
weight_path=f'weights/weights_LF={params['LF_N']}_sparse/w=0.40_0.pth'
dm.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'), weights_only=True))

k_pred = dm.forward(torch.from_numpy(lengths_test).float(), params).detach().numpy()
#%%
plt.scatter(k_pred, k_test)
plt.show()