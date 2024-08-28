from src.DataGenerator import DataGenerator as DG
import matplotlib.pyplot as plt
from src.LF_1D import LF_Layer as LF
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from src.DiffusionModel import DiffusionModel as DM
import shutil
SEED=42
#%%
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
          'epochs': 300,
          # 'w': 0.5
          }
data_dir = './data'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
grid_width = params['width'] / params['num_wells']
params['grid_width'] = grid_width
anchors = np.linspace(0, params['width'], params['num_wells'] + 1)[:params['num_wells']] + (
            params['grid_width'] / 2)
params['anchors'] = anchors
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
y_train, y_temp, lengths_train, lengths_temp, k_train, k_temp = train_test_split(
    y_array, lengths_array, k_array, train_size=params['num_train'], random_state=SEED
)

y_test, y_val, lengths_test, lengths_val, k_test, k_val = train_test_split(
    y_temp, lengths_temp, k_temp, train_size=params['num_test'], test_size=params['num_val'], random_state=SEED
)
#%%
weight_path='./weights/'
w_list=np.round(np.linspace(0, 0.9, 10), decimals=2).tolist()
w_list.append(0.99)
best_val_list=[]

# if os.path.exists(weight_path):
#     shutil.rmtree(weight_path)
os.makedirs(weight_path)
for w in w_list:
    params['w']=w
    input_dim=lengths_array.shape[1]
    dm=DM(input_dim,params)
    dm.load_data(lengths_train,k_train,lengths_val,k_val)
    train_losses, val_losses,best_val_loss=dm.fit(learning_rate=1e-5,save_path=os.path.join(weight_path,f'w={w}_LF_N={params['LF_N']}.pt'))
    best_val_list.append(best_val_loss)
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.title(f"Loss for w={w}")
    plt.legend()
    plt.show()
#0.0177
#%%
plt.plot(w_list,best_val_list)
plt.xlabel('weight w')
plt.ylabel('validation loss')
plt.show()