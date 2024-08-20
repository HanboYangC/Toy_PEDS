from DataGenerator import DataGenerator as DG
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from DiffusionModel import DiffusionModel as DM
import shutil
SEED=42
#%%
'''Define Parameters'''
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
        'epochs':200,
        'avg_num':10
        }
grid_width = params['width'] / params['num_wells']
params['grid_width'] = grid_width
anchors=np.linspace(0,params['width'],params['num_wells']+1)[:params['num_wells']]+(params['grid_width']/2)
params['anchors']=anchors

'''Define Path'''
data_dir=os.path.join('.','data')
lengths_dir=os.path.join(data_dir,'lengths')
y_dir=os.path.join(data_dir,'y')
k_dir=os.path.join(data_dir,'k')
weights_dir=os.path.join('.','weights')
print(f'device:{device}')

# dg = DG(params)
# geo_list = dg.generate_geo(seed=SEED)
# lengths_array = np.zeros((params['num_samples'], params['num_wells']))
# y_array = np.zeros((params['num_samples'], params['HF_N']))

'''Load Data'''
y_array=np.load(y_dir+'.npy')
lengths_array=np.load(lengths_dir+'.npy')
k_array=np.load(k_dir+'.npy')

y_train, y_temp, lengths_train, lengths_temp, k_train, k_temp = train_test_split(
        y_array, lengths_array, k_array, train_size=params['num_train'], random_state=SEED
)

y_test, y_val, lengths_test, lengths_val, k_test, k_val = train_test_split(
        y_temp, lengths_temp, k_temp, train_size=params['num_test'], test_size=params['num_val'], random_state=SEED
)

input_dim=lengths_array.shape[1]

'''Search w sparse'''
ws=np.linspace(0,1,11)
ws[-1]=0.99
all_loss=[]
weight_name_sparse = f'weights_LF={params['LF_N']}_sparse'

weight_dir_sparse = os.path.join(weights_dir, weight_name_sparse)
if os.path.exists(weight_dir_sparse):
    shutil.rmtree(weight_dir_sparse)
    print(f'Directory {weight_dir_sparse} exists, deleting contents.')

os.makedirs(weight_dir_sparse)
print(f'Directory {weight_dir_sparse} created.')
for w in ws:
    print(f'Testing w={w}')
    params['w']=w
    w_loss=np.zeros(params['avg_num'])
    for i in range(params['avg_num']):
        dm=DM(input_dim=input_dim,geometry_dim=params['LF_N'],seed=False)
        dm.load_data(lengths_train, k_train,lengths_val, k_val)
        weight_path=os.path.join(weight_dir_sparse, f'w={w:.2f}_{i}.pth')
        train_losses, val_losses,best_val_loss=dm.fit(params=params,learning_rate=0.1,save_path=weight_path)
        if best_val_loss is not None:
            w_loss[i]=best_val_loss
        else:
            w_loss[i]=np.nan
    all_loss.append(w_loss)
all_loss_np=np.array(all_loss)
np.save(os.path.join(weight_dir_sparse, weight_name_sparse), all_loss_np)
#%%
'''Search w Dense'''
mean_loss=np.nanmean(all_loss_np,axis=1)
min_index = np.argmin(mean_loss)
w_best=ws[min_index]
ws_d = np.linspace(max(0, w_best - 0.1), min(0.99, w_best + 0.1), 21)
all_loss=[]
weight_name_dense = f'weights_LF={params['LF_N']}_dense'

weight_dir_dense = os.path.join(weights_dir, weight_name_dense)
if os.path.exists(weight_dir_dense):
    shutil.rmtree(weight_dir_dense)
    print(f'Directory {weight_dir_dense} exists, deleting contents.')

os.makedirs(weight_dir_dense)
print(f'Directory {weight_dir_dense} created.')
for w in ws_d:
    print(f'Testing w={w}')
    params['w']=w
    w_loss=np.zeros(params['avg_num'])
    for i in range(params['avg_num']):
        dm=DM(input_dim=input_dim,geometry_dim=params['LF_N'],seed=False)
        dm.load_data(lengths_train, k_train,lengths_val, k_val)
        weight_path=os.path.join(weight_dir_dense, f'w={w:.2f}_{i}.pth')
        train_losses, val_losses,best_val_loss=dm.fit(params=params,learning_rate=0.1,save_path=weight_path)
        if best_val_loss is not None:
            w_loss[i]=best_val_loss
        else:
            w_loss[i]=np.nan
    all_loss.append(w_loss)
all_loss_np=np.array(all_loss)
np.save(os.path.join(weight_dir_dense, weight_name_dense), all_loss_np)

