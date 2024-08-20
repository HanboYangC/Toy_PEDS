from DataGenerator import DataGenerator as DG
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from DiffusionModel import DiffusionModel as DM
SEED=42

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
        'epochs':200
        }
grid_width = params['width'] / params['num_wells']
params['grid_width'] = grid_width
anchors=np.linspace(0,params['width'],params['num_wells']+1)[:params['num_wells']]+(params['grid_width']/2)
params['anchors']=anchors
data_dir=os.path.join('.','data')
lengths_dir=os.path.join(data_dir,'lengths')
y_dir=os.path.join(data_dir,'y')
k_dir=os.path.join(data_dir,'k')
weights_dir=os.path.join('.','weights')
print(f'device:{device}')

dg = DG(params)
geo_list = dg.generate_geo(seed=SEED)
lengths_array = np.zeros((params['num_samples'], params['num_wells']))
y_array = np.zeros((params['num_samples'], params['HF_N']))

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
dm=DM(input_dim=input_dim,geometry_dim=params['LF_N'])
dm.load_data(lengths_train, k_train,lengths_val, k_val)

'''Search w'''
ws=np.linspace(0,0.99,10)

for w in ws:
    print(f'Testing w={w}')
    params['w']=w
    weight_path=os.path.join(os.path.join(weights_dir,'weights_1'),f'w_{w}.pth')
    train_losses, val_losses=dm.fit(params=params,learning_rate=0.1,save_path=weight_path)




