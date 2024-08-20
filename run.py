#!/usr/bin/env python
# coding: utf-8

# In[1]:


from DataGenerator import DataGenerator as DG
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from DiffusionModel import DiffusionModel as DM
import utils as ut
from tqdm import tqdm
SEED=42


# In[2]:


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


# In[5]:

# Initialize DG and other variables
dg = DG(params)
geo_list = dg.generate_geo(seed=SEED)
lengths_array = np.zeros((params['num_samples'], params['num_wells']))
y_array = np.zeros((params['num_samples'], params['HF_N']))


# # In[ ]:
#
#
# # label and save the data
# log_interval = 100
# log_file_path = './data_labeling_log.txt'
# # Check if the log file exists, delete it if it does
# if os.path.exists(log_file_path):
#     os.remove(log_file_path)
#
# # Create the log file
# with open(log_file_path, 'w') as log_file:
#     log_file.write("Log file created\n")
#
# # Processing loop
# for i, geo in enumerate(tqdm(geo_list, desc='Labeling data')):
#     lengths = geo.lengths
#     lengths_array[i] = lengths
#     y = ut.label_lengths(lengths, params)
#     y_array[i] = y
#     if (i + 1) % log_interval == 0:
#         with open(log_file_path, 'a') as log_file:
#             log_file.write(f"Processed {i + 1} samples\n")
#
# np.save(y_dir,y_array)
# np.save(lengths_dir,lengths_array)
#
#
# #%%
# '''Label k'''
# y_array = np.load(y_dir + '.npy')
# k_list=[]
# for i, geo in enumerate(tqdm(geo_list, desc='Labeling k')):
#     D=geo.get_D(params['HF_N'])
#     k=ut.give_K(D,y_array[i],params)
#     k_list.append(k)
# k_array=np.array(k_list)
# np.save(k_dir + '.npy',k_array)
#
#
# #%%
# # Simple Check
# # lengths_array=np.load(lengths_dir+'.npy')
# y_array=np.load(y_dir+'.npy')
# check_num=5
#
#
# # d_LF=geo_list[:check_num].get_D(N=params['LF_N'])
# # d_LF_tensor=torch.tensor(d_LF)
# # y_lf=LF.apply(d_LF_tensor,params)
# for i in range(check_num):
#     d_HF = geo_list[i].get_D(N=params['HF_N'])
#     y_hf=y_array[i]
#     plt.plot(y_hf)
#     plt.plot(d_HF)
#     plt.show()


# In[25]:
# Split data
y_array=np.load(y_dir+'.npy')
lengths_array=np.load(lengths_dir+'.npy')
k_array=np.load(k_dir+'.npy')

y_train, y_temp, lengths_train, lengths_temp, k_train, k_temp = train_test_split(
    y_array, lengths_array, k_array, train_size=params['num_train'], random_state=SEED
)

y_test, y_val, lengths_test, lengths_val, k_test, k_val = train_test_split(
    y_temp, lengths_temp, k_temp, train_size=params['num_test'], test_size=params['num_val'], random_state=SEED
)


# In[ ]:

input_dim=lengths_array.shape[1]
dm=DM(input_dim=input_dim,geometry_dim=params['LF_N'],seed=42)
dm.load_data(lengths_train, k_train,lengths_val, k_val)
#%%
#epochs=125 is ok
train_losses, val_losses,_=dm.fit(params=params,learning_rate=1e-5)
#%%
plt.plot(train_losses, label='train')
plt.show()
plt.plot(val_losses, label='val')
plt.show()
#%%
k_pred = dm.forward(torch.from_numpy(lengths_val).float(), params).detach().numpy()
fig, ax = plt.subplots()
ax.plot(k_pred, k_pred)
ax.scatter(k_pred, k_val)
ax.set_aspect('equal', 'box')

# 显示图形
plt.show()
#%%

