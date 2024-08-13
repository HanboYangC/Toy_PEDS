#!/usr/bin/env python
# coding: utf-8

# In[25]:


from DataGenerator import DataGenerator as DG
from tqdm import tqdm
import utils as ut
import matplotlib.pyplot as plt
import os
import numpy as np
SEED=42


# In[40]:


# Set the Parameters
params={'num_samples':6000,
        'num_wells':4,
        'width':10,
        'd_hole':0.1,
        'd_med':1,
        'HF_N':128,
        'LF_N':16,
        'num_train':700,
        'num_test':200,
        'num_val':100
        }
data_dir='./data'
lengths_dir=os.path.join(data_dir,'lengths')
y_dir=os.path.join(data_dir,'y')


# In[ ]:


# Generate and label the data

dg=DG(params)
geo_list=dg.generate_geo(seed=SEED)
lengths_array = np.zeros((params['num_samples'], params['num_wells']))
y_array = np.zeros((params['num_samples'], params['HF_N']))
for i,geo in enumerate(tqdm(geo_list,desc='Labeling data')):
        lengths=geo.lengths
        lengths_array[i]=lengths
        y=ut.label_lengths(lengths,params)
        y_array[i]=y

np.save(lengths_dir,lengths_array)
np.save(y_dir,y_array)


# In[31]:


# Simple Check
check_num=5
for i in range(5):
    d=geo_list[i].get_D(N=params['HF_N'])
    y=y_list[i]
    plt.plot(y)
    plt.plot(d)
    plt.show()


# In[ ]:




