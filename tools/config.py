import numpy as np
import os
params = {'num_samples': 1000,
          'num_wells': 4,
          'width': 10,
          'd_hole': 0.1,
          'd_med': 1,
          'HF_N': 128,
          'LF_N': 8,
          'num_train': 700,
          'num_test': 200,
          'num_val': 100,
          'epochs': 100,
          }
proj_path='.'
data_dir = os.path.join(proj_path,'data')

w_list=[0,0.3,0.6,0.9,0.99]