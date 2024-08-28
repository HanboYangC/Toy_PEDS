import numpy as np
import os
params = {'num_samples': 1000,# Total number of samples needed to generate high fidelity data
          'num_wells': 4,# The diffusion coefficients are binary values. This param means the number of low regions of the diffusion coefficient.
          'width': 10,# Total width of the whole geometry
          'd_hole': 0.1,# The low value for diffusion coefficients
          'd_med': 1,# The high value for diffusion coefficients
          'HF_N': 128,# Number of samples in HF solver
          'LF_N': 4,# Number of samples in LF solver
          'num_train': 700,# Number of training samples
          'num_test': 200,# Number of testing samples
          'num_val': 100,# Number of validation samples
          'epochs': 100,# Training Epoch (The model stores the weight with the best performance in the whole training process, so this param can be very high, but can't be lower than 75.)
          }
proj_path='.' # The working directory, should be set the same as the project dir. If you cd to the project dir before running the codes, this variable can be keep as '.'
data_dir = os.path.join(proj_path,'data')# Dir where you save the data
weight_name='weights'# Name of weight folder where you gonna save the weights. The code will create this folder under the project directory.

w_list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99] # The input of the LF is a weighted sum of downsampled geometry and generated geometry. This is the weight of downsampled geometry.