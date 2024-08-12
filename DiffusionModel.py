import torch

from LF_1D import LF_Layer
import torch.nn as nn
import utils as ut


# to reproduce same result for each running notebook
SEED = 42

class DiffusionModel(nn.Module):
    def __init__(self,input_dim,geometry_dim):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,geometry_dim)

    def forward(self,D:torch.tensor,params:dict):
        x=self.fc1(params)
        x=self.fc2(x)
        fine_D=self.fc3(x)
        w=params['w']
        final_D=w*(fine_D)+(1-w)*D
        x=LF_Layer.forward(final_D,params)
        return x

    # def load_data(self,lengths_list,y_list):


