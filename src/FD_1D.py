import numpy as np
import torch

from Geometry_1D import Geometry_1D as G1D
from matplotlib import pyplot as plt
import utils as ut
from scipy.stats import pearsonr
class Diffusion_FD_1D():
    def __init__(self,thre=1-1e-5):
        super().__init__()
        self.thre=thre
        self.d_hole=0.1
        self.d_med=1
        self.geo=None
    def add_geometry(self,geo:G1D):
        self.geo = geo
        print('Geometry added.')
        return self.geo

    def solve(self,N=None,D:torch.Tensor=None,plot=True):
        if D is None:
            print('D is None')
        if self.geo is None:
            width=10
        else:
            width=self.geo.width
        A,b=ut.get_Ab(D)

        try:
            u = torch.linalg.solve(A, b)
        except RuntimeError as e:
            if 'singular' in str(e):
                print('Matrix A is singular, using pseudoinverse.')
                u = torch.linalg.pinv(A) @ b
        u=u.reshape((-1))
        if plot:
            plt.plot(np.linspace(0,width,len(u)), u)
            plt.show()
        return u






