import numpy as np
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
    #
    def solve_it(self,N=None,D:np.ndarray=None,plot=True):
        if D is None:
            if N is None:
                print('Please provide N')
                return None
            grid = self.geo.get_grid(N)
            D = np.where(grid, self.d_med, self.d_hole)
        if self.geo is None:
            width=10
        else:
            width=self.geo.width
        U = np.random.rand(len(D))
        # for t in range(self.T):
        corr=0
        while corr<self.thre:
            U_ = U.copy()
            for i in range(len(U)):
                ul=U[i-2] if i-2 >= 0 else 0
                ur=U[i+2] if i+2 < len(U) else 1
                dl=D[i-1] if i-1 >= 0 else 1
                dr=D[i+1] if i+1 < len(D) else 1
                U_[i]=(dl*ul+dr*ur)/(dl+dr)
            corr, _ = pearsonr(U_, U)
            U=U_
        if plot:
            plt.plot(np.linspace(0,width,len(U)), U)
            plt.show()
        return U

    def solve(self,N=None,D:np.ndarray=None,plot=True):
        if D is None:
            if N is None:
                print('Please provide N')
                return None
            grid = self.geo.get_grid(N)
            D = np.where(grid, self.d_med, self.d_hole)
        if self.geo is None:
            width=10
        else:
            width=self.geo.width
        A,b=ut.get_Ab(D)
    
        try:
            u = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # print('Matrix A is singular, using pseudoinverse.')
            u = np.linalg.pinv(A) @ b
        u=u.reshape((-1))
        if plot:
            plt.plot(np.linspace(0,width,len(u)), u)
            plt.show()
        return u






