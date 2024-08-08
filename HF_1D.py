import numpy as np
from Geometry_1D import Geometry_1D as G1D
class Diffusion_HF_1F():
    def __init__(self,N:int,T:int):
        super().__init__()
        self.T=T
        self.d_hole=0.1
        self.d_med=1
        self.N=N
    def add_geometry(self,geo:G1D):
        self.geo = geo
        print('Geometry added.')
        return self.geo

    def solve(self):
        grid = self.geo.get_grid(self.N)
        U = np.random.rand(len(grid))
        D = np.where(grid, self.d_med, self.d_hole)
        for t in range(self.T):
            U_ = U.copy()
            for i in range(len(U)):
                ul=U[i-2] if i-2 >= 0 else 0
                ur=U[i+2] if i+2 < len(U) else 1
                dl=D[i-1] if i-1 >= 0 else 1
                dr=D[i+1] if i+1 < len(D) else 1
                U_[i]=(dl*ul+dr*ur)/(dl+dr)
            U=U_
        return U





