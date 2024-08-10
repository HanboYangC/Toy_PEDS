import numpy as np
from Geometry_2D import Geometry

class Diffusion_HF():
    def __init__(self,N:int,T:int):
        super().__init__()
        self.T=T
        d_hole=0.1
        d_med=1
    def add_geometry(self,geo:Geometry):
        self.geo = geo
        print('Geometry added.')
        return self.geo
    def solve(self):
        grid=self.geo.get_grid(self.N)
        U=np.rand(grid.shape[0],grid.shape[1])
        D=np.where(grid, 1.0, 0.1)
        for t in range(self.T):
            u_=U.copy()
            for i in range(len(U)):
                for j in range(len(U[0])):
                    uij=U[i,j]
                    if j+1>=len(U[0]):
                        uu=0
                    else:
                        uu=U[i,j+1]

                    if i+1>=len(U):
                        ur=U[0,j]
                    else:
                        ur=U[i+1,j]

                    if j-1<0:
                        ud=1
                    else:
                        ud=U[i,j-1]

                    if i-1<0:
                        ul=U[-1,j]
                    else:
                        ul=U[i-1,j]
                u_[i,j]=(D[i,j]/






