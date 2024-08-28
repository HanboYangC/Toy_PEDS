import numpy as np
import matplotlib.pyplot as plt
import torch

class Patch(object):
    def __init__(self,x,p):
        self.x=x
        self.p=p
class Geometry_1D(object):
    def __init__(self,lengths,params):
        if len(lengths)!=len(params['anchors']):
            print('Invalid parameters')
            return
        self.params=params
        self.anchors = self.params['anchors']
        self.lengths = lengths
        self.width = self.params['width']
        self.add_patches()
        self.d_hole=self.params['d_hole']
        self.d_med=self.params['d_med']
        return
    def add_patches(self):
        patch_list = []
        for i,p in enumerate(self.lengths):
            x = self.anchors[i]
            patch = Patch(x,p)
            patch_list.append(patch)
        self.patches = patch_list
        return patch_list

    def get_grid(self,N:int):
        grid=np.full((N), True, dtype=bool)
        hx=self.width/N
        x=[n*hx for n in range(N)]
        for patch in self.patches:
            lp=patch.x-patch.p/2
            rp=patch.x+patch.p/2
            if isinstance(lp, torch.Tensor):
                lp = lp.cpu().numpy() if lp.is_cuda else lp.numpy()
            if isinstance(rp, torch.Tensor):
                rp = rp.cpu().numpy() if rp.is_cuda else rp.numpy()
            ln=int(np.floor(lp/hx))
            rn=int(np.floor(rp/hx))
            if ((rp/hx)%1==0):
                rn-=1
            grid[ln:rn+1]=False
        return grid

    def get_D(self,N:int):
        grid=self.get_grid(N)
        D=np.where(grid, self.d_med, self.d_hole)
        return D
    def plot(self,N:int):
        D=self.get_D(N)
        # potential = np.where(grid, 1, 0)
        plt.step(np.linspace(0, self.width, len(D)), D)
        plt.show()


    @staticmethod
    def plot_grid(grid: np.ndarray,width):
        if grid is None:
            print('Grid is None !')
        else:
            potential=np.where(grid,1,0)
            plt.step(np.linspace(0,width,len(potential)),potential)
            plt.show()