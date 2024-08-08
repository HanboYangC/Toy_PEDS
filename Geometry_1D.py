import numpy as np
import matplotlib.pyplot as plt

class Patch(object):
    def __init__(self,x,p):
        self.x=x
        self.p=p
class Geometry_1D(object):
    def __init__(self,params,width,anchors:np.ndarray):
        '''The left bottom is the origin ,which is contradictory with the index rules for array.
        If you wanna plot the geometry, flip it first.'''
        if len(params)!=len(anchors):
            print('Invalid parameters')
            return
        self.params = params
        self.width = width
        self.anchors = anchors
        self.add_patches()
        return

    def add_patches(self):
        patch_list = []
        for i,p in enumerate(self.params):
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
            ln=int(np.floor(lp/hx))
            rn=int(np.floor(rp/hx))
            if ((rp/hx)%1==0):
                rn-=1
            grid[ln:rn+1]=False
        return grid

    @staticmethod
    def plot(grid: np.ndarray,width):
        if grid is None:
            print('Grid is None !')
        else:
            potential=np.where(grid,1,0)
            plt.step(np.linspace(0,width,len(potential)),potential)
            plt.show()