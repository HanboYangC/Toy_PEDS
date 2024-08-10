import numpy as np
import matplotlib.pyplot as plt

class Patch(object):
    def __init__(self,x,y,p):
        self.x=x
        self.y=y
        self.p=p


class Geometry_2D(object):
    def __init__(self,params,width,height,anchors:np.ndarray):
        '''The left bottom is the origin ,which is contradictory with the index rules for array.
        If you wanna plot the geometry, flip it first.'''
        if len(params)!=len(anchors):
            print('Invalid parameters')
            return
        self.params = params
        self.width,self.height = width,height
        self.anchors = anchors
        self.add_patches()
        return

    def add_patches(self):
        patch_list = []
        for i,p in enumerate(self.params):
            x,y = self.anchors[i]
            patch = Patch(x,y,p)
            patch_list.append(patch)
        self.patches = patch_list
        return patch_list

    def get_grid(self,N:int):
        grid=np.full((N,N), True, dtype=bool)
        hx,hy=self.width/N,self.height/N
        x=[n*hx for n in range(N)]
        y=[n*hy for n in range(N)]
        for patch in self.patches:
            lp=patch.x-patch.p/2
            rp=patch.x+patch.p/2
            dp=patch.y-patch.p/2
            up=patch.y+patch.p/2

            ln=int(np.floor(lp/hx))
            rn=int(np.floor(rp/hx))
            dn=int(np.floor(dp/hy))
            un=int(np.floor(up/hy))

            if ((rp/hx)%1==0):
                rn-=1
            if ((up/hy)%1==0):
                un-=1
            grid[ln:rn+1,dn:un+1]=False
        return grid

    @staticmethod
    def plot(grid: np.ndarray):
        if grid is None:
            print('Grid is None !')
        else:
            grid = np.transpose(grid)
            plt.imshow(grid, origin='lower', cmap='viridis')
            plt.colorbar()
            plt.show()



