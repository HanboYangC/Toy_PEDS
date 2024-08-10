from Geometry_1D import Geometry_1D as G1D
import numpy as np
from FD_1D import Diffusion_FD_1D as FD1
import torch
import numpy as np
from LF_1D import LF_Layer as LF
from matplotlib import pyplot as plt

#%%
width=10
params=np.array([1,1,1,1])
anchors=np.array([1.25,3.75,6.25,8.75])
geo=G1D(params,width,anchors)
d_hole = 0.1
d_med = 1
#%%
fd=FD1(thre=1-1e-5)
fd.add_geometry(geo)
fd.solve(N=4)
fd.solve(N=8)
fd.solve(N=16)
fd.solve(N=32)
fd.solve(N=64)
fd.solve(N=128)
fd.solve(N=256)
#%%
'''Test LF Layer'''
device='cpu'
grid=geo.get_grid(N=64)
D_tensor=torch.from_numpy(np.where(grid, d_med, d_hole))
print("D Tensor:", D_tensor)
output=LF.apply(D_tensor)
output_np=output.detach().numpy()
print("LF output:", output)
plt.plot(output_np)
plt.show()