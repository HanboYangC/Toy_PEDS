from Geometry_1D import Geometry_1D as G1D
import numpy as np
from FD_1D import Diffusion_FD_1D as FD1
import torch
import numpy as np
from LF_1D import LF_Layer as LF
from matplotlib import pyplot as plt


#%%

width=10
lengths=np.array([1])
anchors=np.array([5])
geo=G1D(lengths,width,anchors)
d_hole = 0.1
d_med = 1
# #%%
# fd=FD1(thre=1-1e-5)
# fd.add_geometry(geo)
# fd.solve(N=4)
# fd.solve(N=8)
# fd.solve(N=16)
# fd.solve(N=32)
# fd.solve(N=64)
# fd.solve(N=128)
# fd.solve(N=256)
#%%
'''Test LF Layer'''
device='cpu'
N=4
grid=geo.get_grid(N=N)
D_tensor=torch.from_numpy(np.where(grid, d_med, d_hole))
print("D Tensor:", D_tensor)

params={
    'N':N,
    'h':geo.width/N
}

output=LF.apply(D_tensor,params)
output_np=output.detach().numpy()
# print("LF output:", output)
# plt.plot(output_np)
# plt.show()
grad=LF.backward(D_tensor,params,output)
