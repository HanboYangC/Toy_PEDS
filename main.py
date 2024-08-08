from Geometry_1D import Geometry_1D as G1D
import numpy as np
from HF_1D import Diffusion_HF_1F as HF1
from matplotlib import pyplot as plt

# width=10
# height=10
# params=np.array([1,2,3,4])
# # x=np.array([0.25,0.25,0.75,0.75])
# # y=np.array([0.25,0.75,0.25,0.75])
# anchors=np.array([[2.5,2.5],[2.5,7.5],[7.5,2.5],[7.5,7.5]])
# geo=Geometry(params,height,width,anchors)
# grid=geo.get_grid(N=4)
# Geometry.plot(grid)

width=10
params=np.array([1,2,1,1.5])
anchors=np.array([1.25,3.75,6.25,8.75])
geo=G1D(params,width,anchors)
grid=geo.get_grid(N=20)
# G1D.plot(grid,width=width)

hf=HF1(N=20,T=400)
hf.add_geometry(geo)
U=hf.solve()
plt.plot(U)
plt.show()