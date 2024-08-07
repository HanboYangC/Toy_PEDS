from Geometry import Geometry
import numpy as np

width=10
height=10
params=np.array([1,2,3,4])
# x=np.array([0.25,0.25,0.75,0.75])
# y=np.array([0.25,0.75,0.25,0.75])
anchors=np.array([[2.5,2.5],[2.5,7.5],[7.5,2.5],[7.5,7.5]])
geo=Geometry(params,height,width,anchors)
grid=geo.get_grid(N=4)
Geometry.plot(grid)