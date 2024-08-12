from FD_1D import Diffusion_FD_1D as FD1
from Geometry_1D import Geometry_1D as G1D

def label_D(D):
    hf=FD1()
    return hf.solve(D)

def label_lengths(lengths,params,N=100):
    if N<100:
        print('Warning: The N is too small for high fidelity solver')
    geo=G1D(lengths,params)
    D=geo.get_D(N)
    hf=FD1()
    y=hf.solve(D,plot=False)
    return y