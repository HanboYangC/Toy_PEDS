import numpy as np
from Geometry_1D import Geometry_1D as G1D

class DataGenerator:
    def __init__(self,num_sample,num_wells,width):
        self.num_sample=num_sample
        self.num_wells=num_wells
        self.width=width
        self.grid_width = self.width / self.num_wells
        self.xs=np.linspace(0,self.width,self.num_wells+1)[:num_wells]+(self.grid_width/2)
        return

    def generate_geo(self,seed=None):
        if seed is not None:
            np.random.seed(seed)
        geo_list=[]
        for i in range(self.num_sample):
            lengths=np.random.rand(self.num_wells)*self.grid_width
            geo=G1D(lengths,self.width,self.xs)
            geo_list.append(geo)
        return geo_list


