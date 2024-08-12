import numpy as np
from Geometry_1D import Geometry_1D as G1D
from tqdm import tqdm

class DataGenerator:
    def __init__(self,params):
        self.params = params
        self.num_sample=self.params['num_samples']
        self.num_wells=self.params['num_wells']
        self.width=self.params['width']
        self.grid_width = self.width / self.num_wells
        self.anchors=np.linspace(0,self.width,self.num_wells+1)[:self.num_wells]+(self.grid_width/2)
        self.params['anchors']=self.anchors
        return

    def generate_geo(self,seed=None):
        if seed is not None:
            np.random.seed(seed)
        geo_list=[]
        for i in tqdm(range(self.num_sample),desc='Generating data'):
            lengths=np.random.rand(self.num_wells)*self.grid_width
            geo=G1D(lengths,self.params)
            geo_list.append(geo)
        return geo_list


