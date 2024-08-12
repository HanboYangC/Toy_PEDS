from DataGenerator import DataGenerator as DG
from tqdm import tqdm
import utils as ut
SEED=42

#%%
params={'num_samples':1000,
        'num_wells':4,
        'width':10,
        'd_hole':0.1,
        'd_med':1,
        'HF_N':512,
        'LF_N':16,
        'num_train':700,
        'num_test':200,
        'num_val':100

        }
#%%
'''Generate Data'''
dg=DG(params)
geo_list=dg.generate_geo(seed=SEED)
lengths_list=[]
y_list=[]
for geo in tqdm(geo_list,desc='Labeling data'):
        lengths=geo.lengths
        y=ut.label_lengths(lengths,params)
        y_list.append(y)
