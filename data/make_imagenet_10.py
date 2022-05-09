import os
import shutil
from tqdm import tqdm
import json

#inst-01
imagenet_loc = '/nobackup/ImageNet'
imagenet20_loc = '/nobackup/dataset_myf/ImageNet10'

class_dict =   {"warplane": "n04552348", "sports car":"n04285008", 
        'brambling bird':'n01530575', "Siamese cat": 'n02123597', 
        'antelope': 'n02422699', 'swiss mountain dog':'n02107574',
         "bull frog":"n01641577", 'garbage truck':"n03417042",
         "horse" :"n02389026", "container ship": "n03095699"}

for split in ['train', 'val']:
    for cls in class_dict.values():
        shutil.copytree(os.path.join(imagenet_loc, split, cls), os.path.join(imagenet20_loc, split, cls), dirs_exist_ok=True)