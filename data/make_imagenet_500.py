import os
import shutil
from tqdm import tqdm
import numpy.random as npr

imagenet_loc = '/nobackup-slow/dataset/ILSVRC-2012'
imagenet500_loc = '/nobackup-slow/dataset/ImageNet500'

class_list = os.listdir(os.path.join(imagenet_loc, 'train'))
class_subset = npr.permutation(class_list)[0:500]

for split in ['train', 'val']:
    for line in tqdm(class_subset):
        cls = line.strip()
        shutil.copytree(os.path.join(imagenet_loc, split, cls), os.path.join(imagenet500_loc, split, cls), dirs_exist_ok=True)
