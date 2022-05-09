import os
import shutil
from tqdm import tqdm
import json

#inst-01
imagenet_loc = '/nobackup/ImageNet'
imagenet30_loc = '/nobackup/dataset_myf/ImageNet30'


for split in ['train', 'val']:
    with open(f'/u/a/l/alvinming/ood/CLIP_OOD/data/ImageNet30/class_list.txt') as file:
        for line in tqdm(file.readlines()):
            cls = line.strip()
            # os.makedirs(os.path.join(imagenet100_loc, split, cls), exist_ok=True)
            shutil.copytree(os.path.join(imagenet_loc, split, cls), os.path.join(imagenet30_loc, split, cls), dirs_exist_ok=True)