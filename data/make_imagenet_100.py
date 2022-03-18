import os
import shutil
from tqdm import tqdm

imagenet_loc = '/nobackup-slow/dataset/ILSVRC-2012'
imagenet100_loc = '/nobackup-slow/dataset/ImageNet100'

for split in ['train', 'val']:
    with open(f'./ImageNet100/{split}_100.txt') as file:
        for line in tqdm(file.readlines()):
            path = line.split(' ')[0]
            os.makedirs(os.path.join(imagenet100_loc, os.path.dirname(path)), exist_ok=True)
            shutil.copy(os.path.join(imagenet_loc, path), os.path.join(imagenet100_loc, path))
