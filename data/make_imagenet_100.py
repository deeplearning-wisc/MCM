import os
import shutil
from tqdm import tqdm
import json

# imagenet_loc = '/nobackup-slow/dataset/ILSVRC-2012'
# imagenet100_loc = '/nobackup-slow/dataset/ImageNet100'

imagenet_loc = '/nobackup/ImageNet'
imagenet100_loc = '/nobackup/dataset_myf/ImageNet40'

# for split in ['train', 'val']:
#     with open(f'./ImageNet100/{split}_100.txt') as file:
#         for line in tqdm(file.readlines()):
#             path = line.split(' ')[0]
#             os.makedirs(os.path.join(imagenet100_loc, os.path.dirname(path)), exist_ok=True)
#             shutil.copy(os.path.join(imagenet_loc, path), os.path.join(imagenet100_loc, path))

for split in ['train', 'val']:
    with open(f'./ImageNet100/class_list.txt') as file:
        for line in tqdm(file.readlines()):
            cls = line.strip()
            # os.makedirs(os.path.join(imagenet100_loc, split, cls), exist_ok=True)
            shutil.copytree(os.path.join(imagenet_loc, split, cls), os.path.join(imagenet100_loc, split, cls), dirs_exist_ok=True)

# class_set = set()
# with open('ImageNet100/val_100.txt') as file:
#     for line in file.readlines():
#         class_set.add(line.split(' ')[1].strip())

# class_name_set = []
# with open('imagenet_class_index.json') as file: 
#     class_index = json.load(file)
#     class_name_set = [class_index[c] for c in class_set]

# class_name_set = sorted(class_name_set, key=lambda item: item[1])
# class_name_set = [x[1] for x in class_name_set]
# print(class_name_set)