import os
import shutil
from tqdm import tqdm
import json

#inst-01
imagenet_loc = '/nobackup/ImageNet'
imagenet20_loc = '/nobackup/dataset_myf/ImageNet20'

class_dict =   {"n04147183": "sailboat", "n02951358": "canoe" , "n02782093": "balloon", "n04389033": "tank", "n03773504": "missile",
    "n02917067": "bullet train", "n02317335": "starfish", "n01632458":"spotted salamander", "n01630670":"common newt", "n01631663": "zebra",
    "n02391049": "frilled lizard", "n01693334":"green lizard", "n01697457": "African crocodile", "n02120079": "Arctic fox", "n02114367": "timber wolf",  
    "n02132136": "brown bear", "n03785016": "moped", "n04310018": "steam locomotive", "n04266014": "space shuttle", "n04252077": "snowmobile"}

for split in ['train', 'val']:
    for cls in class_dict.keys():
        shutil.copytree(os.path.join(imagenet_loc, split, cls), os.path.join(imagenet20_loc, split, cls), dirs_exist_ok=True)