from utils.coco_dataset import CLIPDataset_ViT, build_caption_dataframe
import torch
import os
import json

def labels_from_wordnet_ids(wordnet_ids):
    class_name_set = []
    with open('data/ImageNet/imagenet_class_index.json') as file: 
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid, class_name in class_index_raw.values()}
        class_name_set = [class_index[c] for c in wordnet_ids]
    class_name_set = [x.replace('_', ' ') for x in class_name_set]
    return class_name_set

def build_dogs_loader(params, option = 'train'):
    cls_list_loc = './data/ImageNetDogs'
    with open(os.path.join(cls_list_loc, 'class_list.txt')) as f:
        wordnet_ids = [l.strip() for l in f.readlines()]
    image_filenames = [os.path.join(params.image_dir, option, cls_name, filename) for cls_name in wordnet_ids for filename in os.listdir(os.path.join(params.image_dir, option, cls_name))]
    labels = labels_from_wordnet_ids(wordnet_ids)
    captions = [f'This is a photo of {l}' for l in labels]

    dataset = CLIPDataset_ViT(params, image_filenames, captions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=(option == 'train'))
    return dataloader
