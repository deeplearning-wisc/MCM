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
    wordnet_ids = get_dogs_cls()
    labels = labels_from_wordnet_ids(wordnet_ids)
    labels_by_wordnet_ids = {wordnet_ids[i]: labels[i] for i in range(len(wordnet_ids))}
    image_filenames = [os.path.join(params.image_dir, option, cls_name, filename) for cls_name in wordnet_ids for filename in os.listdir(os.path.join(params.image_dir, option, cls_name))]
    captions = [f'This is a photo of {labels_by_wordnet_ids[id]}' for id in wordnet_ids for filename in os.listdir(os.path.join(params.image_dir, option, id))]
    targets = [i for i, id in enumerate(wordnet_ids) for filename in os.listdir(os.path.join(params.image_dir, option, id))]
    dataset = CLIPDataset_ViT(params, image_filenames, captions, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=(option == 'train'))
    return dataloader

def get_dogs_cls():
    cls_list_loc = './data/ImageNetDogs'
    with open(os.path.join(cls_list_loc, 'class_list.txt')) as f:
        wordnet_ids = [l.strip() for l in f.readlines()]
    return wordnet_ids
