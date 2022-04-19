import argparse
import math
import os
import torch
import numpy as np
from tqdm import tqdm
import logging
import clip
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.utils import shuffle
from models.linear import LinearClassifier
import torch.backends.cudnn as cudnn
# from transformers import ViTFeatureExtractor,  ViTModel,  CLIPModel

def set_model_vit():
    '''
    load Huggingface ViT
    '''
    model =  ViTModel.from_pretrained('google/vit-base-patch16-224').cuda()
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                        std=(0.5, 0.5, 0.5)) # for ViT
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    return model, val_preprocess

def set_model_clip(args):
    '''
    load Huggingface CLIP
    '''
    ckpt_mapping = {"ViT-B/16":"openai/clip-vit-base-patch16", 
                    "ViT-B/32":"openai/clip-vit-base-patch32",
                    "ViT-L/14":"openai/clip-vit-large-patch14"}
    args.ckpt = ckpt_mapping[args.CLIP_ckpt]
    model =  CLIPModel.from_pretrained(args.ckpt)
    if args.finetune_ckpt:
        model.load_state_dict(torch.load(args.finetune_ckpt, map_location=torch.device(args.gpu)))
    model = model.cuda()

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                        std=(0.229, 0.224, 0.225)) # for ViT

    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    return model, val_preprocess

def set_val_loader_vit(args,  preprocess, kwargs = {'num_workers': 4, 'pin_memory': True}):
    root = args.root_dir
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                        std=(0.5, 0.5, 0.5)) # for ViT
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    if args.in_dataset == "ImageNet":
        if args.server in ['inst-01', 'inst-04']:
            path = os.path.join('/nobackup','ImageNet')
        elif args.server in ['galaxy-01', 'galaxy-02']:
            path = os.path.join(root, 'ILSVRC-2012')
        val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(os.path.join(path, "val"), transform=preprocess),
                batch_size=args.batch_size, shuffle=False,  **kwargs)
    return val_loader

def set_train_loader_vit(args, preprocess, subset = False, kwargs = {'num_workers': 4, 'pin_memory': True}):
    root = args.root_dir
    if preprocess is None: #training mode
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                        std=(0.5, 0.5, 0.5)) # for ViT
        preprocess = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    if args.in_dataset == "ImageNet":
        if args.server in ['inst-01', 'inst-04']:
            path = os.path.join('/nobackup','ImageNet')
        elif args.server in ['galaxy-01', 'galaxy-02']:
            path = os.path.join(root, 'ILSVRC-2012')
        dataset = datasets.ImageFolder(os.path.join(path, "train"), transform=preprocess)
        if subset: 
            from collections import defaultdict
            classwise_count = defaultdict(int)
            indices = []
            for i, label in enumerate(dataset.targets): 
                if classwise_count[label] < args.max_count:
                    indices.append(i)
                    classwise_count[label] += 1
            dataset = torch.utils.data.Subset(dataset, indices)            
        train_loader = torch.utils.data.DataLoader(dataset,
                batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader