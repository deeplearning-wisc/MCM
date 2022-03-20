import argparse
import math
import os
import torch
import numpy as np
import skimage
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

import clip
from torch.utils.data import DataLoader
# from torchvision import transforms
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from utils import *

def zero_shot_evaluation_CLIP(image_dataset_name, test_labels, ckpt = 'ViT-B/16'):
    # Load the model
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(ckpt, device)  
    #zero shot evaluation
    dataloader = get_image_dataloader(image_dataset_name, preprocess, train = False)
    evaluate_classification(dataloader, test_labels, model, preprocess, device)

def linear_probe_evaluation_CLIP_cpu(image_dataset_name, ckpt = 'ViT-B/16'):
    '''
    train a logistic regression on top of frozen image features extracted by CLIP image encoder (ViT or ResNet)
    V1.1: CPU version with sklearn
    '''
    from sklearn.linear_model import LogisticRegression
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(ckpt, device)

    train_loader = get_image_dataloader(image_dataset_name, preprocess, train = True)
    test_loader = get_image_dataloader(image_dataset_name, preprocess, train = False)
    train_features, train_labels = get_features(model, train_loader, device)
    test_features, test_labels = get_features(model, test_loader, device)
    # Note: C is the inverse of regularization strength; must be a positive float. 
    # Like in support vector machines, smaller values specify stronger regularization.
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")

def play_with_skimage(ckpt = "ViT-B/16"):
    '''
        A simple test of CLIP with 8 images from Skimage and their descriptions
    '''
    descriptions = {
        "page": "a page of text about segmentation",
        "chelsea": "a facial photo of a tabby cat",
        "astronaut": "a portrait of an astronaut with the American flag",
        "rocket": "a rocket standing on a launchpad",
        "motorcycle_right": "a red motorcycle standing in a garage",
        "camera": "a person looking at a camera on a tripod",
        "horse": "a black-and-white silhouette of a horse", 
        "coffee": "a cup of coffee on a saucer"
    }

    model, preprocess = clip.load(ckpt)
    # e.g. get std of Normalization in preprocess: preprocess.transforms[-1].std
    model.cuda().eval()

    original_images, images, texts, file_names = read_skimage(preprocess, descriptions)

    image_input = torch.tensor(np.stack(images)).cuda() # convert a list of images to tensors
    text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda() 
    # print(image_input.shape, text_tokens.shape) # torch.Size([8, 3, 224, 224]) torch.Size([8, 77])
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
    # print(image_features.shape, text_features.shape) # torch.Size([8, 512]) torch.Size([8, 512])
    similarity = calculate_cosine_similarity(image_features, text_features)
    plot_similarity(similarity, len(descriptions), original_images, texts)


def parse_option():
    parser = argparse.ArgumentParser('argument for playing with CLIP')
    parser.add_argument('--img_dataset', type=str, default='CIFAR-10',
                        choices=['CIFAR-10', 'CIFAR-100'], help='img dataset')
    parser.add_argument('--ckpt', type=str, default='ViT-L/14',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='which pretrained img encoder to use')
    
    # parser.add_argument('--feat_dim', type=int, default=784, help='feat dim')
    # parser.add_argument('--learning_rate', type=float, default=0.1,
    #                     help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0,
    #                     help='weight decay')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='momentum')
    # parser.add_argument('--batch_size', type=int, default=512,
    #                     help='batch_size')
    # parser.add_argument('--cosine', action='store_true',
    #                     help='using cosine annealing')
    # parser.add_argument('--warm', action='store_false',
    #                     help='warm-up for large batch training')
    parser.add_argument('--normalize', action='store_true',
                         help='whether the feautures are normalized')
    parser.add_argument('--name', type=str, default='test',
                        help='name of the run')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_option()
    cifar_cls = obtain_cifar_classes(root = '/nobackup/dataset_myf', which_cifar=args.img_dataset)
    if args.img_dataset == 'CIFAR-10':
        args.n_cls = 10
    elif args.img_dataset == 'CIFAR-100':
        args.n_cls = 100

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_{}'.\
        format(args.img_dataset, args.ckpt, args.learning_rate, args.weight_decay,
               args.batch_size, args.name)
    
    if args.cosine:
        args.model_name = '{}_cosine'.format(args.model_name)
    # warm-up for large-batch training
    if args.warm:
        args.model_name = '{}_warm'.format(args.model_name)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    if args.normalize: # do we need to normalize the feature 
        args.model_name = '{}_normalize'.format(args.model_name)
    # indices = np.random.choice(len(c100_cls), size=10, replace=False)
    # import random
    # len = 100
    #sample = random.sample(c100_cls, len)
    #test_labels =  c10_cls + sample

    # corpus = read_file('noun_en_test.txt')
    # test_labels = cifar_cls + corpus
    test_labels = cifar_cls
    zero_shot_evaluation_CLIP(args.img_dataset, test_labels, args.ckpt)
    # play_with_skimage()