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

from utils import (AverageMeter, obtain_cifar_classes,  evaluate_classification, 
                    get_image_dataloader, read_file, read_skimage, calculate_cosine_similarity, 
                    plot_similarity, get_features)

def zero_shot_evaluation_CLIP(image_dataset_name, test_labels, ckpt = 'ViT-B/16'):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(ckpt, device)  
    #zero shot evaluation
    dataloader = get_image_dataloader(image_dataset_name, preprocess, train = False)
    evaluate_classification(dataloader, test_labels, model, preprocess, device)

def linear_probe_evaluation_CLIP(image_dataset_name, ckpt = 'ViT-B/16'):
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


if __name__ == '__main__':
    print("Torch version:", torch.__version__)
    image_dataset_name = 'cifar10'
    ckpt = 'ViT-B/16'
    c10_cls, c100_cls = obtain_cifar_classes()
    # indices = np.random.choice(len(c100_cls), size=10, replace=False)
    # import random
    # len = 100
    #sample = random.sample(c100_cls, len)
    #test_labels =  c10_cls + sample

    corpus = read_file('noun_en_test.txt')
    test_labels = c10_cls + corpus
    zero_shot_evaluation_CLIP(image_dataset_name, test_labels, ckpt)
    # linear_probe_evaluation_CLIP(image_dataset_name)
    # play_with_skimage()