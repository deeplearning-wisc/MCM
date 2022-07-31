
from torchvision.datasets import CIFAR10
from collections import defaultdict
from torchvision import datasets
import random

import numpy as np
import torch
import clip
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch

print("Torch version:", torch.__version__)
gpu = 7

torch.cuda.set_device(gpu)
model, preprocess = clip.load("ViT-B/16", gpu)
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

def obtain_ImageNet10_classes(loc = None):
    # class_dict = {'plane': 'n04552348', 'car': 'n04285008', 'bird': 'n01530575', 'cat':'n02123597', 
    #     'antelope' : 'n02422699', 'dog':'n02107574', 'frog':'n01641577',  'snake':'n01728572', 
    #     'ship':'n03095699', 'truck':'n03417042'}

    class_dict =   {"warplane": "n04552348", "sports car":"n04285008", 
        'brambling bird':'n01530575', "Siamese cat": 'n02123597', 
        'antelope': 'n02422699', 'Swiss mountain dog':'n02107574',
         "bull frog":"n01641577", 'garbage truck':"n03417042",
         "horse" :"n02389026", "container ship": "n03095699"}
    # sort by values
    class_dict =  {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1])}
    return class_dict.keys()

softmax = False
filename = 'tree.png'
texts = ['bird', 'frog', 'dog', 'cat', 'horse', 'antelope']
# texts = ['landbird', 'waterbird']
images = []
original_images = []
image = Image.open(filename).convert("RGB")
images.append(preprocess(image))
original_images.append(image)
image_input = torch.tensor(np.stack(images)).cuda() # get a batch of imgs
text_tokens = clip.tokenize([f"This is {desc}" for desc in texts]).cuda() 
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()
    
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu() @ image_features.cpu().T 
if softmax: 
    similarity = (100* similarity).softmax(dim = 0).numpy()
else:
    similarity = similarity.numpy()
count = len(texts)

def plot_similarity(count, original_images, texts):
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower") # (left, right, bottom, top)
    for x in range(similarity.shape[1]): 
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
      plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])
    # plt.ylim([-2,count + 0.5])
    plt.title("Cosine similarity between text and image features", size=20)
    plt.savefig(f'single.pdf',bbox_inches='tight',pad_inches = 0 )

plot_similarity(count, original_images,texts)