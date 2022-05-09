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

from torchvision import transforms, datasets
from transformers import CLIPModel


from collections import OrderedDict
import torch

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

print("Torch version:", torch.__version__)
gpu = 7
torch.cuda.set_device(gpu)

def plot_similarity(original_images, texts, similarity, text = False, color_bar = False):
    count = len(texts)
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # fig.colorbar(neg, ax=ax2, location='right', anchor=(0, 0.3), shrink=0.7)
    # specify levels from vmim to vmax
    # v = np.linspace(0, 0.4, 9, endpoint=True)
    if color_bar:
        cbar = plt.colorbar(shrink=0.8)
        tick_font_size = 20
        cbar.ax.tick_params(labelsize=tick_font_size)
    plt.yticks(range(count), texts, fontsize=40)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower") # (left, right, bottom, top)
    if text:
        for x in range(similarity.shape[1]): 
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=5)

    for side in ["left", "top", "right", "bottom"]:
      plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])
    # plt.ylim([-2,count + 0.5])
    # plt.title("Cosine similarity between text and image features ImageNet-10", size=12)
    plt.savefig('test_ood.pdf', bbox_inches = 'tight',pad_inches = 0)


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


model, preprocess = clip.load("ViT-B/16", gpu)
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

ID = False
# if ID:
#     random.seed(16)
#     dataset = datasets.ImageFolder(os.path.join('/nobackup/dataset_myf', 'ImageNet10', 'val'),  transform=preprocess)
#     classwise_count = defaultdict(int)
#     indices = []
#     for i, label in enumerate(dataset.targets): 
#         if classwise_count[label] < 1:
#             if random.random() > 0.9:
#                 indices.append(i)
#                 classwise_count[label] += 1

#     sorted_idx = np.array(indices)[np.argsort(np.array(dataset.targets)[indices])]
#     subset = torch.utils.data.Subset(dataset, sorted_idx)  

if ID:
    subset = datasets.ImageFolder(os.path.join('/nobackup/dataset_myf', 'custom_id', 'val'),  transform=preprocess)
else:
    random.seed(7)
    #dataset = datasets.ImageFolder(os.path.join('/nobackup/dataset_myf', 'ImageNet20', 'val'),  transform=preprocess)
    dataset = datasets.ImageFolder(os.path.join( '/nobackup/dataset_myf/ImageNet_OOD_dataset', 'custom'),  transform=preprocess)
    indices = np.random.choice(np.arange(len(dataset)), 8, replace = False )
    subset = torch.utils.data.Subset(dataset, indices)  

loader = torch.utils.data.DataLoader(subset,
                    batch_size=1, shuffle=False)

    
image_features = []
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(loader):
        # print(images.shape)
        image_feature = model.encode_image(images.cuda()).float()
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        image_features.append(image_feature.cpu())

original_images = []

MEAN =  torch.tensor(preprocess.transforms[-1].mean)
STD = torch.tensor(preprocess.transforms[-1].std)

for image, label in loader:
    image = image[0] * STD[:, None, None] + MEAN[:, None, None]
    original_images.append(image.numpy().transpose(1, 2, 0))
    

test_labels = obtain_ImageNet10_classes()
print(test_labels)
image_features = torch.cat(image_features)

text_descriptions = [f"This is a photo of a {label}" for label in test_labels] # 100 sentences in total
text_tokens = clip.tokenize(text_descriptions).cuda()
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float().cpu()
    text_features /= text_features.norm(dim=-1, keepdim=True) # cannot remove keepdim = True
text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

similarity = text_features[:6].numpy() @ image_features[:6].numpy().T
test_labels = ['bird', 'frog', 'dog', 'cat', 'horse', 'antelope', 'container ship', 'garbage truck']
plot_similarity( original_images[:6],list(test_labels)[:6], similarity)
