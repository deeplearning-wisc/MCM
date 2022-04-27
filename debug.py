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


def obtain_ImageNet100_classes(loc):
    # sort by values
    with open(os.path.join(loc, 'train_100.txt')) as f:
        class_set = list({line.split('/')[1].strip() for line in f.readlines()})
        class_set.sort()
        print(class_set)
    class_name_set = []
    with open('data/ImageNet/imagenet_class_index.json') as file: 
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid, class_name in class_index_raw.values()}
        # class_index =  {k: v for k, v in sorted(class_index.items(), key=lambda item: item[0])}
        class_name_set = [class_index[c] for c in class_set]
        # class_name_set = class_index.values()
    class_name_set = [x.replace('_', ' ') for x in class_name_set]
    print(class_name_set)
    return list(class_name_set)

def plot_similarity(similarity, original_images,texts):
    count = len(texts)
    plt.figure(figsize=(40, 15))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=20)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower") # (left, right, bottom, top)
    for x in range(similarity.shape[1]): 
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=1)

    for side in ["left", "top", "right", "bottom"]:
      plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, similarity.shape[1] - 0.5])
    plt.ylim([count + 0.5, -2])
    # plt.ylim([-2,count + 0.5])
    # plt.title("Cosine similarity between text and image features ImageNet-100", size=20)
    plt.savefig('6.pdf',bbox_inches='tight',pad_inches = 0 )



random.seed(4)
dataset = datasets.ImageFolder(os.path.join('/nobackup/dataset_myf', 'ImageNet100', 'val'),  transform=preprocess)
                               
classwise_count = defaultdict(int)
indices = []
for i, label in enumerate(dataset.targets): 
    if classwise_count[label] < 1:
        if random.random() > 0.8:
            indices.append(i)
            classwise_count[label] += 1

sorted_idx = np.array(indices)[np.argsort(np.array(dataset.targets)[indices])]
print(np.array(dataset.targets)[sorted_idx])
subset = torch.utils.data.Subset(dataset, sorted_idx)   
loader = torch.utils.data.DataLoader(subset,
                batch_size=1, shuffle=False)

image_features = []
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(loader):
        print(labels, end = " ")
        image_feature = model.encode_image(images.cuda()).float()
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        image_features.append(image_feature.cpu())

original_images = []
MEAN =  torch.tensor(preprocess.transforms[-1].mean)
STD = torch.tensor(preprocess.transforms[-1].std)
with torch.no_grad():
    for id, (image, label) in enumerate(loader):
        print(f"{id}: {label}", end = " ")
        image = image[0] * STD[:, None, None] + MEAN[:, None, None]
        original_images.append(image.numpy().transpose(1, 2, 0))
        # if id < 20:
        #     plt.imsave(f'test_{id}.png', np.transpose(image.numpy(), (1,2,0)))

print("start examing test labels")
test_labels = obtain_ImageNet100_classes(loc = os.path.join('./data', 'ImageNet100'))

subset = True
pop_cls = 2
if pop_cls:
    original_images.pop(pop_cls)
    image_features.pop(pop_cls)
    test_labels.pop(pop_cls)

image_features = torch.cat(image_features)    
print(len(image_features))  

if subset:
    test_labels = test_labels[:3] + ['culture', 'philosophy', 'phenonemon']
  

text_descriptions = [f"This is a photo of a {label}" for label in test_labels] # 100 sentences in total
text_tokens = clip.tokenize(text_descriptions).cuda()
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float().cpu()
    text_features /= text_features.norm(dim=-1, keepdim=True) # cannot remove keepdim = True
similarity = text_features.numpy() @ image_features[:20].numpy().T

plot_similarity(similarity, original_images[:20],test_labels[:20])