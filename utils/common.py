import torch
import torch.nn.functional as F
import os
import numpy as np
import json
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_test_labels(args, loader = None):
    if args.in_dataset == "ImageNet":
        test_labels = obtain_ImageNet_classes()
    elif args.in_dataset == "ImageNet10":
        test_labels = obtain_ImageNet10_classes()
    elif args.in_dataset == "ImageNet20":
        test_labels = obtain_ImageNet20_classes()
    elif args.in_dataset == "ImageNet100":
        test_labels = obtain_ImageNet100_classes()
    elif args.in_dataset in ['bird200', 'car196', 'food101','pet37']:
        test_labels = loader.dataset.class_names_str
    return test_labels

def obtain_ImageNet_classes():
    loc = os.path.join('data', 'ImageNet')
    with open(os.path.join(loc, 'imagenet_class_clean.npy'), 'rb') as f:
        imagenet_cls = np.load(f)
    return imagenet_cls


def obtain_ImageNet10_classes():

    class_dict = {"warplane": "n04552348", "sports car": "n04285008",
                  'brambling bird': 'n01530575', "Siamese cat": 'n02123597',
                  'antelope': 'n02422699', 'swiss mountain dog': 'n02107574',
                  "bull frog": "n01641577", 'garbage truck': "n03417042",
                  "horse": "n02389026", "container ship": "n03095699"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[1])}
    return class_dict.keys()


def obtain_ImageNet20_classes():

    class_dict = {"n04147183": "sailboat", "n02951358": "canoe", "n02782093": "balloon", "n04389033": "tank", "n03773504": "missile",
                  "n02917067": "bullet train", "n02317335": "starfish", "n01632458": "spotted salamander", "n01630670": "common newt", "n01631663": "eft",
                  "n02391049": "zebra", "n01693334": "green lizard", "n01697457": "African crocodile", "n02120079": "Arctic fox", "n02114367": "timber wolf",
                  "n02132136": "brown bear", "n03785016": "moped", "n04310018": "steam locomotive", "n04266014": "space shuttle", "n04252077": "snowmobile"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[0])}
    return class_dict.values()

def obtain_ImageNet100_classes():
    loc=os.path.join('data', 'ImageNet100')
    # sort by values
    with open(os.path.join(loc, 'class_list.txt')) as f:
        class_set = [line.strip() for line in f.readlines()]

    class_name_set = []
    with open('data/ImageNet/imagenet_class_index.json') as file: 
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid, class_name in class_index_raw.values()}
        class_name_set = [class_index[c] for c in class_set]
    class_name_set = [x.replace('_', ' ') for x in class_name_set]

    return class_name_set

def get_num_cls(args):
    NUM_CLS_DICT = {
        'ImageNet10': 10,
        'ImageNet20': 20,
        'pet37': 37,
        'ImageNet100': 100, 
        'food101': 101, 
        'car196': 196,
        'bird200':200, 
        'ImageNet': 1000,
    }
    n_cls = NUM_CLS_DICT[args.in_dataset]
    return n_cls


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # values, indices = input.topk(k, dim=1, largest=True, sorted=True)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def read_file(file_path, root='corpus'):
    corpus = []
    with open(os.path.join(root, file_path)) as f:
        for line in f:
            corpus.append(line[:-1])
    return corpus


def calculate_cosine_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    return similarity


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

