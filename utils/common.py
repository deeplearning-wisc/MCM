from tqdm import tqdm
import torch
import torch.nn.functional as F
import clip
import os
import skimage
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
import json
import random


def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def obtain_cifar_classes(root, which_cifar='CIFAR-10'):
    if which_cifar == 'CIFAR-100':
        cifar = CIFAR100(os.path.join(root, 'cifar10'),
                         download=True, train=False)
    else:
        cifar = CIFAR10(os.path.join(root, 'cifar10'),
                        download=True, train=False)
    return cifar.classes


def obtain_ImageNet_classes(loc, option='clean'):
    if option == 'original':
        # idx2label = []
        cls2label = {}
        with open(loc, "r") as read_file:
            class_idx = json.load(read_file)
            # idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1]
                         for k in range(len(class_idx))}
        return cls2label.values()
    elif option == 'clean':
        with open(os.path.join(loc, 'imagenet_class_clean.npy'), 'rb') as f:
            imagenet_cls = np.load(f)
    elif option == 'simple':
        with open(os.path.join(loc, 'imagenet-simple-labels.json')) as f:
            imagenet_cls = json.load(f)
    return imagenet_cls


def obtain_ImageNet10_classes(loc=None):

    class_dict = {"warplane": "n04552348", "sports car": "n04285008",
                  'brambling bird': 'n01530575', "Siamese cat": 'n02123597',
                  'antelope': 'n02422699', 'swiss mountain dog': 'n02107574',
                  "bull frog": "n01641577", 'garbage truck': "n03417042",
                  "horse": "n02389026", "container ship": "n03095699"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[1])}
    return class_dict.keys()


def obtain_ImageNet20_classes(loc=None):

    class_dict = {"n04147183": "sailboat", "n02951358": "canoe", "n02782093": "balloon", "n04389033": "tank", "n03773504": "missile",
                  "n02917067": "bullet train", "n02317335": "starfish", "n01632458": "spotted salamander", "n01630670": "common newt", "n01631663": "zebra",
                  "n02391049": "frilled lizard", "n01693334": "green lizard", "n01697457": "African crocodile", "n02120079": "Arctic fox", "n02114367": "timber wolf",
                  "n02132136": "brown bear", "n03785016": "moped", "n04310018": "steam locomotive", "n04266014": "space shuttle", "n04252077": "snowmobile"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[0])}
    return class_dict.values()


def obtain_ImageNet30_classes():

    all_labels = ['stingray', 'american_alligator', 'dragonfly', 'airliner',
                  'ambulance', 'banjo', 'barn', 'bikini', 'rotary_dial_telephone', 'digital_clock',
                  'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hourglass', 'manhole_cover',
                  'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'schooner', 'snowmobile',
                  'soccer_ball', 'tank', 'toaster', 'hotdog', 'strawberry', 'volcano', 'acorn']
    return all_labels


def obtain_ImageNet100_classes(loc):
    # sort by values
    with open(os.path.join(loc, 'train_100.txt')) as f:
        class_set = list({line.split('/')[1].strip()
                         for line in f.readlines()})
        class_set.sort()

    class_name_set = []
    with open('data/ImageNet/imagenet_class_index.json') as file:
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid,
                       class_name in class_index_raw.values()}
        # class_index =  {k: v for k, v in sorted(class_index.items(), key=lambda item: item[0])}
        class_name_set = [class_index[c] for c in class_set]
        # class_name_set = class_index.values()

    class_name_set = [x.replace('_', ' ') for x in class_name_set]
    return class_name_set


def get_num_cls(args):
    NUM_CLS_DICT = {
        'CIFAR-10': 10, 'ImageNet10': 10,
        'ImageNet20': 20, 'ImageNet30': 30,
        'pet37': 37,
        'ImageNet100': 100, 'CIFAR-100': 100,
        'food101': 101, 'flower102': 102,
        'car196': 196, 'bird200': 200,
        'ImageNet': 1000,
    }
    if args.in_dataset in ['ImageNet-subset', 'ImageNet-dogs']:
        n_cls = args.num_imagenet_cls
    else:
        n_cls = NUM_CLS_DICT[args.in_dataset]
    return n_cls


def get_image_dataloader(image_dataset_name, preprocess, train=False):
    data_dir = os.path.join('data', image_dataset_name)
    if image_dataset_name.startswith('CIFAR'):
        if image_dataset_name == 'CIFAR-100':
            image_dataset = CIFAR100(
                data_dir, transform=preprocess, download=True, train=train)
        elif image_dataset_name == 'CIFAR-10':
            image_dataset = CIFAR10(
                data_dir, transform=preprocess, download=True, train=train)
    dataloader = DataLoader(image_dataset, batch_size=200,
                            shuffle=train, drop_last=True, num_workers=4)
    return dataloader


def get_features(args, model, dataloader, to_np=True, dataset='none'):
    '''
    extract image features from the dataset
    '''
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            if args.model == 'CLIP':
                features = model.encode_image(images.to(args.device))
            elif args.model == 'vit':
                features = model(pixel_values=images.to(
                    args.device)).last_hidden_state[:, 0, :]
            if args.normalize:
                features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
            all_labels.append(labels)
    if to_np:
        all_features = torch.cat(all_features).numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        save_dir = os.path.join(
            args.template_dir, 'all_feat', f'{args.name}_{args.K}')
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir,  f'all_feat_{dataset}_{args.max_count}_{args.normalize}.npy'), 'wb') as f:
            np.save(f, all_features)
            np.save(f, all_labels)

        return all_features, all_labels
    else:
        return torch.cat(all_features), torch.cat(all_labels)


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


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def read_file(file_path, root='corpus'):
    corpus = []
    with open(os.path.join(root, file_path)) as f:
        for line in f:
            corpus.append(line[:-1])
    return corpus


def examine_clip_model(model):
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size
    print("Model parameters:",
          f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Available pretrained models: ", clip.available_models())


def display_processed_image(images, preprocess, file_names, texts):
    '''
        display images given Torch tensors
    '''
    plt.figure(figsize=(16, 5))

    for i, image in enumerate(images):
        plt.subplot(2, 4, i + 1)
        mean = torch.tensor(preprocess.transforms[-1].mean)
        std = torch.tensor(preprocess.transforms[-1].std)
        recovered_img = image*std[:, None, None] + mean[:, None, None]
        plt.imshow(np.transpose(recovered_img.numpy(), (1, 2, 0)))
        plt.title(f"i={i}: {file_names[i]}\n{texts[i]}")
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()


def read_skimage(preprocess, descriptions):
    '''
        read images from skimage pkg with file names in the keys of the description dict
    '''
    original_images = []
    images = []
    texts = []
    file_names = []
    plt.figure(figsize=(16, 5))
    print(os.listdir(skimage.data_dir))
    for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]  # around 30 imgs in total
        if name not in descriptions:
            continue
        image = Image.open(os.path.join(
            skimage.data_dir, filename)).convert("RGB")

        plt.subplot(2, 4, len(images) + 1)
        plt.imshow(image)
        plt.title(f"{filename}\n{descriptions[name]}")
        plt.xticks([])
        plt.yticks([])

        original_images.append(image)
        images.append(preprocess(image))
        texts.append(descriptions[name])
        file_names.append(filename)

    plt.tight_layout()
    plt.savefig('skimgs.png')
    return original_images, images, texts, file_names


def calculate_cosine_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    return similarity


def plot_similarity(similarity, count, original_images, texts):
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6),
                   origin="lower")  # (left, right, bottom, top)
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}",
                     ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])
    # plt.ylim([-2,count + 0.5])
    plt.title("Cosine similarity between text and image features", size=20)
    plt.tight_layout()
    plt.savefig('cosine.png')


def create_subset_imagenet(class_ids, src_root='/nobackup/ImageNet/val', dst_root='data'):
    import shutil
    dst_dir = os.path.join(dst_root, 'ImageNetS')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for class_id in os.listdir(src_root):
        if class_id in class_ids:
            src_path = os.path.join(src_root, class_id)
            dst_path = os.path.join(dst_dir, class_id)
            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path)


def examine_scores_clip(net, loader, test_labels, softmax=True):
    def to_np(x): return x.data.cpu().numpy()
    def concat(x): return np.concatenate(x, axis=0)
    _right_score = []
    _wrong_score = []

    _lables = []
    _smax_scores = []

    tqdm_object = tqdm(loader, total=len(loader))
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in test_labels]).cuda()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            labels = labels.long().cuda()
            images = images.cuda()
            image_features = net.encode_image(images)
            text_features = net.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            output = image_features @ text_features.T
            if softmax:
                smax = to_np(F.softmax(output, dim=1))
            else:
                smax = to_np(output)

            _smax_scores.append(smax)
            labels.append(labels)

            preds = np.argmax(smax, axis=1)
            targets = labels.cpu().numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
    right_score, wrong_score = concat(_right_score), concat(_wrong_score)
    num_right = len(right_score)
    num_wrong = len(wrong_score)
    print('Error Rate {:.2f}'.format(
        100 * num_wrong / (num_wrong + num_right)))
    return concat(_smax_scores), concat(_lables)


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


if __name__ == '__main__':
    class_ids = ['n04552348', 'n04285008', 'n01530575', 'n02123597', 'n02422699',
                 'n02107574', 'n01641577', 'n01728572', 'n03095699', 'n03417042']
    create_subset_imagenet(class_ids)
