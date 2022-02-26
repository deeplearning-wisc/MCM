from tqdm import tqdm
import torch
import clip
import os
import skimage
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10

def obtain_cifar_classes(root):
    cifar100 = CIFAR100( os.path.join(root, 'cifar10'), download=True, train=False)
    cifar10 = CIFAR10( os.path.join(root, 'cifar10'), download=True, train=False)
    return cifar10.classes, cifar100.classes

def obtain_ImageNet_classes(loc, cleaned = False):
    if not cleaned:
        import json
        idx2label = []
        cls2label = {}
        with open(loc, "r") as read_file:
            class_idx = json.load(read_file)
            # idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        return cls2label.values()
    else:
        with open('loc.npy', 'rb') as f:
            imagenet_cls = np.load(f)
        return imagenet_cls



def get_image_dataloader(image_dataset_name, preprocess, train = False):
    data_dir = os.path.join('data',image_dataset_name)
    if image_dataset_name.startswith('cifar'):
      if image_dataset_name == 'cifar100':
          image_dataset = CIFAR100(data_dir, transform=preprocess, download=True, train=train)
      elif image_dataset_name == 'cifar10':
          image_dataset = CIFAR10(data_dir, transform=preprocess, download=True, train=train)
    dataloader = DataLoader(image_dataset, batch_size=200, shuffle=train, drop_last=True, num_workers=4)
    return dataloader

def evaluate_classification(dataloader, test_labels, model, preprocess, device):
    tqdm_object = tqdm(dataloader, total=len(dataloader))
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_labels]).to(device)

    top5, top1 = AverageMeter(), AverageMeter()
    with torch.no_grad():
      for (images, labels) in tqdm_object:
          labels = labels.long().to(device)
          images = images.to(device)
          image_features = model.encode_image(images)
          text_features = model.encode_text(text_inputs)
          image_features /= image_features.norm(dim=-1, keepdim=True)
          text_features /= text_features.norm(dim=-1, keepdim=True)   
          # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
          logits = image_features @ text_features.T
          # _, pred = logits.topk(1, 1, True, True)
          # pred = pred.t()
          precs = accuracy(logits, labels, topk=(1, 5))
          top1.update(precs[0].item(), images.size(0))
          top5.update(precs[1].item(), images.size(0))

    print(f"Classification Top 1 acc: {top1.avg}; Top 5 acc: {top5.avg}")

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
            
def read_file(file_path, root = 'corpus'):
    corpus = []
    with open(os.path.join(root, file_path)) as f:
        for line in f:
            corpus.append(line[:-1])
    return corpus

def examine_clip_model(model):
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
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
        recovered_img = image*std[:,None, None] + mean[:, None, None]
        plt.imshow(np.transpose(recovered_img.numpy(), (1,2,0)))
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
        name = os.path.splitext(filename)[0] #around 30 imgs in total
        if name not in descriptions: 
            continue
        image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
    
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
    plt.tight_layout()
    plt.savefig('cosine.png')

def get_features(model, dataloader, device):
    '''
    extract all image and text features from the dataset
    V1.1: only supports CPU
    '''
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def create_subset_imagenet(class_ids, src_root = '/nobackup/ImageNet/val', dst_root = 'data'):
    import shutil
    dst_dir = os.path.join(dst_root, 'ImageNetS')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for class_id in os.listdir(src_root):
        if class_id in class_ids:
            src_path = os.path.join(src_root,class_id)
            dst_path = os.path.join(dst_dir,class_id)
            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path)
                

    
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
    class_ids = ['n04552348','n04285008','n01530575','n02123597','n02422699',
                'n02107574', 'n01641577','n01728572','n03095699','n03417042']
    create_subset_imagenet(class_ids)