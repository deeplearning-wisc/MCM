import os
import torch
import numpy as np
from tqdm import tqdm
import clip
import torchvision
import matplotlib.pyplot as plt
from scipy import stats
from torchvision.transforms import transforms
import torch.nn.functional as F
from torchvision import datasets

class ImageFolderWithNames(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithNames, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        file_name = path.split("/")[-1]
        file_name_no_ext = file_name.split(".")[0]

        tuple_with_name = (original_tuple + (file_name_no_ext,))
        return tuple_with_name


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root = '/nobackup/dataset_myf/ImageNet_OOD_dataset'):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if args.score == 'clipcap_nouns':
        ImageFolder = datasets.ImageFolder
    elif args.score == 'ofa_nouns':
        ImageFolder = ImageFolderWithNames
    
    if out_dataset == 'iNaturalist':
        testsetout = ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365': # filtered places
        testsetout = ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)  
    elif out_dataset == 'placesbg': 
        testsetout = ImageFolder(root= os.path.join(root, 'placesbg'),transform=preprocess)  
    elif out_dataset == 'dtd':
        if args.server == 'galaxy-01':
            testsetout = ImageFolder(root=os.path.join(root, 'Textures'),
                                    transform=preprocess)
        elif args.server == 'galaxy-02':
            root = '/nobackup-slow/dataset'
        else:
            root = '/nobackup/dataset_myf/ood_datasets'
        testsetout = ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                        transform=preprocess)
    elif out_dataset == 'ImageNet10':
        testsetout = ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'train'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    elif out_dataset == 'ImageNet30':
        testsetout = ImageFolder(os.path.join(args.root_dir, 'ImageNet30', 'val'), transform=preprocess)
    elif out_dataset == 'ImageNet100':
        if args.server in ['inst-01', 'inst-04']:
            path = os.path.join('/nobackup','ImageNet')
        elif args.server in ['galaxy-01', 'galaxy-02']:
            path = os.path.join(root, 'ILSVRC-2012')
        id1 = 'test_imagenet100_10_seed_4'
        testsetout = ImageNetSubset(100, path, train=False, seed=args.seed, transform=preprocess, id=id1)
    # if len(testsetout) > 10000: 
    #     testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return testloaderOut


def set_val_loader(args, preprocess = None):
    # normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
    #                                       std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #for c-10
    # normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) #for c-100
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                        std=(0.26862954, 0.26130258, 0.27577711)) # for CLIP
        preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "CIFAR-10":
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(root, 'cifar10'), train=False,download=True, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "CIFAR-100":
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(root, 'cifar100'), train=False, download=True,transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset in  ["ImageNet10", "ImageNet20", "ImageNet30", "ImageNet100"]:
        data_dir = os.path.join(root, args.in_dataset, 'val')
        if args.score == 'clipcap_nouns':
            dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        elif args.score == 'ofa_nouns':
            dataset = ImageFolderWithNames(data_dir, transform = preprocess)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    return val_loader

def get_nouns_scores_clip(args, preprocess, net, image_loader, ID_labels, dataset_name, in_dist = True, filter = None, debug = False):
    '''
    used for nouns score. 1 - sum_{i \in K} p(\hat{y}=i|x)
    '''
    import pandas as pd
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    if  args.score == 'clipcap_nouns':
        captions_nouns_dir = 'clipcap_nouns'
        captions_nouns_path = os.path.join(captions_nouns_dir, f'{dataset_name}_clipcap_captions_and_nouns.csv')
    elif args.score == 'ofa_nouns':
        captions_nouns_dir = 'ofa_nouns'
        captions_nouns_path = os.path.join(captions_nouns_dir, f'{dataset_name}_OFA_captions.csv')
    df = pd.read_csv(f"{captions_nouns_path}", sep=',', converters={'Nouns': pd.eval})
    # text_dataset = TextDataset(df["Nouns"], df["Type"])
    # text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=args.batch_size, shuffle=False)
    bz = image_loader.batch_size
    with torch.no_grad():
        for i, (images, labels, image_ids) in enumerate(tqdm(image_loader)):
            image_id = image_ids[0]
            if  args.score == 'clipcap_nouns':
                generated_labels = list(df["Nouns"][i])[0]
            elif args.score == 'ofa_nouns':
                generated_labels = list(df["Nouns"][df["ImageID"]==image_id])[0] # [['bird', 'snow']] --> ['bird', 'snow']
            if debug:
                mean = torch.tensor(preprocess.transforms[-1].mean)
                std = torch.tensor(preprocess.transforms[-1].std)
                recovered_img = images[0]*std[:,None, None] + mean[:, None, None]
                plt.imsave(f'test_{i}.png', np.transpose(recovered_img.numpy(), (1,2,0)))

            if i >= 2000 and in_dist is False:
                break

            if filter == "str":
                generated_labels = [label for label in generated_labels if label not in ID_labels ]
            all_labels = ID_labels + generated_labels
            text_features = net.encode_text(torch.cat([clip.tokenize(f"a photo of a {c}") for c in all_labels]).cuda()).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            images = images.cuda()
            image_features = net.encode_image(images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            output = image_features @ text_features.T

            smax = to_np(F.softmax(output *100, dim=1))
           # if not in_dist:
                # print(np.around(smax,3))
                # print(1 -np.sum(smax[: args.n_cls], axis=1))
            score = 1 -np.sum(smax[0, : args.n_cls])
            # print(score)
            _score.append(score) 

        return np.array(_score)