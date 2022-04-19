import os
import torch
import numpy as np
from tqdm import tqdm
import clip
import torchvision
import sklearn.metrics as sk
from data.imagenet_subset import ImageNetDogs
from utils.common import get_features, get_fingerprint
from utils.plot_util import plot_distribution
import utils.svhn_loader as svhn
from torchvision.transforms import transforms
import torch.nn.functional as F
import faiss
import scipy
import matplotlib.pyplot as plt
from scipy import stats
from utils.train_eval_util import set_train_loader
from utils.imagenet_templates import openai_imagenet_template, openai_imagenet_template_subset

from transformers import CLIPTokenizer, CLIPModel

import umap

def set_ood_loader(args, out_dataset, preprocess, root = '/nobackup/dataset_myf'):
    '''
    set OOD loader for CIFAR scale datasets
    '''
    # normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
    #                                       std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #for c-10
    # normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) #for c-100
    # normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) # for CLIP
    if out_dataset == 'SVHN':

        testsetout = svhn.SVHN(os.path.join(root, 'ood_datasets', 'svhn'), split='test',
                                transform=preprocess, download=False)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'ood_datasets', 'dtd', 'images'),
                                    transform=preprocess)
    elif out_dataset == 'places365': # original places dataset, much larger size than 10,000
        # root_tmp= "/nobackup-slow/dataset/places365_test/test_subset" #galaxy
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'places365'),
            transform=preprocess)
    elif out_dataset == 'CIFAR-100':
        testsetout = torchvision.datasets.CIFAR100(root=os.path.join(root, 'cifar100'), train=False, download=True, transform=preprocess)
    elif out_dataset == 'CIFAR-10':
        testsetout = torchvision.datasets.CIFAR10(root=os.path.join(root, 'cifar10'), train=False, download=True, transform=preprocess)
    else:
        testsetout = torchvision.datasets.ImageFolder(os.path.join(root, "ood_datasets", f"{out_dataset}"),
                                    transform=preprocess)
    
    if len(testsetout) > 10000: 
        testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=True, num_workers=0)
    return testloaderOut

def set_ood_loader_ImageNet(args, out_dataset, preprocess, root = '/nobackup/dataset_myf_ImageNet_OOD'):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365': # filtered places
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)  
    elif out_dataset == 'placesbg': 
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'placesbg'),transform=preprocess)  
    elif out_dataset == 'dtd':
        if args.server == 'galaxy-01':
            testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Textures'),
                                    transform=preprocess)
        else:
            root = '/nobackup/dataset_myf'
            testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'ood_datasets', 'dtd', 'images'),
                                        transform=preprocess)
    # if len(testsetout) > 10000: 
    #     testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return testloaderOut

def set_ood_loader_ImageNet_dogs(args, preprocess):
    if args.server in ['inst-01', 'inst-04']:
        path = os.path.join('/nobackup','ImageNet')
    elif args.server in ['galaxy-01', 'galaxy-02']:
        path = os.path.join(args.root_dir, 'ILSVRC-2012')
    kwargs = {'num_workers': 4, 'pin_memory': True}
    dataset = ImageNetDogs(args.num_imagenet_cls, path, train=False, seed=args.seed, transform=preprocess, id=args.name, save=False, in_dist=False)
    ood_loader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs)
    return ood_loader

def print_measures(log, auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    if log == None: 
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
        print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
        print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    else:
        log.debug('\t\t\t\t' + method_name)
        log.debug('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
        log.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def get_ood_scores(args, net, loader, in_dist=False):
    '''
    useless for now. just for reference (calculate OOD score based on CNN backbones)
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    _right_score = []
    _wrong_score = []
    # embeddings = []
    # targets_embed = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= len(loader.dataset)  // args.batch_size and in_dist is False:
                break
            data = data.cuda()
            embed= net.encoder(data)
            # output,embed = net(data)
            output = net.fc(embed)
            smax = to_np(F.softmax(output, dim=1))
            # embeddings.append(embed.cpu().numpy())
            # targets_embed.append(target)

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if args.score == 'energy':
                    _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
                else: # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:len(loader.dataset)].copy()


def input_preprocessing(args, net, images, text_features = None, classifier = None):
    criterion = torch.nn.CrossEntropyLoss()
    if args.model == 'vit-Linear':
        image_features = net(pixel_values = images.float()).last_hidden_state
        image_features = image_features[:, 0, :]
    elif args.model == 'CLIP-Linear':
        image_features = net.encode_image(images).float()
    if classifier:
        outputs = classifier(image_features) / args.T
    else: 
        image_features = image_features/ image_features.norm(dim=-1, keepdim=True) 
        outputs = image_features @ text_features.T / args.T
    pseudo_labels = torch.argmax(outputs.detach(), dim=1)
    loss = criterion(outputs, pseudo_labels) # loss is NEGATIVE log likelihood
    loss.backward()

    sign_grad =  torch.ge(images.grad.data, 0) # sign of grad 0 (False) or 1 (True)
    sign_grad = (sign_grad.float() - 0.5) * 2  # convert to -1 or 1

    std=(0.26862954, 0.26130258, 0.27577711) # for CLIP model
    for i in range(3):
        sign_grad[:,i] = sign_grad[:,i]/std[i]

    processed_inputs = images.data  - args.noiseMagnitude * sign_grad # because of nll, here sign_grad is actually: -sign of gradient
    return processed_inputs

def get_ood_scores_clip_odin(args, net, loader, test_labels, classifier = None, in_dist=False):
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    _right_score = []
    _wrong_score = []
    text_features = None
    if classifier is None:
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_labels]).cuda()
            text_features = net.encode_text(text_inputs).float()
            text_features /= text_features.norm(dim=-1, keepdim=True) 

    tqdm_object = tqdm(loader, total=len(loader))
    for batch_idx, (images, labels) in enumerate(tqdm_object):
        if batch_idx >= len(loader.dataset)  // args.batch_size and in_dist is False:
            break
        # bz = images.size(0)
        labels = labels.long().cuda()
        images = images.cuda()

        images.requires_grad = True
        images = input_preprocessing(args, net, images, text_features, classifier)

        with torch.no_grad():
            if args.model == 'vit-Linear':
                image_features = net(pixel_values = images.float()).last_hidden_state
                image_features = image_features[:, 0, :].detach()
            elif args.model == 'CLIP-Linear':
                image_features = net.encode_image(images).float()
            if classifier: 
                if args.normalize: 
                    image_features /= image_features.norm(dim=-1, keepdim=True)  
                output = classifier(image_features)
                
            else: 
                image_features /= image_features.norm(dim=-1, keepdim=True) 
                output = image_features @ text_features.T
            smax = to_np(F.softmax(output/ args.T, dim=1))
            _score.append(-np.max(smax, axis=1)) 
            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = labels.cpu().numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:len(loader.dataset)].copy()   

def get_ood_scores_clip(args, net, loader, test_labels, in_dist=False, softmax = True):
    '''
    used for scores based on img-caption product inner products: MIP, entropy, energy score. 
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    _right_score = []
    _wrong_score = []
    if args.model == 'H-CLIP':
        tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)
    multi_template= args.score == 'MIPT'
    #debug
    # fingerprints = []
    # labels_all = []
    #end
    # wordnet_labels = []
    # if args.score == 'MIPT-wordnet':
    #     for c in test_labels:
    #         word = wn.synsets(c)[0]
    #         wordnet_labels.append([c])
    #         for i in range(0, 9):
    #             word = word.hypernyms()[0]
    #             wordnet_labels[i].append()

    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            if batch_idx >= len(loader.dataset)  // args.batch_size and in_dist is False:
                break
            bz = images.size(0)
            labels = labels.long().cuda()
            images = images.cuda()
            if args.model == 'CLIP':
                image_features = net.encode_image(images).float()
            elif args.model == 'H-CLIP':
                image_features = net.get_image_features(pixel_values = images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if multi_template:
                if (args.template == 'full'):
                    templates = openai_imagenet_template
                elif (args.template == 'subset1'):
                    templates = openai_imagenet_template_subset[0]
                elif (args.template == 'subset2'):
                    templates = openai_imagenet_template_subset[1]

                # output = torch.zeros(bz,len(test_labels), device = args.device)
                # template_weights = [0.4,0.15,0.15,0.15,0.15]
                # template_weights = [0.2,0.2,0.2,0.2,0.2]
                template_len = len(templates)
                text_features_avg = torch.zeros(args.n_cls, args.feat_dim, device = args.device)
                for i, temp in enumerate(templates):
                    text_inputs = torch.cat([clip.tokenize(temp(c)) for c in test_labels]).cuda()
                    text_features = net.encode_text(text_inputs)
                    text_features /= text_features.norm(dim=-1, keepdim=True) 
                    text_features_avg += text_features * 1/template_len
                text_features_avg /= text_features_avg.norm(dim=-1, keepdim=True) 
                output = image_features @ text_features_avg.T 
                
                # zeroshot_weights = []
                # for i, c in enumerate(test_labels):
                #     texts = [temp(c) for temp in templates]
                #     text_inputs = clip.tokenize(texts).cuda()
                #     text_features = net.encode_text(text_inputs).float()
                #     text_features /= text_features.norm(dim=-1, keepdim=True) 
                #     text_feature = text_features.mean(dim=0)
                #     text_feature /= text_feature.norm()
                #     zeroshot_weights.append(text_feature)
                #     # output += image_features @ text_features.T * template_weights[i]
                # zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
                # output = 100. * image_features @ zeroshot_weights
            elif args.score == 'MIPT-wordnet':
                template_len = len(templates)
                text_features_avg = torch.zeros(args.n_cls, 768, device = args.device)
                for i, temp in enumerate(templates):
                    text_inputs = torch.cat([clip.tokenize(temp(c)) for c in test_labels]).cuda()
                    text_features = net.encode_text(text_inputs)
                    text_features /= text_features.norm(dim=-1, keepdim=True) 
                    text_features_avg += text_features * 1/template_len
                text_features_avg /= text_features_avg.norm(dim=-1, keepdim=True) 
                output = image_features @ text_features_avg.T 
            else: # for MIP
                if args.model == 'CLIP':
                    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_labels]).cuda()
                    text_features = net.encode_text(text_inputs).float()
                elif args.model == 'H-CLIP':
                    text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels], padding=True, return_tensors="pt")
                    # for k in text_inputs.keys():
                    #     text_inputs[k] = text_inputs[k].cuda()
                    text_features = net.get_text_features(input_ids = text_inputs['input_ids'].cuda(), 
                                                    attention_mask = text_inputs['attention_mask'].cuda()).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)   
                output = image_features @ text_features.T
                if args.score == 'MIP_topk':
                    pass
                #debug 
                # fingerprints.append(to_np(output))
                # labels_all.append(to_np(labels))
                #end
            # output, _ = output.sort(descending=True, dim=1)[0:args.n_cls]
            if softmax:
                smax = to_np(F.softmax(output/ args.T, dim=1))
            else:
                smax = to_np(output/ args.T)
            # if multi_template:
            #     smax = smax * num_temp
            if args.score == 'energy':
                #Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))  #energy score is expected to be smaller for ID
            elif args.score == 'entropy':  
                from scipy.stats import entropy
                _score.append(entropy(smax)) 
            elif args.score in ['MIP', 'MIPT']:
                _score.append(-np.max(smax, axis=1)) 

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = labels.cpu().numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
    #debug 
    # if in_dist:
    #     name = 'img_templates/fingerprint.npy'
    # else:
    #     name = 'img_templates/fingerprint_ood.npy'
    # with open(name, 'wb') as f:
    #     np.save(f, concat(fingerprints))
    #     np.save(f, concat(labels_all))
    #end
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:len(loader.dataset)].copy()   

def get_ood_scores_clip_linear(args, net, classifier, loader, in_dist=False):
    '''
    used for scores based on logit layer (i.e. after fine-tuning a linear layer): MSP, entropy_logits, energy score. 
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    _right_score = []
    _wrong_score = []

    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object): #enumerate-> tqdm is the correct order; not tqdm -> enumerate
            if batch_idx >= len(loader.dataset)  // args.batch_size and in_dist is False:
                break
            labels = labels.long().cuda()
            images = images.cuda()
            if args.model == 'CLIP-Linear':
                image_features = net.encode_image(images).float()
            elif args.model == 'vit-Linear':
                image_features = net(pixel_values = images.float()).last_hidden_state
                image_features = image_features[:, 0, :].detach()
            if args.normalize: 
                image_features /= image_features.norm(dim=-1, keepdim=True)  
            output = classifier(image_features)
            # output, _ = output.sort(descending=True, dim=1)[0:args.n_cls]
            smax = to_np(F.softmax(output/ args.T, dim=1))
            if args.score == 'energy_logits':
                #Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))  #energy score is expected to be smaller for ID
            elif args.score == 'MSP':
                _score.append(-np.max(smax, axis=1)) 

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = labels.cpu().numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:len(loader.dataset)].copy()  

def get_MIPC_scores_clip(args, net, text_df, test_labels, in_dist=True):
    '''
    used for MIPC score. take the maximum of caption input to caption template inner product.
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    with torch.no_grad():
        text_templates = net.encode_text(torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_labels]).cuda())
        text_templates /= text_templates.norm(dim=-1, keepdim=True)
        text_inputs =torch.cat([clip.tokenize(sent) for sent in text_df["caption"]]).cuda()
        text_dataset = TextDataset(text_inputs, text_df["cls"])
        text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=args.batch_size, shuffle=False)
        tqdm_object = tqdm(text_loader, total=len(text_loader))   
        for batch_idx, (texts, labels) in enumerate(tqdm_object):
            text_features = net.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)   
            output = text_features @ text_templates.T
            # _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1)))) 
            # smax  = to_np(output/ args.T)
            smax = to_np(F.softmax(output/ args.T, dim=1))
            _score.append(-np.max(smax, axis=1)) 
        return concat(_score).copy()

def get_nouns_scores_clip(args, preprocess, net, image_loader, ID_labels, dataset_name, captions_nouns_dir = 'captions_nouns', in_dist = True):
    '''
    used for nouns score. 1 - sum_{i \in K} p(\hat{y}=i|x)
    '''
    import pandas as pd
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    captions_nouns_path = os.path.join(captions_nouns_dir, f'{dataset_name}_clipcap_captions_and_nouns.csv')
    df = pd.read_csv(f"{captions_nouns_path}", sep=',', converters={'Nouns': pd.eval})
    # text_dataset = TextDataset(df["Nouns"], df["Type"])
    # text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=args.batch_size, shuffle=False)
    bz = image_loader.batch_size
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(image_loader)):
            
            #if not in_dist:
            mean = torch.tensor(preprocess.transforms[-1].mean)
            std = torch.tensor(preprocess.transforms[-1].std)
            recovered_img = images[0]*std[:,None, None] + mean[:, None, None]
            plt.imsave(f'test_{i}.png', np.transpose(recovered_img.numpy(), (1,2,0)))

            if i >= 2000  // args.batch_size and in_dist is False:
                break
            generated_labels = list(df["Nouns"][i*bz: (i+1)*bz])[0]
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

def generate_img_template(args, net, preprocess, num_per_cls, template_dir):
    from collections import defaultdict
    per_class_counter = defaultdict(int)
    classwise_features = []
    for _ in range(args.n_cls):
       classwise_features.append([]) 
    train_loader = set_train_loader(args, preprocess, batch_size = 1, shuffle=True) #bz = 1; introduce some randomness for template selection
    with torch.no_grad():
        for image, label in tqdm(train_loader):
            if per_class_counter[label.item()] < num_per_cls:
                features = net.encode_image(image.cuda())
                features /= features.norm(dim=-1, keepdim=True)
                classwise_features[label.item()].append(features.view(1, -1))
                per_class_counter[label.item()] += 1
    for cls in range(args.n_cls):
        classwise_features[cls] = torch.cat(classwise_features[cls], 0)
    concat_features = torch.cat(classwise_features, 0)
    torch.save(concat_features, os.path.join(template_dir,f'img_template_{num_per_cls}.pt'))
    return concat_features

def get_retrival_scores_clip(args, net, text_df, preprocess, num_per_cls, generate = False, template_dir = 'img_templates', option = 'mode'):
    if generate:
        image_templates = generate_img_template(args, net, preprocess, num_per_cls, template_dir)
    else: 
        image_templates = torch.load(os.path.join(template_dir,f'img_template_{num_per_cls}.pt'), map_location= 'cpu').cuda()
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    text_inputs =torch.cat([clip.tokenize(sent) for sent in text_df["caption"]]).cuda()
    text_dataset = TextDataset(text_inputs, text_df["cls"])
    text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=args.batch_size, shuffle=False)
    tqdm_object = tqdm(text_loader, total=len(text_loader)) 
    with torch.no_grad():  
        for batch_idx, (texts, labels) in enumerate(tqdm_object):
            text_features = net.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)   
            output = text_features @ image_templates.T
            values, indices = torch.topk(output, k = num_per_cls, dim = 1) # by default top k largest
            if option == 'avg': # avg of top k retrived inner product scores
                score = to_np(torch.mean(values, dim = 1))
                _score.append(-score)
            elif option == 'mode':
                classes_id = indices // num_per_cls
                mode, count = stats.mode(to_np(classes_id), axis = 1)
                _score.append(-count.squeeze())

        return concat(_score).copy()

def get_retrival_scores_from_classwise_mean_clip(args, net, text_df, preprocess, softmax = True, generate = False, template_dir = 'img_templates', dataset_name=None, in_dist=False):
    if generate: 
        image_templates = get_mean(args, net, preprocess)
    else: 
        image_templates = torch.load(os.path.join(template_dir,f'classwise_mean_{args.in_dataset}.pt'), map_location= 'cpu').cuda()
    image_templates = image_templates.half() # cast dtype to float16
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    text_inputs =torch.cat([clip.tokenize(sent) for sent in text_df["caption"]]).cuda()
    text_dataset = TextDataset(text_inputs, text_df["cls"])
    text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=args.batch_size, shuffle=False)
    tqdm_object = tqdm(text_loader, total=len(text_loader))

    act = []
    with torch.no_grad():
        for batch_idx, (texts, labels) in enumerate(tqdm_object):
            text_features = net.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)   
            output = text_features @ image_templates.T
            smax = to_np(output)
            act.append(smax)
            if softmax:
                smax = to_np(F.softmax(output/ args.T, dim=1))
            _score.append(-np.max(smax, axis=1))
        
            _right_score = []
            _wrong_score = []
            # if in_dist:
            #     preds = np.argmax(smax, axis=1)
            #     targets = labels.cpu().numpy().squeeze()
            #     right_indices = preds == targets
            #     wrong_indices = np.invert(right_indices)

            #     _right_score.append(-np.max(smax[right_indices], axis=1))
            #     _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
    torch.save(act, f'img_templates/{args.score}_activations_{dataset_name}.pt')
    # return concat(_score).copy()
    if in_dist:
        print('accuracy not working yet')
        # return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
        return concat(_score).copy(), [1], [1]
    else:
        return concat(_score).copy()
    

def analysis_feature_manitude(args, net, preprocess, id_loader):
    args.normalize = False
    fid, _ = get_features(args, net, id_loader)
    fid_norm = np.linalg.norm(fid, axis = 1)
    print(f"in norms: {stats.describe(fid_norm)}")
    if args.in_dataset in ['ImageNet','ImageNet10', 'ImageNet100']: 
        out_datasets =  ['places365','SUN', 'dtd', 'iNaturalist']
    elif args.in_dataset == 'CIFAR-10':
        out_datasets = ['places365','SVHN', 'iSUN', 'dtd', 'LSUN', 'CIFAR-100']
    for out_dataset in out_datasets:
        print(f"Evaluting OOD dataset {out_dataset}")
        if args.in_dataset in ['ImageNet', 'ImageNet10', 'ImageNet100']:
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, 
                        root= os.path.join(args.root_dir,'ImageNet_OOD_dataset'))
        else: #for CIFAR
            ood_loader = set_ood_loader(args, out_dataset, preprocess)
        food, _ = get_features(args, net, ood_loader)
        food_norm = np.linalg.norm(food, axis = 1)
        print(f"out norms: {stats.describe(food_norm)}")
        plot_distribution(args, -fid_norm, -food_norm, out_dataset)

def get_knn_scores_from_img_encoder_id(args, net, train_loader, test_loader):
    '''
    used for KNN score for ID dataset  
    '''
    if args.generate: 
        ftrain, _ = get_features(args, net, train_loader, dataset = 'ID_train')
        ftest,_ = get_features(args, net, test_loader, dataset = 'ID_test')
    else:
        with open(os.path.join(args.template_dir, 'all_feat', f'all_feat_ID_train_{args.max_count}_{args.normalize}.npy'), 'rb') as f:
            ftrain =np.load(f)
        with open(os.path.join(args.template_dir,'all_feat', f'all_feat_ID_test_{args.max_count}_{args.normalize}.npy'), 'rb') as f:
            ftest =np.load(f)
    index = faiss.IndexFlatL2(ftrain.shape[1])
    ftrain = ftrain.astype('float32')
    ftest = ftest.astype('float32')
    index.add(ftrain)
    D, _ = index.search(ftest, args.K, )
    scores = D[:,-1]
    return scores, index

def get_knn_scores_from_img_encoder_ood(args, net, ood_loader, out_dataset, index):
    '''
    used for KNN score for OOD dataset
    '''
    if args.generate: 
        food, _ = get_features(args, net, ood_loader, dataset = out_dataset) 
    else: 
        with open(os.path.join(args.template_dir, 'all_feat', f'all_feat_{out_dataset}_{args.max_count}_{args.normalize}.npy'), 'rb') as f:
            food =np.load(f)
    D, _ = index.search(food.astype('float32'), args.K)
    scores_ood = D[:,-1]
    return scores_ood

def get_fp_scores_from_clip_id(args, net, test_labels, train_loader, test_loader):
    '''
    used for fingerprint knn score for ID dataset  
    '''
    if args.generate: 
        ftrain,  = get_fingerprint(args, net, test_labels, train_loader, dataset = 'ID_train')
        ftest,_ = get_fingerprint(args, net, test_labels, test_loader, dataset = 'ID_test')
    else:
        with open(os.path.join(args.template_dir, 'all_feat', f'all_fp_ID_train_{args.max_count}_softmax_{args.softmax}.npy'), 'rb') as f:
            ftrain =np.load(f)
        with open(os.path.join(args.template_dir,'all_feat', f'all_fp_ID_test_{args.max_count}_softmax_{args.softmax}.npy'), 'rb') as f:
            ftest =np.load(f) 
    
    ftrain = ftrain.astype('float32') / args.T
    ftest = ftest.astype('float32') /args.T

    ftrain = torch.from_numpy(ftrain)
    filter_val, _ = torch.topk(ftrain, k = 10, dim = 1)
    ftrain.masked_fill_(ftrain < filter_val[:,-1].view(-1,1),0) #filter_val[:,-1]: tensor([2.6386, 2.2513, 2.2839,  ..., 2.5120, 2.3718, 2.3910])
    ftrain = ftrain.numpy()

    ftest = torch.from_numpy(ftest)
    filter_val, _ = torch.topk(ftest, k = 10, dim = 1)
    ftest.masked_fill_(ftest < filter_val[:,-1].view(-1,1),0)
    ftest = ftest.numpy()

    n_neighbors = 20
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
    ftest = reducer.fit_transform(ftest)
    ftrain = reducer.fit_transform(ftrain)

    # ftrain = scipy.special.softmax(ftrain, axis = 1)
    # ftest = scipy.special.softmax(ftest, axis = 1)
    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)
    index_bad = index
    D, _ = index_bad.search(ftest, args.K, )
    scores = D[:,-1]
    return scores, index_bad

def get_fp_scores_from_clip_ood(args, net, test_labels, ood_loader, out_dataset, index_bad):
    '''
    used for fingerprint knn score for OOD dataset
    '''
    # args.generate = True
    if args.generate: 
        food, _ = get_fingerprint(args, net, test_labels, ood_loader,  dataset = out_dataset) 
    else: 
        with open(os.path.join(args.template_dir, 'all_feat', f'all_fp_{out_dataset}_{args.max_count}_softmax_{args.softmax}.npy'), 'rb') as f:
            food =np.load(f) 
        
    food = food.astype('float32') / args.T

    food = torch.from_numpy(food)
    filter_val, _ = torch.topk(food, k = 10, dim = 1)
    food.masked_fill_(food < filter_val[:,-1].view(-1,1),0)
    food = food.numpy()

    n_neighbors = 20
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
    food = reducer.fit_transform(food)

    # food = scipy.special.softmax(food, axis = 1)

    D, _ = index_bad.search(food, args.K)
    scores_ood = D[:,-1]
    return scores_ood

def get_mean_prec(args, net, train_loader):
    '''
    used for Mahalanobis score. Calculate class-wise mean and inverse covariance matrix
    '''
    classwise_mean = torch.empty(args.n_cls, args.feat_dim, device = args.device)
    all_features = []
    # classwise_features = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    # for _ in range(args.n_cls):
    #    classwise_features.append([]) 
    # train_loader = set_train_loader(args, preprocess, batch_size = args.batch_size, shuffle = False, subset = args.subset) #no Shuffle
    with torch.no_grad():
        for idx, (image, labels) in enumerate(tqdm(train_loader)):
            if args.model == 'CLIP':
                features = net.encode_image(image.to(args.device))
            elif args.model == 'vit':
                image = image.float().cuda()
                features = net(pixel_values = image).last_hidden_state
                features = features[:, 0, :].detach()
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
            # construct class-conditional sample matrix
            # classwise_features[label.item()] = torch.cat((classwise_features[label.item()], features.view(1, -1)), 0)
            #if len(classwise_features[label.item()]) < MAX_COUNT:
            # classwise_features[label.item()].append(features.view(1, -1))
            for label in labels:
                classwise_idx[label.item()].append(idx)
            all_features.append(features.cpu()) #for vit
    all_features = torch.cat(all_features)
    for cls in range(args.n_cls):
        # classwise_features[cls] = torch.cat(classwise_features[cls], 0)
        # classwise_mean[cls] = torch.mean(classwise_features[cls].float(), dim = 0)
        classwise_mean[cls] = torch.mean(all_features[classwise_idx[cls]].float(), dim = 0)
        if args.normalize: 
            classwise_mean[cls] /= classwise_mean[cls].norm(dim=-1, keepdim=True)
    # now, classwise_mean is of shape [10, 512]
    cov = torch.cov(all_features.T.double()) # shape: [512, 512]
    # cov = cov + 1e-7*torch.eye(all_features.shape[1]).cuda()
    precision = torch.linalg.inv(cov).float()
    print(f'cond number: {torch.linalg.cond(precision)}')
    torch.save(classwise_mean, os.path.join(args.template_dir,f'{args.model}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    torch.save(precision, os.path.join(args.template_dir,f'{args.model}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    return classwise_mean, precision

def get_mean(args, net, preprocess, mean_dir = 'img_templates'):
    '''
    used for Maha score. calculate class-wise mean only
    '''
    print(args.n_cls)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # feat_dim = 512
    # classwise_features = [torch.empty(0,feat_dim) for i in range(args.n_cls)]
    classwise_mean = torch.empty(args.n_cls, args.feat_dim)
    classwise_features = []
    for _ in range(args.n_cls):
       classwise_features.append([]) 
    train_loader = set_train_loader(args, preprocess, batch_size = 1) #bz = 1; no Shuffle
    with torch.no_grad():
        for image, label in tqdm(train_loader):
            features = net.encode_image(image.to(device))
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
            #construct class-conditional sample matrix
            # classwise_features[label.item()] = torch.cat((classwise_features[label.item()], features.view(1, -1)), 0)
            classwise_features[label.item()].append(features.view(1, -1))
    for cls in range(args.n_cls):
        classwise_features[cls] = torch.cat(classwise_features[cls], 0)
        classwise_mean[cls] = torch.mean(classwise_features[cls].float(), dim = 0)
    torch.save(classwise_mean, os.path.join(mean_dir,f'classwise_mean_{args.in_dataset}.pt'))
    return classwise_mean
            

def get_prec(args, net, train_loader):
    '''
    used for Maha score. calculate inverse covariance matrix only
    '''
    ftrain, _ = get_features(net, train_loader, args.device, normalize = args.normalize, to_np=False)
    cov = torch.cov(ftrain.T.double())
    # cov = cov + 1e-7*torch.eye(all_features.shape[1]).cuda()
    precision = torch.linalg.inv(cov).float()
    print(f'cond number: {torch.linalg.cond(precision)}')

    return precision

def get_Mahalanobis_score(args, net, test_loader, classwise_mean, precision, in_dist = True):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    '''
    # net.eval()
    Mahalanobis_score_all = []
    total_len = len(test_loader.dataset)
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            if (batch_idx >= total_len // args.batch_size) and in_dist is False:
                break   
            images, labels = images.cuda(), labels.cuda()
            if args.model == 'CLIP':
                features = net.encode_image(images)
            elif args.model == 'vit':
                features = net(pixel_values = images).last_hidden_state[:, 0, :]
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
            for i in range(args.n_cls):
                class_mean = classwise_mean[i]
                zero_f = features - class_mean
                Mahalanobis_dist = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                if i == 0:
                    Mahalanobis_score = Mahalanobis_dist.view(-1,1)
                else:
                    Mahalanobis_score = torch.cat((Mahalanobis_score, Mahalanobis_dist.view(-1,1)), 1)      
            Mahalanobis_score, _ = torch.max(Mahalanobis_score, dim=1)
            Mahalanobis_score_all.extend(-Mahalanobis_score.cpu().numpy())
        
    return np.asarray(Mahalanobis_score_all, dtype=np.float32)

def get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
    if args.out_as_pos: # in case out samples are defined as positive (as in OE)
        measures = get_measures(out_score, in_score)
    else:
        measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')
    # print(f'in score samples (min): {in_score[-3:]}, out score samples: {out_score[-3:]}')
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr) # used to calculate the avg over multiple OOD test sets
    print_measures(log, auroc, aupr, fpr, args.score)

class TextDataset(torch.utils.data.Dataset):
    '''
    used for MIPC score. wrap up the list of captions as Dataset to enable batch processing
    '''
    def __init__(self, texts, labels):
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # Load data and get label
        X = self.texts[index]
        y = self.labels[index]

        return X, y