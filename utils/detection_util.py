import os
import torch
import numpy as np
from tqdm import tqdm
import clip
import torchvision
import sklearn.metrics as sk
import utils.svhn_loader as svhn
from torchvision.transforms import transforms
import torch.nn.functional as F


def set_ood_loader(args, out_dataset, preprocess, root = '/nobackup/dataset_myf'):
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
    elif out_dataset == 'places365':
        # root_tmp= "/nobackup-slow/dataset/places365_test/test_subset" #galaxy
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'places365'),
            transform=preprocess)
    elif out_dataset == 'cifar100':
        testsetout = torchvision.datasets.CIFAR100(root=os.path.join(root, 'cifar100'), train=False, download=True, transform=preprocess)
    elif out_dataset == 'cifar10':
        testsetout = torchvision.datasets.CIFAR10(root=os.path.join(root, 'cifar10'), train=False, download=True, transform=preprocess)
    else:
        testsetout = torchvision.datasets.ImageFolder(os.path.join(root, "ood_datasets", f"{out_dataset}"),
                                    transform=preprocess)
    
    if len(testsetout) > 10000: 
        testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=True, num_workers=8)
    return testloaderOut

def set_ood_loader_ImageNet(args, out_dataset, preprocess, root = '/nobackup/dataset_myf_ImageNet_OOD'):
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)   
    elif out_dataset == 'dtd':
        root = '/nobackup/dataset_myf'
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'ood_datasets', 'dtd', 'images'),
                                    transform=preprocess) 
    # if len(testsetout) > 10000: 
    #     testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=8)
    return testloaderOut

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

def get_ood_scores_clip(args, net, loader, test_labels, in_dist=False):
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    _right_score = []
    _wrong_score = []

    tqdm_object = tqdm(loader, total=len(loader))
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_labels]).cuda()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            if batch_idx >= len(loader.dataset)  // args.batch_size and in_dist is False:
                break
            labels = labels.long().cuda()
            images = images.cuda()
            image_features = net.encode_image(images)
            text_features = net.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)   
            # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            output = image_features @ text_features.T
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
                targets = labels.cpu().numpy().squeeze()
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
    print(f'in score samples: {in_score[:3]}, out score samples: {out_score[:3]}')
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr) # used to calculate the avg over multiple OOD test sets
    print_measures(log, auroc, aupr, fpr, args.score)