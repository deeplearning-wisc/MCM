import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import torchvision
import sklearn.metrics as sk
from transformers import CLIPTokenizer
from torchvision import datasets
import torch.nn.functional as F
import torchvision


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
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
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                        transform=preprocess)
    elif out_dataset == 'ImageNet10': # the train split is used due to larger and comparable size with ID dataset
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'train'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
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
  
def get_mean_prec(args, net, train_loader):
    '''
    used for Mahalanobis score. Calculate class-wise mean and inverse covariance matrix
    '''
    classwise_mean = torch.empty(args.n_cls, args.feat_dim, device =args.gpu)
    all_features = []
    # classwise_features = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.cuda()
            if args.model == 'CLIP': 
                features = net.get_image_features(pixel_values = images).float()
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
            for label in labels:
                classwise_idx[label.item()].append(idx)
            all_features.append(features.cpu()) #for vit
    all_features = torch.cat(all_features)
    for cls in range(args.n_cls):
        classwise_mean[cls] = torch.mean(all_features[classwise_idx[cls]].float(), dim = 0)
        if args.normalize: 
            classwise_mean[cls] /= classwise_mean[cls].norm(dim=-1, keepdim=True)
    cov = torch.cov(all_features.T.double()) 
    precision = torch.linalg.inv(cov).float()
    print(f'cond number: {torch.linalg.cond(precision)}')
    torch.save(classwise_mean, os.path.join(args.template_dir,f'{args.model}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    torch.save(precision, os.path.join(args.template_dir,f'{args.model}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    return classwise_mean, precision

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
                features = net.get_image_features(pixel_values = images).float()
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

def get_ood_scores_clip(args, net, loader, test_labels, in_dist=False):
    '''
    used for scores based on img-caption product inner products: MIP, entropy, energy score. 
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)

    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            bz = images.size(0)
            labels = labels.long().cuda()
            images = images.cuda()
  
            image_features = net.get_image_features(pixel_values = images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if args.model == 'CLIP':
                text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels], padding=True, return_tensors="pt")
                text_features = net.get_text_features(input_ids = text_inputs['input_ids'].cuda(), 
                                                attention_mask = text_inputs['attention_mask'].cuda()).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)   
                output = image_features @ text_features.T
            if args.score == 'max-logit':
                smax = to_np(output)
            else:
                smax = to_np(F.softmax(output/ args.T, dim=1))
            if args.score == 'energy':
                #Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))  #energy score is expected to be smaller for ID
            elif args.score == 'entropy':  
                # raw_value = entropy(smax)
                # filtered = raw_value[raw_value > -1e-5]
                _score.append(entropy(smax, axis = 1)) 
                # _score.append(filtered) 
            elif args.score == 'var':
                _score.append(-np.var(smax, axis = 1))
            elif args.score in ['MCM', 'max-logit']:
                _score.append(-np.max(smax, axis=1)) 
    return concat(_score)[:len(loader.dataset)].copy()   



def get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
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
