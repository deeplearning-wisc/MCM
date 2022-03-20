import os
import torch
import numpy as np
from tqdm import tqdm
import clip
import torchvision
import sklearn.metrics as sk
from utils.common import get_features
import utils.svhn_loader as svhn
from torchvision.transforms import transforms
import torch.nn.functional as F
import faiss
from scipy import stats
from utils.train_eval_util import set_train_loader
from utils.imagenet_templates import openai_imagenet_template


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
    elif out_dataset == 'places365':
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)   
    elif out_dataset == 'dtd':
        # root = '/nobackup/dataset_myf'
        # testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'ood_datasets', 'dtd', 'images'),
        #                             transform=preprocess) 
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Textures'),
                                    transform=preprocess) 
    # if len(testsetout) > 10000: 
    #     testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=0)
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

def get_ood_scores_clip(args, net, loader, test_labels, in_dist=False, softmax = True):
    '''
    used for scores based on img-caption product inner products: MSP (MIP), entropy, energy score. 
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    _right_score = []
    _wrong_score = []
    multi_template=args.score == 'MIPT'

    tqdm_object = tqdm(loader, total=len(loader))
    if multi_template:
        num_temp = 80
        text_inputs_list = [torch.cat([clip.tokenize(temp(c)) for c in test_labels]).cuda() for temp in openai_imagenet_template[0:num_temp]]
    else:
        text_inputs_list = [torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_labels]).cuda()]

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            if batch_idx >= len(loader.dataset)  // args.batch_size and in_dist is False:
                break
            labels = labels.long().cuda()
            images = images.cuda()
            image_features = net.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            output_list = []
            for i, text_inputs in enumerate(text_inputs_list):
                # text_inputs_list[i].cuda()
                text_features = net.encode_text(text_inputs_list[i])
                text_features /= text_features.norm(dim=-1, keepdim=True)   
                # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                output = image_features @ text_features.T
                output_list.append(output)
            output_list = torch.cat(output_list)
            # output, _ = output.sort(descending=True, dim=1)[0:args.n_cls]
            if softmax:
                smax = to_np(F.softmax(output_list/ args.T, dim=1))
            else:
                smax = to_np(output_list/ args.T)

            # if multi_template:
            #     smax = smax * num_temp

            if args.score == 'energy':
                #Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                _score.append(-to_np((args.T*torch.logsumexp(output_list / args.T, dim=1))))  #energy score is expected to be smaller for ID
            elif args.score == 'entropy':  
                from scipy.stats import entropy
                _score.append(entropy(smax)) 
            else: # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
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

def get_retrival_scores_from_classwise_mean_clip(args, net, text_df, preprocess, softmax = True, generate = False, template_dir = 'img_templates'):
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
    with torch.no_grad():  
        for batch_idx, (texts, labels) in enumerate(tqdm_object):
            text_features = net.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)   
            output = text_features @ image_templates.T
            if softmax:
                smax = to_np(F.softmax(output/ args.T, dim=1))
            _score.append(-np.max(smax, axis=1))
        return concat(_score).copy()
       
def get_knn_scores_from_clip_img_encoder_id(args, net, train_loader, test_loader):
    '''
    used for KNN score. ID dataset only 
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ftrain, _ = get_features(net, train_loader, device, args.normalize)
    ftest,_ = get_features(net, test_loader, device, args.normalize)
    index = faiss.IndexFlatL2(ftrain.shape[1])
    ftrain = ftrain.astype('float32')
    ftest = ftest.astype('float32')
    index.add(ftrain)
    index_bad = index
    D, _ = index_bad.search(ftest, args.K, )
    scores = D[:,-1]
    return scores, index_bad

def get_knn_scores_from_clip_img_encoder_ood(args, net, ood_loader, index_bad):
    '''
    used for KNN score. OOD dataset only
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    food, _ = get_features(net, ood_loader, device, args.normalize)
    food = food.astype('float32')
    D, _ = index_bad.search(food, args.K)
    scores_ood = D[:,-1]
    return scores_ood

def get_mean_prec(args, net, preprocess):
    '''
    used for Mahalanobis score. Calculate class-wise mean and inverse covariance matrix
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dim = 512
    # classwise_features = [torch.empty(0,feat_dim) for i in range(args.n_cls)]
    classwise_mean = torch.empty(args.n_cls, feat_dim)
    all_features = []
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
            all_features.append(features)
    for cls in range(args.n_cls):
        classwise_features[cls] = torch.cat(classwise_features[cls], 0)
        classwise_mean[cls] = torch.mean(classwise_features[cls].float(), dim = 0)
    # now, classwise_mean is of shape [10, 512]
    all_features = torch.cat(all_features, 0)
    cov = torch.cov(all_features.T.double()) # shape: [512, 512]
    # cov = cov + 1e-7*torch.eye(all_features.shape[1]).cuda()
    precision = torch.linalg.inv(cov).float()
    print(f'cond number: {torch.linalg.cond(precision)}')

    return classwise_mean, precision

def get_mean(args, net, preprocess, mean_dir = 'img_templates' ):
    '''
    used for Maha score. calculate class-wise mean only
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dim = 512
    # classwise_features = [torch.empty(0,feat_dim) for i in range(args.n_cls)]
    classwise_mean = torch.empty(args.n_cls, feat_dim)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ftrain, _ = get_features(net, train_loader, device, normalize = args.normalize, to_np=False)
    cov = torch.cov(ftrain.T.double())
    # cov = cov + 1e-7*torch.eye(all_features.shape[1]).cuda()
    precision = torch.linalg.inv(cov).float()
    print(f'cond number: {torch.linalg.cond(precision)}')

    return precision

def get_Mahalanobis_score(args, net, test_loader, classwise_mean, precision):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    '''
    # net.eval()
    Mahalanobis = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # if batch_idx >= num_batches and in_dist is False:
            #     break       
            images, labels = images.cuda(), labels.cuda()
            features = net.encode_image(images)
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
            for i in range(args.n_cls):
                class_mean = classwise_mean[i]
                zero_f = features - class_mean
                Mahalanobis_dist = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                if i == 0:
                    noise_gaussian_score = Mahalanobis_dist.view(-1,1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, Mahalanobis_dist.view(-1,1)), 1)      
            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
            Mahalanobis.extend(-noise_gaussian_score.cpu().numpy())
        
    return np.asarray(Mahalanobis, dtype=np.float32)

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