import os, sys
import argparse
import numpy as np
import torch
import clip
from scipy import stats
# from torchvision.transforms import transforms
from utils.common import *
from utils.detection_util import *
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
# sys.path.append(os.path.dirname(__file__))

def process_args():
    parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #unique setting for each run
    parser.add_argument('--in_dataset', default='ImageNet30', type=str, 
                        choices = ['ImageNet30'], help='in-distribution dataset')
    parser.add_argument('--name', default = "test_imagenet30", type =str, help = "unique ID for the run")  
    parser.add_argument('--seed', default = 7, type =int, help = "random seed")  
    parser.add_argument('--server', default = 'inst-01', type =str, 
                choices = ['inst-01', 'inst-04', 'A100', 'galaxy-01', 'galaxy-02'], help = "on which server the experiment is conducted")
    parser.add_argument('--gpu', default=7, type=int, help='the GPU indice to use')
    # batch size
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                            help='mini-batch size; 1 for nouns score; 75 for odin_logits; 512 for other scores [clip]')
    #encoder loading
    parser.add_argument('--model', default='CLIP', choices = ['CLIP'], type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'RN50x4', 'ViT-L/14'], help='which pretrained img encoder to use')
    #detection setting  
    parser.add_argument('--score', default='MIP', type=str, choices = ['MIP'], help='score options')  
    # for ODIN score 
    parser.add_argument('--T', default = 1, type =float, help = "temperature") 
    # for fingerprint score 
    parser.add_argument('--softmax', type = bool, default = False, help='whether to apply softmax to the inner prod')
    #Misc 
    parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
    #for MIP variants score
    parser.add_argument('--template', default=['subset1'], type=str, choices=['full', 'subset1', 'subset2'])
    args = parser.parse_args()

    args.n_cls = 1
    
    if args.server in ['inst-01', 'inst-04']:
        args.root_dir = '/nobackup/dataset_myf' #save dir of dataset
        args.save_dir = f'/nobackup/checkpoints/clip_linear/{args.in_dataset}' # save dir of linear classsifier
    elif args.server in ['galaxy-01', 'galaxy-02']:
        args.root_dir = '/nobackup-slow/dataset'
        args.save_dir = f'/nobackup/zcai/checkpoints/clip_linear/{args.in_dataset}' # save dir of linear classsifier
    elif args.server in ['A100']:
        args.root_dir = ''

    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok= True)

    return args

def set_ood_loader_ImageNet30(args, out_dataset, preprocess, root = '/nobackup/dataset_myf'):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    exclude_idx = int(out_dataset.split("_")[-1])
    dataset = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet30', 'val'), transform=preprocess)
    all_idx = np.arange(len(dataset))
    indices = all_idx[np.array(dataset.targets) != exclude_idx]
    testsetout = torch.utils.data.Subset(dataset, indices)  
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return testloaderOut


def set_id_loader_ImageNet30(args, in_dataset, preprocess, root = '/nobackup/dataset_myf'):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    include_idx = int(in_dataset.split("_")[-1])
    dataset = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet30', 'val'), transform=preprocess)
    all_idx = np.arange(len(dataset))
    indices = all_idx[np.array(dataset.targets) == include_idx]
    testsetin = torch.utils.data.Subset(dataset, indices)  
    testloaderIn = torch.utils.data.DataLoader(testsetin, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return testloaderIn

def obtain_ImageNet30_class(id):
    all_labels = ['stingray', 'american_alligator', 'dragonfly', 'airliner', 
    'ambulance', 'banjo', 'barn', 'bikini', 'rotary_dial_telephone', 'digital_clock',
     'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hourglass', 'manhole_cover', 
     'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'schooner', 'snowmobile', 
     'soccer_ball', 'tank', 'toaster', 'hotdog', 'strawberry', 'volcano', 'acorn']
    return [all_labels[id]]

def get_ood_scores_clip(args, net, loader, test_labels, in_dist=False, softmax = True):
    '''
    used for scores based on img-caption product inner products: MIP, entropy, energy score. 
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    _right_score = []
    _wrong_score = []
    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            # if batch_idx >= len(loader.dataset)  // args.batch_size and in_dist is False:
            #     break
            bz = images.size(0)
            labels = labels.long().cuda()
            images = images.cuda()

            image_features = net.encode_image(images).float()

            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_labels]).cuda()
            text_features = net.encode_text(text_inputs).float()

            text_features /= text_features.norm(dim=-1, keepdim=True)   
            output = image_features @ text_features.T
            if softmax:
                smax = to_np(F.softmax(output/ args.T, dim=1))
            else:
                smax = to_np(output/ args.T)
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

def main():
    args = process_args()
    setup_seed(args)
    log = setup_log(args)
    torch.cuda.set_device(args.gpu)
    args.device = 'cuda'
    if args.model == "CLIP": #pre-trained CLIP
        net, preprocess = clip.load(args.CLIP_ckpt, args.gpu) 

    net.eval()
    id = 29
    in_dataset = f"ImageNet30_{id}" 
    out_datasets = [f"ImageNet30_{i}" for i in range(29)]

    test_loader = set_id_loader_ImageNet30(args,in_dataset, preprocess)
    test_label = obtain_ImageNet30_class(id)

    in_score, right_score, wrong_score= get_ood_scores_clip(args, net, test_loader, test_label, in_dist=True, softmax = False)

    num_right = len(right_score)
    num_wrong = len(wrong_score)
    log.debug('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
    log.debug('\n\nError Detection')

    with open(f'score_T_{args.T}_{args.in_dataset}.npy', 'wb') as f:
            np.save(f, in_score)
    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet30(args, out_dataset, preprocess)
        out_score = get_ood_scores_clip(args, net, ood_loader, test_label, softmax = False) 
        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        #debug
        with open(f'score_T_{args.T}_{out_dataset}.npy', 'wb') as f:
            np.save(f, out_score)
        #end
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)

if __name__ == '__main__':
    main()