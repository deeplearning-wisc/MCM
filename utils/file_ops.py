import os
import shutil 
import numpy as np
import logging
import pandas as pd


def save_scores(args, scores, dataset_name):
    with open(os.path.join(args.log_directory, f'{dataset_name}_scores.npy'), 'wb') as f:
        np.save(f, scores)

def load_scores(args, dataset_name):
    with open(os.path.join(args.log_directory, f'{dataset_name}_scores.npy'), 'rb') as f:
        scores = np.load(f)
    return scores
    
def setup_log(args):    
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(args.log_directory, "ood_eval_info.log"), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 
    log.debug(f"#########{args.name}############")
    return log

def save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list):
    fpr_list = [float('{:.2f}'.format(100*fpr)) for fpr in fpr_list]
    auroc_list = [float('{:.2f}'.format(100*auroc)) for auroc in auroc_list]
    aupr_list = [float('{:.2f}'.format(100*aupr)) for aupr in aupr_list]
    import pandas as pd
    data = {k:v for k,v in zip(out_datasets, zip(fpr_list,auroc_list,aupr_list))}
    data['AVG'] = [np.mean(fpr_list),np.mean(auroc_list),np.mean(aupr_list) ]
    data['AVG']  = [float('{:.2f}'.format(metric)) for metric in data['AVG']]
    # Specify orient='index' to create the DataFrame using dictionary keys as rows
    df = pd.DataFrame.from_dict(data, orient='index',
                       columns=['FPR95', 'AUROC', 'AUPR'])
    df.to_csv(os.path.join(args.log_directory,f'{args.name}.csv'))

def create_ImageNet_subset(src, dst, target_dirs):
    assert(os.path.exists(src))
    if not os.path.exists(dst):
        os.makedirs(dst)
    types = ['train', 'val']
    for type in types:
        for dir_name in os.listdir(os.path.join(src, type)):
            if dir_name in target_dirs:
                shutil.copytree(os.path.join(src, type, dir_name), os.path.join(dst,type, dir_name))

            
def prepare_dataframe(captions_dir = 'gen_captions', dataset_name = 'imagenet_val', multiple = False):
    # load caption file
    captions_path = os.path.join(captions_dir, f'{dataset_name}_captions.tsv')
    df = pd.read_csv(f"{captions_path}", sep='\t')
    df.columns = ["image_id","caption","cls"]
    if multiple: # in case a single img has multiple captions
        x = list(set(df['image_id'].values)) 
        image_ids = np.arange(0, len(x))
        train_images = [x[i] for i in image_ids]
        df = df[df["image_id"].isin(train_images)].reset_index(drop=True)
    return df
