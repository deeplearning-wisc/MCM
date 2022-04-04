import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

num_cls = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# num_cls = [100, 500]
mean_list = []
var_list = []
trials = 5

for c in num_cls:
    df_list = []
    for exp_num in range(trials):
        # c_str = '' if c == 1000 else str(c)
        data_temp = {}
        try:
            name = f'imagenet-subset-exp{exp_num}'
            df = pd.read_csv(f'./results/ImageNet{str(c)}/MIP/CLIP_ViT-L/14_T_0.1_ID_{name}_normalize_False/{name}.csv', index_col=0)
            df_list.append(df)
                
        except FileNotFoundError:
            exit = os.system(f'python eval_ood_detection.py --score="MIP" --CLIP_ckpt="ViT-L/14" --gpu=4 --server="galaxy-01" --in_dataset="ImageNet-subset" --model="CLIP" --feat_dim=768 --batch-size=250 --T=0.1 --num_imagenet_cls={c} --name=imagenet-subset-exp{exp_num} --seed={exp_num}')
            if (exit > 0): quit()
    df = pd.concat(df_list, keys=list(range(trials)), names=['trials', 'dataset'])
    var = df.groupby(level=1).var()
    mean = df.groupby(level=1).mean()
    var_list.append(var)
    mean_list.append(mean)

mean_df = pd.concat(mean_list, keys=num_cls, names=['num_cls'])
var_df = pd.concat(var_list, keys=num_cls, names=['num_cls'])
stats = mean_df.axes[1]
datasets = mean_df.index.levels[1]

fig, axs = plt.subplots(1, 3)
fig.set_size_inches(18, 6)

for i, score in enumerate(stats):
    for dataset in datasets:
        mean = mean_df.loc[(slice(None), dataset), score].values
        var = var_df.loc[(slice(None), dataset), score].values
        axs[i].errorbar(num_cls, mean, yerr=var, label=dataset)
        axs[i].set_title(score)
plt.legend()
plt.tight_layout()
plt.savefig('imagenet_subset.png')
