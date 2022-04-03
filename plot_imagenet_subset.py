import csv
import os
from matplotlib import pyplot as plt

num_cls = [10, 100, 400, 500, 600, 800, 1000]
# num_cls = [100, 500]
stats = ['FPR95', 'AUROC', 'AUPR']
data = {}

for c in num_cls:
    for exp_num in range(5):
        # c_str = '' if c == 1000 else str(c)
        try:
            name = f'imagenet-subset-exp{exp_num}'
            with open(f'./results/ImageNet{str(c)}/MIP/CLIP_ViT-L/14_T_0.1_ID_{name}_normalize_False/{name}.csv') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    for score in stats:
                        if line[''] in data:
                            if score in data[line['']]:
                                data[line['']][score] += [float(line[score])]
                            else:
                                data[line['']][score] = [float(line[score])]
                        else:
                            data[line['']] = {score: [float(line[score])]}
        except:
            os.system(f'python eval_ood_detection.py --score="MIP" --CLIP_ckpt="ViT-L/14" --gpu=6 --server="galaxy-01" --in_dataset="ImageNet-subset" --model="CLIP" --feat_dim=768 --batch-size=250 --T=0.1 --num_imagenet_cls={c} --name=imagenet-subset-exp{exp_num} --seed={exp_num}')

print(data)

fig, axs = plt.subplots(1, 3)
fig.set_size_inches(18, 6)

for i, score in enumerate(stats):
    for dataset in data.keys():
        axs[i].plot(num_cls, data[dataset][score], label=dataset)
        axs[i].set_title(score)
plt.legend()
plt.tight_layout()
plt.savefig('imagenet_subset.png')
