import csv
from matplotlib import pyplot as plt

# num_cls = [10, 100, 200, 400, 500, 600, 800, 1000]
num_cls = [100, 500]
stats = ['FPR95', 'AUROC', 'AUPR']
data = {}

for c in num_cls:
    c_str = '' if c == 1000 else str(c)
    with open(f'./results/ImageNet{c_str}/MIP/CLIP_ViT-L/14_T_0.1_ID_save_fingerprint_normalize_True/save_fingerprint.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            for score in stats:
                if line[''] in data:
                    if score in data[line['']]:
                        data[line['']][score] += [line[score]]
                    else:
                        data[line['']][score] = [line[score]]
                else:
                    data[line['']] = {score: [line[score]]}


fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
fig.set_size_inches(9, 3)
for i, score in enumerate(stats):
    for dataset in data.keys():
        axs[i].plot(num_cls, data[dataset][score])
plt.tight_layout()
plt.savefig('imagenet_subset.png')
