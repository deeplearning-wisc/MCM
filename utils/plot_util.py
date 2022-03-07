
import seaborn as sns
from matplotlib import pyplot as plt
# import numpy as np
import os


def plot_distribution(args, id_scores, ood_scores, out_dataset):
    sns.set(style="white", palette="muted")
    sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", fill = True, alpha = 0.5)
    plt.title(f"ID v.s. {out_dataset} {args.score} score")
    # plt.ylim(0, 0.3)
    # plt.xlim(-10, 50)
    plt.savefig(os.path.join(args.log_directory,f"{args.score}_{out_dataset}.png"), bbox_inches='tight')


