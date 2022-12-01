
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import seaborn as sns
import torch.nn.functional as F

def plot_distribution(args, id_scores, ood_scores, out_dataset):
    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']
    sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
    plt.savefig(os.path.join(args.log_directory,f"{args.score}_{out_dataset}.png"), bbox_inches='tight')

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize=9) 
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


