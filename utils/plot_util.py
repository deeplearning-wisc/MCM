
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import umap
import scipy

# plot kde plots
def plot_distribution(args, id_scores, ood_scores, out_dataset):
    sns.set(style="white", palette="muted")
    sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", fill = True, alpha = 0.5)
    plt.title(f"ID v.s. {out_dataset} {args.score} score")
    # plt.ylim(0, 0.3)
    # plt.xlim(-10, 50)
    plt.savefig(os.path.join(args.log_directory,f"{args.score}_{out_dataset}.png"), bbox_inches='tight')

# plot umaps
def plot_umap_id_only(name = 'all_feat_ID_test_500_True', template_dir = 'img_templates', subset = False):
    with open(os.path.join(template_dir, 'all_feat', f'{name}.npy'), 'rb') as f:
        feat =np.load(f)
        labels = np.load(f)
    if subset:
        size = 50000
        idx = np.random.choice(range(len(feat)), size = size, replace = False)
        feat = feat[idx]
        labels = labels[idx]
    n_neighbors = 20
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
    name +=f'n_neighbor_{n_neighbors}'
    umap_results = reducer.fit_transform(feat)
    print(umap_results.shape)
    # Create the figure
    fig = plt.figure( figsize=(16,16) )
    ax = fig.add_subplot(1, 1, 1, title='Umap' )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Create the scatter
    import seaborn as sns
    import pandas as pd
    data = pd.DataFrame(list(zip(umap_results[:,0],umap_results[:,1], labels)), columns =['x','y', 'labels'])
    ax = sns.scatterplot(x= 'x', y= 'y', data = data, cmap ='Spectral', c = labels )
    ax.legend(fontsize = 15) 

    plt.tight_layout()
    plt.savefig(f'{name}_umap_train_subsample.pdf')

def plot_umap_id_ood(id = 'all_feat_ID_test_500_True', ood_name = "ID_train", template_dir = 'img_templates'):
    ood = f'all_feat_{ood_name}_500_True'
    label_to_class_idx = {0: ood_name, 1: "ID"}
    with open(os.path.join(template_dir, 'all_feat', f'{id}.npy'), 'rb') as f:
        id_feat =np.load(f)
    with open(os.path.join(template_dir, 'all_feat', f'{ood}.npy'), 'rb') as f:
        ood_feat =np.load(f)
        if len(ood_feat) > len(id_feat): # subsample
            size = len(id_feat)
            idx = np.random.choice(range(len(ood_feat)), size = size, replace = False)
            ood_feat = ood_feat[idx]
    labels = np.zeros(len(id_feat) + len(ood_feat))
    labels[:len(id_feat)] = 1
    feat = np.concatenate((id_feat, ood_feat))
    n_neighbors = 20
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
    ood_name +=f'n_neighbor_{n_neighbors}'
    umap_results = reducer.fit_transform(feat)
    print(umap_results.shape)
    # Create the figure
    fig = plt.figure( figsize=(16,16) )
    ax = fig.add_subplot(1, 1, 1, title='Umap')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Create the scatter plot
    import seaborn as sns
    # colors = ['#f4d35e', '#f95738', '#9381ff', '#00a8e8', '#a57548']
    colors = ['#f4d35e', '#f95738']
    sns.set_palette(sns.color_palette(colors))
    labels = [label_to_class_idx[i] for i in labels]
    import pandas as pd
    data = pd.DataFrame(list(zip(umap_results[:,0],umap_results[:,1], labels)), columns =['x','y', 'labels'])
    ax = sns.scatterplot(x= 'x', y= 'y', data = data, palette = colors, hue='labels' )
    ax.legend(fontsize = 15) 

    plt.tight_layout()
    plt.savefig(f'{ood_name}_umap.pdf')

# plot histograms
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

def plot_hist():
    x = np.array([0.2859, 0.1933, 0.1683, 0.1937, 0.1974, 0.1822, 0.1898, 0.1540, 0.1974, 0.1968])
    T = 100
    x = x / T
    softmax = np.exp(x)/sum(np.exp(x)).tolist()
    print(softmax)
    class_idx = list(range(10))
    temp_data = pd.DataFrame(list(zip(softmax ,class_idx)), columns =['softmax pr','class_idx'])
    # ax = sns.barplot(x="alpha", y="AUROC", data=auroc_alpha,
    #                  palette=sns.color_palette("summer"))

    ax = sns.barplot(x="class_idx", y="softmax pr", data=temp_data,
                    palette = ['#e5f99d','#c2f694','#a0f290','#75ec9f','#51e6bc','#27e1e7','#0bbdf2','#0fa2e7','#34619f','#2e568e' ])
    # ax.set(ylim=(94, 96.5))
    ax.set_title(f"T = {T}")
    show_values_on_bars(ax)
    plt.savefig(f"T={T}.pdf", bbox_inches = 'tight',
        pad_inches = 0)

def plot_umap_id_fingerprint(name = 'fingerprint', template_dir = 'img_templates', T = 1):
    with open(os.path.join(template_dir, f'{name}.npy'), 'rb') as f:
        feat =np.load(f)
        labels = np.load(f)
    feat *= T
    feat = scipy.special.softmax(feat, axis = 1)
    n_neighbors = 20
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
    name +=f'n_neighbor_{n_neighbors}'
    umap_results = reducer.fit_transform(feat)
    print(umap_results.shape)
    # Create the figure
    fig = plt.figure( figsize=(16,16) )
    ax = fig.add_subplot(1, 1, 1, title='Umap' )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Create the scatter
    import seaborn as sns
    import pandas as pd
    data = pd.DataFrame(list(zip(umap_results[:,0],umap_results[:,1], labels)), columns =['x','y', 'labels'])
    ax = sns.scatterplot(x= 'x', y= 'y', data = data, cmap ='Spectral', c = labels )
    ax.legend(fontsize = 15) 

    plt.tight_layout()
    plt.savefig(f'{name}_umap_T_{T}.pdf')

if __name__ == '__main__':
    ood_names = ['dtd','places365', 'SUN']
    # for ood_name in ood_names:
    #     plot_umap_id_ood(ood_name = ood_name)
    #plot_umap_id_only()
    # plot_umap_id_ood()
    plot_umap_id_fingerprint(T = 0.1)

