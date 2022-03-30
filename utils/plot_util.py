
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import umap

def plot_distribution(args, id_scores, ood_scores, out_dataset):
    sns.set(style="white", palette="muted")
    sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", fill = True, alpha = 0.5)
    plt.title(f"ID v.s. {out_dataset} {args.score} score")
    # plt.ylim(0, 0.3)
    # plt.xlim(-10, 50)
    plt.savefig(os.path.join(args.log_directory,f"{args.score}_{out_dataset}.png"), bbox_inches='tight')

def plot_umap_id_only(name = 'all_feat_ID_train_500_True', template_dir = 'img_templates'):
    with open(os.path.join(template_dir, 'all_feat', f'{name}.npy'), 'rb') as f:
        feat =np.load(f)
        labels = np.load(f)
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

def plot_umap_id_ood(id = 'all_feat_ID_test_500_True', ood = 'all_feat_iNaturalist_500_True', ood_name = "iNaturalist", template_dir = 'img_templates'):
    ood = f'all_feat_{ood_name}_500_True'
    label_to_class_idx = {0: ood_name, 1: "ID"}
    with open(os.path.join(template_dir, 'all_feat', f'{id}.npy'), 'rb') as f:
        id_feat =np.load(f)
    with open(os.path.join(template_dir, 'all_feat', f'{ood}.npy'), 'rb') as f:
        ood_feat =np.load(f)
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

if __name__ == '__main__':
    ood_names = ['dtd','places365', 'SUN']
    # for ood_name in ood_names:
    #     plot_umap_id_ood(ood_name = ood_name)
    plot_umap_id_only()
