
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import umap
import scipy
import pandas as pd
import torch
import seaborn as sns
import torch.nn.functional as F
# plot kde plots
def plot_distribution(args, id_scores, ood_scores, out_dataset):
    # args.score = 'CLS'
    sns.set(style="white", palette="muted")
    sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", fill = True, alpha = 0.5)
    plt.title(f"ID v.s. {out_dataset} {args.score} score")
    # plt.ylim(0, 0.3)
    # plt.xlim(-10, 50)
    plt.savefig(os.path.join(args.log_directory,f"{args.score}_{out_dataset}.png"), bbox_inches='tight')


# plot umaps
def plot_umap_id_only(name = 'all_feat_ID_test_100_False', template_dir = '/nobackup/img_templates', subset = False):
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
    plt.savefig(f'{name}_umap_train_subsample.png')

def plot_umap_id_ood(id = 'all_feat_ID_train_100_False', ood_name = "ID_train", template_dir = '/nobackup/img_templates',  name = 'knn_debug_original_40_10', subset = False):
    ood = f'all_feat_{ood_name}_100_False'
    label_to_class_idx = {0: ood_name, 1: "ID"}
    with open(os.path.join(template_dir, 'all_feat', name, f'{id}.npy'), 'rb') as f:
        id_feat =np.load(f)
    if subset:
        size = 50000
        idx = np.random.choice(range(len(id_feat)), size = size, replace = False)
        id_feat = id_feat[idx]
    with open(os.path.join(template_dir, 'all_feat',name, f'{ood}.npy'), 'rb') as f:
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
    plt.savefig(f'{name}_{ood_name}_umap.png')

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

def plot_hist_all_output(template_dir = '/nobackup/img_templates', dataset = 'ID_test', max_count = 500, softmax = False, T = 1):
    with open(os.path.join(template_dir, 'all_feat', f'all_fp_{dataset}_{max_count}_softmax_{softmax}.npy'), 'rb') as f:
            output =np.load(f)
    output = output.astype('float32') / T
    # output = scipy.special.softmax(output, axis = 1)

    for j in range(20):
        class_idx = ['i' for i in range(1000)][j*50: (1+j)*50]
        temp_data = pd.DataFrame(output[:5000,j*50: (1+j)*50].tolist(), columns =class_idx)
        # ax = sns.barplot(x="alpha", y="AUROC", data=auroc_alpha,
        #                  palette=sns.color_palette("summer"))
        ax = temp_data.boxplot(column=class_idx, fontsize = 1, figsize = (500, 20))  
        # ax = sns.boxplot(x="class_idx", y="inner prod", data=temp_data, palette=sns.color_palette("summer"))
        # ax.set(ylim=(94, 96.5))
        ax.set_title(f"{dataset}_T = {T}_class_idx_{j*50} to {50*(j+1)}")
        show_values_on_bars(ax)
        plt.savefig(f"{dataset}_T={T}_{j}.png", bbox_inches = 'tight', pad_inches = 0)

def plot_umap_id_fingerprint(name = 'fingerprint', template_dir = 'img_templates', T = 1):
    with open(os.path.join(template_dir, f'{name}.npy'), 'rb') as f:
        feat =np.load(f)
        labels = np.load(f)
    feat /= T
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

    data = pd.DataFrame(list(zip(umap_results[:,0],umap_results[:,1], labels)), columns =['x','y', 'labels'])
    ax = sns.scatterplot(x= 'x', y= 'y', data = data, cmap ='Spectral', c = labels )
    ax.legend(fontsize = 15) 

    plt.tight_layout()
    plt.savefig(f'{name}_umap_T_{T}.pdf')

def debug_umap_fingerprint(out_dataset, id_dataset, template_dir = '/nobackup/img_templates', T = 1, k = 10, softmax = False):

    with open(os.path.join(template_dir, 'all_feat', f'all_fp_{id_dataset}_500_softmax_False.npy'), 'rb') as f:
        fid =np.load(f)
        labels = np.load(f)
    fid = fid.astype('float32') / T

    fid = torch.from_numpy(fid)
    if softmax:
        fid = F.softmax(fid, dim = 1)
    filter_val_id, _ = torch.topk(fid, k = k, dim = 1)
    fid.masked_fill_(fid < filter_val_id[:,-1].view(-1,1),0)
    # if softmax:
    #     fid = F.softmax(filter_val, dim = 1)
    fid = fid.numpy()

    with open(os.path.join(template_dir, 'all_feat', f'all_fp_{out_dataset}_500_softmax_False.npy'), 'rb') as f:
            food =np.load(f) 
        
    food = food.astype('float32') / T

    food = torch.from_numpy(food)
    if softmax:
        food = F.softmax(food, dim = 1)
    filter_val_ood, _ = torch.topk(food, k = k, dim = 1)
    food.masked_fill_(food < filter_val_ood[:,-1].view(-1,1),0)
    # if softmax:
    #     food = F.softmax(filter_val, dim = 1)
    food = food.numpy()


    # feat = scipy.special.softmax(feat, axis = 1)
    labels = np.zeros(len(fid) + len(food))
    labels[:len(fid)] = 1
    label_to_class_idx = {0: out_dataset, 1: "ID"}
    debug_distribution(fid.max(axis = 1), food.max(axis =1), out_dataset, option = 'max', k = k, softmax = softmax)
    debug_distribution(filter_val_id.mean(axis = 1), filter_val_ood.mean(axis = 1), out_dataset, option = 'mean', k = k, softmax = softmax)
    return
    fall = np.concatenate((fid, food))
    n_neighbors = 20
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
    umap_results = reducer.fit_transform(fall)
    print(umap_results.shape)
    # Create the figure
    fig = plt.figure( figsize=(16,16) )
    ax = fig.add_subplot(1, 1, 1, title='Umap' )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Create the scatter
    colors = ['#f4d35e', '#f95738']
    sns.set_palette(sns.color_palette(colors))
    labels = [label_to_class_idx[i] for i in labels]
    data = pd.DataFrame(list(zip(umap_results[:,0],umap_results[:,1], labels)), columns =['x','y', 'labels'])
    ax = sns.scatterplot(x= 'x', y= 'y', data = data, palette = colors, hue='labels' )
    ax.legend(fontsize = 15) 

    plt.tight_layout()
    name =f'{out_dataset}_{id_dataset}_n_neighbor_{n_neighbors}'
    plt.savefig(f'{name}_umap_T_{T}_k={k}.pdf')

def debug_distribution(id_scores, ood_scores, out_dataset, option = 'max', k = 10, softmax = False):
    sns.set(style="white", palette="muted")
    sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", fill = True, alpha = 0.5)
    plt.title(f"ID v.s. {out_dataset} score")
    # plt.ylim(0, 0.3)
    # plt.xlim(-10, 50)
    plt.savefig(os.path.join("PLOTS", f"topK_softmax={softmax}_k={k}_{option}_{out_dataset}.png"), bbox_inches='tight')

if __name__ == '__main__':
    # plot_umap_id_only()
    ood_names = ['dtd','places365', 'SUN', 'iNaturalist']
    for ood_name in ood_names:
        plot_umap_id_ood(ood_name = ood_name)
    # plot_umap_id_ood()
    # plot_umap_id_fingerprint(T = 0.1)
    # plot_hist_all_output(T = 1)

    # softmax = True
    # for k in [3]:
    #     debug_umap_fingerprint(out_dataset = 'SUN', id_dataset = 'ID_test', k = k, softmax = softmax)
    #     debug_umap_fingerprint(out_dataset = 'iNaturalist', id_dataset = 'ID_test', k = k, softmax = softmax)

