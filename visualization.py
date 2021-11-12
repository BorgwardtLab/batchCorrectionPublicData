import matplotlib as mpl
import numpy as np
import scanpy as sc
import os
import matplotlib.pyplot as plt


# Plotting
def load_LRres(model, name, output):

    folder = os.path.join(os.getcwd(),output)

    bal_accs_batch_LR_raw = []
    infile = open(os.path.join(folder, model + '_' + name + '_bal_accs_batch.txt'), 'r')
    for line in infile:
        bal_accs_batch_LR_raw.append(float(line.strip().split(',')[0]))
    infile.close()

    f1s_batch_LR_raw = []
    infile = open(os.path.join(folder, model + '_' + name + '_f1s_batch.txt'), 'r')
    for line in infile:
        f1s_batch_LR_raw.append(float(line.strip().split(',')[0]))
    infile.close()

    bal_accs_zeroH_LR_raw = []
    infile = open(os.path.join(folder, model + '_' + name + '_bal_accs_zeroH.txt'), 'r')
    for line in infile:
        bal_accs_zeroH_LR_raw.append(float(line.strip().split(',')[0]))
    infile.close()

    f1s_zeroH_LR_raw = []
    infile = open(os.path.join(folder, model + '_' + name + '_f1s_zeroH.txt'), 'r')
    for line in infile:
        f1s_zeroH_LR_raw.append(float(line.strip().split(',')[0]))
    infile.close()

    return bal_accs_batch_LR_raw, f1s_batch_LR_raw, bal_accs_zeroH_LR_raw, f1s_zeroH_LR_raw
def save_LRres(model, name, bal_accs_batch_LR_raw, f1s_batch_LR_raw, bal_accs_zeroH_LR_raw, f1s_zeroH_LR_raw, output):

    folder = os.path.join(os.getcwd(),output)
    with open(os.path.join(folder,model+'_'+name+'_bal_accs_batch.txt'), 'w') as f:
        for item in bal_accs_batch_LR_raw:
            f.write("%s\n" % item)

    with open(os.path.join(folder, model + '_' + name + '_f1s_batch.txt'), 'w') as f:
        for item in f1s_batch_LR_raw:
            f.write("%s\n" % item)

    with open(os.path.join(folder, model + '_' + name + '_bal_accs_zeroH.txt'), 'w') as f:
        for item in bal_accs_zeroH_LR_raw:
            f.write("%s\n" % item)

    with open(os.path.join(folder, model + '_' + name + '_f1s_zeroH.txt'), 'w') as f:
        for item in f1s_zeroH_LR_raw:
            f.write("%s\n" % item)
def plot_new(adata_raw,output, name):
    sc.pp.neighbors(adata_raw, n_neighbors=10, n_pcs=40)
    sc.tl.tsne(adata_raw)

    metadata = adata_raw.obs

    medium_lesscoarse = []
    for m in metadata['MediumCoarse']:

        if m == 'both' or m == 'notGluconeogenic' or m == 'notDefined' or m == 'Gluconeogenic':
            medium_lesscoarse.append('Defined')
        elif m == 'lessRich' or m == 'moreRich' or m == 'rich0':
            medium_lesscoarse.append('Rich')
        elif m == 'in_vivo':
            medium_lesscoarse.append('in vivo')
        elif m == 'in_vivo_like':
            medium_lesscoarse.append('in vivo mimic')
        else:
            medium_lesscoarse.append(m)
    adata_raw.obs['MediumCoarseRel'] = medium_lesscoarse
    metadata['MediumCoarseRel'] = medium_lesscoarse


    gse_ind = []
    uni_gse = list(metadata.gse.unique())
    for g in metadata['gse']:
        gse_ind.append(uni_gse.index(g))
    adata_raw.obs['gse_ind'] = gse_ind
    metadata['gse_ind'] = gse_ind


    zh = [int(iz) for iz in metadata.ZeroHop]
    adata_raw.obs['ZH'] = zh
    metadata['ZH'] = zh

    # activate latex text rendering
    to_colour_by = ['strain',
     'GrowthPhase',
     'ZH',
     'Oxygenation',
     'Culture_Coarse',
     'Temperature_Coarse',
     'MediumCoarseRel',
     'Antibiotic',
     'gse_ind' ]

    for c in to_colour_by:

        if c ==  'GrowthPhase':
           legendEntry = ['Exponential', 'Plateau']
        elif c ==   'Culture_Coarse':
           legendEntry = ['Film', "in vivo", 'Liquid', 'Plate']
        elif c ==   'Temperature_Coarse':
           legendEntry = ['20-25$^\circ$C', '28-32$^\circ$C', '35-38$^\circ$C']
        elif c == 'Antibiotic':
            legendEntry = ['No exposure', 'Interval', 'Continuous']
        elif c == 'Oxygenation':
            legendEntry = ['Aerobic', 'Hypoxic']
        elif c == 'MediumCoarse':
            legendEntry = ['(Not) gluconeogenic', 'Less Rich', 'Rich', 'Not gluconeogenic', 'Defined', 'Rich', 'Defined']
        else:
            legendEntry = metadata[c].unique()

        cmap = plt.cm.jet  # define the colormap
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # force the first color entry to be grey
        cmaplist[0] = (.5, .5, .5, 1.0)

        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)
        sc.pl.tsne(adata_raw, color=[c], show=True, ncols=1, hspace=0.25, legend_fontsize=12, alpha = 0.5,
                   color_map =cmap )
        plt.title(' ')

        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        plt.legend(current_handles,legendEntry,loc='best')


        fig = plt.gcf()
        fig.set_size_inches(5,5)
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), output, name + '_' + c + '_tsne.png'))
def makeBoxPlot(data_a, data_b, ticks, label_a, label_b, output, ylabel='Shannon Entropy', title='Zero-hop cluster'):


    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(8, 5))

    bpl = plt.boxplot(data_a, positions=np.array(np.arange(len(data_a))) * 2.0 - 0.4, patch_artist=True,
                      sym='', widths=0.6, notch=True,
                      boxprops=dict(facecolor='red', color='red'))
    bpr = plt.boxplot(data_b, positions=np.array(np.arange(len(data_b))) * 2.0 + 0.4, patch_artist=True,
                      sym='', widths=0.6, notch=True,
                      boxprops=dict(facecolor='black', color='black'))
    set_box_color(bpl, 'red')
    set_box_color(bpr, 'black')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='red', label=label_a)
    plt.plot([], c='black', label=label_b)

    ymin = np.min([np.percentile(data_a, 2.5),np.percentile(data_b,2.5)])
    ymax = np.max([np.percentile(data_a, 97.5), np.percentile(data_b, 97.5)])
    if ymin <0:
        multipl_min = 1.2
    elif ymin == 0:
        multipl_min = -0.1
        ymin = 1
    else:
        multipl_min = 0.8

    plt.ylim([multipl_min*ymin, 1.2*ymax])
    plt.legend()
    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, rotation=45, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), output, ylabel + '_' + title + '_boxplot.png'))
def makeBoxPlotSingle(data_a, ticks, label_a, output,  ylabel='Shannon Entropy', title='Zero-hop cluster'):


    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(8, 5))

    bpl = plt.boxplot(data_a, positions=np.array(np.arange(len(data_a))), patch_artist=True,
                      sym='', widths=0.6, notch=True,
                      boxprops=dict(facecolor='black', color='black'))

    set_box_color(bpl, 'black')  # colors are from http://colorbrewer2.org/


    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='black', label=label_a)
    #plt.legend()

    plt.xticks(np.arange(len(ticks)), ticks, rotation=45, fontsize = 14)
    # plt.xlim(-2, len(ticks) * 2)
    # plt.ylim(0, 8)
    plt.ylabel(ylabel, fontsize = 14)
    #plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), output, ylabel + '_' + title + '_boxplot.png'))

