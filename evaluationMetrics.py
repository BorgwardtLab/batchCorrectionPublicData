from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score,recall_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans,AgglomerativeClustering

import numpy as np
import anndata as an
import scanpy as sc
import os
import matplotlib.pyplot as plt
import tqdm



# Evaluation Metrics
def DRSall(adata_new):
    datafield = 'ZeroHop'
    batchfield = 'gse'
    metadata_use = adata_new.obs
    cdist = pairwise_distances(adata_new.X)

    dist_dzh = []
    dist_szh_db = []
    for s in list(metadata_use.index):
        i = np.where(metadata_use.index == s)[0][0]
        this_dist = cdist[i, :]
        this_ZH = metadata_use.loc[s, datafield]
        this_batch = metadata_use.loc[s, batchfield]
        ind_diff_ZH = np.where(metadata_use[datafield] != this_ZH)[0]
        ind_same_ZH_diffbatch = \
        np.where(np.logical_and(metadata_use[datafield] == this_ZH, metadata_use[batchfield] != this_batch))[0]

        # min Distance to nearest sample from a different ZH cluster
        dist_diffZH = np.min(this_dist[ind_diff_ZH])
        dist_dzh.append(dist_diffZH)
        if dist_diffZH == 0:
            print('equivalentsample!!')
            print(s)
            print(i)

        # min Distance to the closest sample of the same ZH but different batch
        dist_sameZHdiffBatch = np.min(this_dist[ind_same_ZH_diffbatch])
        if dist_sameZHdiffBatch == 0:
            print('equivalentsample!!')
            print(s)
            print(i)

        dist_szh_db.append(dist_sameZHdiffBatch)

        drs_exp_all= np.mean(np.exp(np.array(dist_dzh) / np.array(dist_szh_db)))-1
        drs_log2_all = np.mean(np.log2(np.array(dist_dzh) / np.array(dist_szh_db)))

    return drs_exp_all, drs_log2_all
def DRSperZH(adata_new):
    datafield = 'ZeroHop'
    batchfield = 'gse'
    metadata_use = adata_new.obs
    cdist = pairwise_distances(adata_new.X)

    drs_exp_all = []
    drs_log2_all = []
    for zh in metadata_use[datafield].unique():
        dist_dzh = []
        dist_szh_db = []
        metadata_use_sub = metadata_use[metadata_use[datafield]==zh]

        for s in list(metadata_use_sub.index):
            i = np.where(metadata_use.index == s)[0][0]
            this_dist = cdist[i,:]
            this_ZH = metadata_use.loc[s,datafield]
            this_batch = metadata_use.loc[s,batchfield]
            ind_diff_ZH = np.where(metadata_use[datafield]!=this_ZH)[0]
            ind_same_ZH_diffbatch = np.where(np.logical_and(metadata_use[datafield]==this_ZH,metadata_use[batchfield]!=this_batch))[0]

            # min Distance to nearest sample from a different ZH cluster
            dist_diffZH = np.min(this_dist[ind_diff_ZH])
            dist_dzh.append(dist_diffZH)
            if dist_diffZH == 0:
                print('equivalentsample!!')
                print(s)
                print(i)

            # min Distance to the closest sample of the same ZH but different batch
            dist_sameZHdiffBatch = np.min(this_dist[ind_same_ZH_diffbatch])
            if dist_sameZHdiffBatch == 0:
                print('equivalentsample!!')
                print(s)
                print(i)

            dist_szh_db.append(dist_sameZHdiffBatch)

        drs_exp_all.append(np.mean(np.exp(np.array(dist_dzh)/np.array(dist_szh_db)))-1)
        drs_log2_all.append(np.mean(np.log2(np.array(dist_dzh) / np.array(dist_szh_db))))

    return drs_exp_all, drs_log2_all
def getnormShannonEntropy(data_filtered, metadata_filtered, n = 5, obs = 'Zero-hop cluster'):

    # calcualate pairwise distance
    from scipy.stats import entropy
    distances = pairwise_distances(data_filtered, data_filtered)

    # get the n nearest neighbours
    all_p_zero = []
    all_p_batch = []
    all_entr_zero = []
    all_entr_batch = []
    for i, s in enumerate(data_filtered.index):

        # get n nn
        thisdist = distances[i,:]
        n_inds   = sorted(range(len(thisdist)), key=lambda k: thisdist[k])[:n+1]

        n_inds.remove(i)
        # number of nn in same zero hop as start:
        # zero_start = metadata_filtered.loc[s][obs]
        # n_zero     = metadata_filtered.iloc[n_inds][obs].value_counts()[zero_start]
        # p_zero     = n_zero/n
        # all_p_zero.append(p_zero)
        zero_entropy = entropy(metadata_filtered.iloc[n_inds][obs].value_counts())
        all_entr_zero.append(zero_entropy)

        # number of nn in same batch
        # batch_start = metadata_filtered.loc[s]['gse']
        # #n_batch     = metadata_filtered.iloc[n_inds]['gse'].value_counts()[batch_start]
        # p_batch     = n_batch/n
        # all_p_batch.append(p_batch)
        batch_entroy = entropy(metadata_filtered.iloc[n_inds]['gse'].value_counts())
        all_entr_batch.append(batch_entroy)


    mean_batch_entr = np.mean(all_entr_batch)
    mean_zero_entr = np.mean(all_entr_zero)

    # ideal: high batch entropy, low zero_hop entropy

    # plt.figure()
    # plt.hist(all_p_zero, alpha = 0.5, label = 'ZeroHop')
    # plt.hist(all_p_batch, alpha=0.5, label='Batch')
    # plt.legend()
    #
    # plt.figure()
    # plt.hist(all_entr_zero, alpha = 0.5, label = 'ZeroHop')
    # plt.hist(all_entr_batch, alpha=0.5, label='Batch')
    # plt.legend()
    # plt.title('Entroy')
    # plt.tight_layout()


    return all_entr_zero, all_entr_batch
def calc_MeanZeroHopDistEmbedding(data, metadata, obs, embedding = 'tsne'):

    # Get embedding
    adata = an.AnnData(X=data,obs=metadata)
    if embedding == 'tsne':
        sc.tl.tsne(adata,use_rep='X')
    elif embedding == 'umap':
        sc.pp.neighbors(adata,use_rep='X')
        sc.tl.umap(adata)
    elif embedding == 'pca':
        sc.tl.pca(adata,use_highly_variable=False)
    else:
        raise('invalid embedding!')


    mean_dists = []
    std_dists = []
    X_use = adata.obsm['X_'+embedding]
    for z in metadata[obs].unique():
        if z == 'nan':
            continue
        else:
            data_sub = X_use[metadata[obs] == z,:]
            distances = pairwise_distances(data_sub, data_sub)
            these_mean_dists = [np.mean(np.concatenate([distances[:i, i], distances[i + 1:, i]])) for i in
                                range(0, distances.shape[0])]

            mean_dists.append(np.mean(these_mean_dists))
            std_dists.append(np.std(these_mean_dists))

    return mean_dists, std_dists
def calc_MeanZeroHopDist(data, metadata, obs):


    # get overall max distance
    distances_all = pairwise_distances(data, data)
    distances_use_all = []
    for i in range(0, data.shape[0] - 1):
        distances_use_all += list(distances_all[i, i + 1:])
    max_dist      = np.max(distances_use_all)
    mean_dist     = np.mean(distances_use_all)
    median_dist   = np.median(distances_use_all)
    # plt.figure()
    # plt.hist(distances_all.flatten())




    max_dists      = []
    std_dists_max  = []

    mean_dists_mean = []
    std_dists_mean  = []
    med_dists_med = []
    std_dists_med  = []
    for z in metadata[obs].unique():
        if z == 'nan':
            continue
        else:
            data_sub       = data[metadata[obs] == z]
            dists          = pairwise_distances(data_sub, data_sub)

            distances_use = []
            for i in range(0, data_sub.shape[0]-1):
                distances_use+= list(dists[i,i+1:])

            mean_dists_mean.append(np.mean(distances_use)/mean_dist)
            std_dists_mean.append(np.std(distances_use)/mean_dist)
            med_dists_med.append(np.median(distances_use)/median_dist)
            std_dists_med.append(np.std(distances_use)/median_dist)
            max_dists.append(np.max(distances_use) / max_dist)
            std_dists_max.append(np.std(distances_use) / max_dist)

    return max_dists, std_dists_max, mean_dists_mean, std_dists_mean, med_dists_med, std_dists_med
def check_clusters(adata_new, n_cluster=10, name = 'raw', batch = 'gse', meta = 'Zero-hop cluster coarse', clustering = 'louvain'):


    # Perfrom clustering
    sc.pp.neighbors(adata_new, use_rep='X')
    sc.tl.tsne(adata_new)

    if clustering == 'louvain':
        resolution = 0.1
        n_cluster_obtained_lou = 0
        while n_cluster_obtained_lou != n_cluster:
            sc.tl.louvain(adata_new, resolution=resolution)
            n_cluster_obtained_lou = len(adata_new.obs[clustering].unique())
            resolution = resolution * 1.2

            if  n_cluster_obtained_lou > n_cluster:
                resolution = (resolution/1.2)*0.95
                sc.tl.louvain(adata_new, resolution=resolution)
                n_cluster_obtained_lou = len(adata_new.obs['louvain'].unique())

    elif clustering == 'leiden':
        resolution = 0.1
        n_cluster_obtained_lou = 0
        while n_cluster_obtained_lou != n_cluster:
            sc.tl.leiden(adata_new, resolution=resolution)
            n_cluster_obtained_lou = len(adata_new.obs[clustering].unique())
            resolution = resolution * 1.2

            if n_cluster_obtained_lou > n_cluster:
                resolution = (resolution / 1.2) * 0.95
                sc.tl.leiden(adata_new, resolution=resolution)
                n_cluster_obtained_lou = len(adata_new.obs['leiden'].unique())

    elif clustering == 'Kmeans':

        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(adata_new.X)
        adata_new.obs[clustering] = kmeans.labels_.astype(str)
        n_cluster_obtained_lou = len(adata_new.obs[clustering].unique())



    sc.pl.tsne(adata_new, color=[clustering])
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(),output, clustering+'_clustering_'+name+'_tsne.png'))



    p_batch_lou   = []
    p_meta_lou    = []
    n_batches = len(adata_new.obs[batch].unique())
    n_meta   = len(adata_new.obs[meta].unique())
    for c in range(n_cluster_obtained_lou):
        adata_sub = adata_new[adata_new.obs[clustering] == str(c)]

        # Check heterogeneity of clusters with respect to gse
        this_n_batch = len(adata_sub.obs[batch].unique())
        p_batch      = this_n_batch/(n_batches/n_cluster_obtained_lou)
        p_batch_lou.append(p_batch)

        # if len(adata_sub.obs['gse'].unique()) < 5:
        #     for this_gse in adata_sub.obs['gse'].unique():
        #         if this_gse not in gse_sep:
        #             gse_sep.append(this_gse)

        # Check homogeneity with respect to metadata
        this_n_meta = len(adata_sub.obs[meta].unique())
        p_meta      = this_n_meta/(n_meta/n_cluster_obtained_lou)
        p_meta_lou.append(p_meta)



    return p_batch_lou, p_meta_lou
def predictSimple(X, Y, n_splits, clf_choice):

    # split per batch and metadata
    skf = StratifiedKFold(n_splits=n_splits)

    f1s_batch = []
    accs_batch = []
    bal_accs_batch = []
    precs_batch = []
    recs_batch = []
    for train_index, val_index in skf.split(X, Y):

        # Split
        X_trn = X.iloc[train_index].values
        y_trn = Y[train_index]

        X_val = X.iloc[val_index]
        y_val = Y[val_index]

        if clf_choice == 'LogisticRegression':
            clf = LogisticRegression(max_iter=10000, class_weight='balanced', solver='lbfgs')
            param_grid = {'C': [0.001, 0.005, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'penalty': ['l2']}
            randomizer = False
        elif clf_choice == 'LightGBM':

            # train_data = lgb.Dataset(X_train, label=y_train)
            # val_data = lgb.Dataset(X_test, label=y_test,  reference=train_data)
            # param = {'num_leaves': 31, 'objective': 'binary'}
            # param['metric'] = 'auc'
            # num_round = 2000

            # lgb.cv(param, train_data, num_round, nfold=5)
            # bst = lgb.train(param, train_data, num_round, valid_sets=[val_data])
            # bst = lgb.train(param, train_data, num_round, valid_sets=[val_data], early_stopping_rounds=200)
            # bst.save_model('modelLGB.txt', num_iteration=bst.best_iteration)
            # ypred_trn = bst.predict(train_data, num_iteration=bst.best_iteration)
            # ypred_val = bst.predict(val_data, num_iteration=bst.best_iteration)

            # Gridsearch
            clf = lgb.LGBMClassifier(
                random_state=10,
                n_estimators=50,
                num_leaves=30,
                max_depth=8,
                learning_rate=0.01,
            )
            param_grid = {'n_estimators': range(10, 600, 50), 'num_leaves': range(10, 200, 20)}
            # fit_params = {"early_stopping_rounds": 100,
            #               "valid_sets": [val_data]}
            # gridSearchCV = GridSearchCV(estimator=clf,
            #                             param_grid=params_grid,
            #                             scoring='roc_auc',
            #                             n_jobs=4,
            #                             iid=False,
            #                             verbose=1,
            #                             cv=5)
            # gridSearchCV.fit(X_train,y_train)
            # gridSearchCV.grid_scores_, gridSearchCV.best_params_, gridSearchCV.best_score_

            randomizer = False
        elif clf_choice == 'RandomForest':
            clf = RandomForestClassifier(random_state=42, class_weight='balanced')
            param_grid = {'bootstrap': [True, False],
                          'max_depth': [5, 10, 50],  # 60, 70, 80, 90, 100, None],
                          'max_features': ['auto'],  # 'sqrt'],
                          'min_samples_leaf': [1, 3, 10],
                          'min_samples_split': [3],
                          'n_estimators': [100, 300, 1000]  # , 1400, 1600, 1800, 2000]
                          }
        elif clf_choice == 'DecisionTree':
            clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
            param_grid = {'criterion': ['gini', 'entropy'],
                          'splitter': ['best', 'random'],
                          'max_depth': [10, 20, None],  # 20],# 30, 40, 50], # 60, 70, 80, 90, 100, None],
                          'max_features': ['auto'],  # 'sqrt'],
                          'min_samples_leaf': [1, 2],  # , 4],
                          'min_samples_split': [2, 5],  # 10],
                          #                  'n_estimators': [200, 400, 600],# 800, 1000, 1200]#, 1400, 1600, 1800, 2000]
                          }
        else:
            print("Invalid Classifier!")

        ####Get classifier:
        print('Getting classifier')
        classifier = clf.fit(X_trn, y_trn)  # get_classifier(X_trn, y_trn, clf, param_grid, randomizer)

        #### Print coefficients (weights):
        if clf_choice == 'RandomForest':
            # plt.figure(figsize=(15,15))
            # plot_tree(classifier.best_estimator_.estimators_[0], filled=True, feature_names=df.columns)
            # plt.suptitle('RandomForestPlot for: ' + key)
            # plt.show(block=False)

            # Create DOT data
            # dot_data = tree.export_graphviz(classifier.best_estimator_.estimators_[0], out_file=None,
            #                                 feature_names=X_train.columns, filled=True)
            #        class_names=iris.target_names)

            # Draw graph
            # graph = pydotplus.graph_from_dot_data(dot_data)

            # Show graph
            # Image(graph.write_pdf("./adni_mri_analysis/figures/" + key + "_RFTree.pdf"))

            for coef, name in zip(classifier.best_estimator_.feature_importances_.ravel(), X.columns):
                print(name, coef)

        if clf_choice == 'DecisionTree':
            # plt.figure(figsize=(20,20))
            # plot_tree(classifier.best_estimator_, filled=True, feature_names=df.columns)
            # plt.suptitle('DecistionTreePlot for: ' + key)
            # plt.show(block=False)
            # Create DOT data
            dot_data = tree.export_graphviz(classifier.best_estimator_, out_file=None, feature_names=X.columns,
                                            class_names=list(
                                                np.array(classifier.best_estimator_.classes_, dtype=str)),
                                            filled=True)

            for coef, name in zip(classifier.best_estimator_.feature_importances_.ravel(), X.columns):
                print(name, coef)

        # # gridSearchCV.grid_scores_, gridSearchCV.best_params_, gridSearchCV.best_score_
        # print("Best parameters for " + clf_choice + ": ", classifier.best_params_)
        # print("Best score for " + clf_choice + ": ", classifier.best_score_)

        y_pred_class = classifier.predict(X_val)
        f1s_batch.append(f1_score(y_val, y_pred_class, average="macro"))
        accs_batch.append(accuracy_score(y_val, y_pred_class))
        bal_accs_batch.append(balanced_accuracy_score(y_val, y_pred_class))
        precs_batch.append(precision_score(y_val, y_pred_class, average="macro"))
        recs_batch.append(recall_score(y_val, y_pred_class, average="macro"))

    print('F1: ' + str(np.mean(f1s_batch)))
    print('Acc: ' + str(np.mean(accs_batch)))
    print('bal Acc: ' + str(np.mean(bal_accs_batch)))
    print('Precision: ' + str(np.mean(precs_batch)))
    print('Recall: ' + str(np.mean(recs_batch)))

    return accs_batch, bal_accs_batch, f1s_batch, precs_batch, recs_batch
def get_classifier(X_train, y_train, classifier, param_grid, randomizer=False):
    '''
    Create and fit a Logistic Regression model.
    Parameters:
    -----------
    X: data
    y: labels
    classifier: classifier (e.g. LogisticRegression())
    param_grid: Parameter Grid

    Returns:
    --------
    classifier
    X_test
    y_test
    '''
    if randomizer == False:
        classifier = GridSearchCV(classifier, param_grid, cv=5, verbose=1, scoring='roc_auc')  # score=...
    else:
        classifier = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, n_iter=100, cv=3,
                                        verbose=1, random_state=42, n_jobs=-1,
                                        scoring='average_precision')  # 'roc_auc')

    classifier.fit(X_train, y_train)

    return classifier
def calc_LRtoPredBatch(data_standardised,metadata, clf_choice, obs_batch = 'gse', obs_meta = 'ZeroHop'):

    n_splits = 10

    # generate split per batch and metadata
    X = data_standardised
    Ybatch = metadata[obs_batch]
    Ymetacat = metadata[obs_meta]

    # predict
    accs_batch_LR, bal_accs_batch_LR, f1s_batch_LR, precs_batch_LR, recs_batch_LR = predictSimple(X, Ybatch, n_splits, clf_choice)
    accs_batch_LR, bal_accs_zeroH_LR, f1s_zeroH_LR, precs_zeroH_LR, recs_zeroH_LR = predictSimple(X, Ymetacat, n_splits,clf_choice)


    return bal_accs_batch_LR, f1s_batch_LR, bal_accs_zeroH_LR, f1s_zeroH_LR
def calc_CorWithBatch(data):
    # Not implemented yet!

    acc  = 0
    return acc
def LDA_score(adata,datafield='ZeroHop'):

    n_splits = 10

    # Peform CV
    skf = StratifiedKFold(n_splits=n_splits)
    X = adata.X
    Y = adata.obs[datafield].values

    lda_score = []
    for train_index, val_index in skf.split(X, Y):

        # Split
        X_trn = X[train_index]
        y_trn = Y[train_index]


        X_val = X[val_index]
        y_val = Y[val_index]

        LDA = LinearDiscriminantAnalysis(n_components=2)
        X_new = LDA.fit_transform(X_trn,y_trn)
        lda_score.append(LDA.score(X_val, y_val))


    return lda_score
def min_speration_n_cluster(adata):

    n_gse = len(adata.obs['gse'].unique())
    n_ZH = len(adata.obs['ZeroHop'].unique())
    zh_break = np.zeros((n_ZH,1))
    zh_tosplit = list(adata.obs['ZeroHop'].unique())
    for n_clusters in tqdm(range(1,n_gse)):

        if len(zh_tosplit)>0:
            adata.obs['Clustering'] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(adata.X)

            for zh in zh_tosplit:
                izh = int(zh)
                df_tmp = adata.obs.loc[adata.obs['ZeroHop'] == zh, ['ZeroHop', 'Clustering']]

                if not np.all(df_tmp.eq(df_tmp.iloc[0],axis='columns')):
                    print('ZH'+zh+' broken!')
                    zh_break[izh] = n_clusters-1
                    zh_tosplit.remove(zh)


    return [z[0] for z in list(zh_break)]
def cluster_impurities(adata,datafield='ZeroHop',n_clusters=10,measure='gini'):


    adata.obs['Clustering'] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(adata.X)
    if measure == 'gini':
        ginis = []
        for i in range(n_clusters):
            df_tmp = adata.obs.loc[adata.obs['Clustering']==i,[datafield]]
            probs = df_tmp.value_counts(normalize=True).values
            ginis.append(1-np.sum(np.power(probs,2)))
        return ginis
    elif measure == 'purity':
        purities = []
        n_points = adata.obs.shape[0]
        for i in range(n_clusters):
            df_tmp = adata.obs.loc[adata.obs['Clustering']==i,[datafield]]
            purity = np.max(df_tmp.value_counts(normalize=False).values)/df_tmp.shape[0]
            purities.append(purity)
        return purities
def distance_decrease(adata,data,data_orig, datafield='ZeroHop'):
    # data_orig = pd.read_csv(data_path+'data_standardized.csv',index_col=0)
    # data_orig.sort_index(inplace=True)
    data_orig = data_orig.loc[adata.obs.index]

    cdist_idx = np.triu_indices(data_orig.shape[0],k=1)
    cdist_orig = pairwise_distances(data_orig.values)[cdist_idx]
    mean_orig = np.mean(cdist_orig)
    median_orig = np.median(cdist_orig)

    cdist = pairwise_distances(data.values)[cdist_idx]
    mean = np.mean(cdist)
    median = np.median(cdist)

    data_clusters = adata.obs[datafield].dropna().unique()
    #data_clusters.sort()
    clusters_ratio_mean = []
    clusters_ratio_median = []
    for d in data_clusters:
        idx=adata[adata.obs[datafield]==d].obs.index
        tmp = adata[adata.obs[datafield]==d].X
        tmp_orig = data_orig.loc[idx].values
        cdist_idx = np.triu_indices(len(idx),k=1)
        cdist_tmp = pairwise_distances(tmp)[cdist_idx]
        cdist_tmp_orig = pairwise_distances(tmp_orig)[cdist_idx]
        clusters_ratio_mean.append((np.mean(cdist_tmp)/mean)/(np.mean(cdist_tmp_orig)/mean_orig))
        clusters_ratio_median.append((np.median(cdist_tmp)/median)/(np.median(cdist_tmp_orig)/median_orig))

    return clusters_ratio_mean,clusters_ratio_median

