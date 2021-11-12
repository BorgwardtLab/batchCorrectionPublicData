from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import anndata as an
import scanpy as sc



# Batch correction methods
def throw_out_marker_genes(data_in,metadata,n_throw_out=2):
    data = data_in.copy()

    gsm_drop = []
    for this_gse in metadata.gse.unique():
        n_gsm = len(metadata[metadata.gse == this_gse])
        if n_gsm <2:

            gsm_drop += list(metadata[metadata.gse == this_gse].index)
    data_sub = data.drop(gsm_drop)
    metadata_sub = metadata.drop(gsm_drop)


    adata = an.AnnData(X=data_sub,obs=metadata_sub)
    sc.pp.neighbors(adata,use_rep='X')
    sc.tl.leiden(adata)
    sc.tl.rank_genes_groups(adata, 'gse', method='t-test')
    to_throw_out = set()
    for n in range(n_throw_out):
        to_throw_out = to_throw_out.union(set(adata.uns['rank_genes_groups']['names'][n]))


    data.drop(columns=to_throw_out,inplace=True)
    print(data.shape)
    return data
def reducePCAMetaData(metasub, metatype,adata):

    if metatype == 'gse':
        gse_sep = metasub.copy()
        adata_sub = adata[adata.obs['gse'].isin(gse_sep)]
    else:
        if metasub == 'all':
            adata_sub = adata.copy()
        else:
            adata_sub = adata[adata.obs[metatype]==metasub]

    if adata_sub.shape[0] <5:
        print('too few samples for '+metasub+' in '+metatype)
        return adata

    # Calculate PCA
    n_component = int(adata_sub.shape[0]/5)
    if n_component < 3:
        n_component = 3

    pca = PCA(n_components=n_component)
    pca_fit = pca.fit(adata_sub.X)
    print(f'Variance ratios:{pca_fit.explained_variance_ratio_}')
    n_throw_out = (pca_fit.explained_variance_ratio_ > 0.1).sum() #int(n_component*0.2)#
    # if n_throw_out<3:
    #     n_throw_out = 3

    pca_componenents = pca_fit.components_[:n_throw_out]
    factors = np.matmul(adata.X, pca_componenents.T)

    # not super elegant but will do for now
    adata_new = adata.copy()
    for i in range(n_throw_out):
        # df = df - np.outer(factors[:,i],pca_componenents[i])
        adata_new.X = adata_new.X - np.outer(factors[:, i], pca_componenents[i])


    return adata_new
def getPCAMetaData(metasub, metatype,adata):

    if metatype == 'gse':
        gse_sep = metasub.copy()
        adata_sub = adata[adata.obs['gse'].isin(gse_sep)]
    else:
        if metasub == 'all':
            adata_sub = adata.copy()
        else:
            adata_sub = adata[adata.obs[metatype]==metasub]

    if adata_sub.shape[0] <5:
        print('too few samples for '+metasub+' in '+metatype)
        return adata

    # Calculate PCA
    n_component = int(adata_sub.shape[0]/3)
    if n_component < 3:
        n_component = 3

    pca = PCA(n_components=n_component)
    pca_fit = pca.fit(adata_sub.X)
    print(f'Variance ratios:{pca_fit.explained_variance_ratio_}')
    n_throw_out = (pca_fit.explained_variance_ratio_ > 0.3).sum() #int(n_component*0.2)#

    pca_componenents = pca_fit.components_[:n_throw_out]
    factors = np.matmul(adata.X, pca_componenents.T)

    return pca_componenents,factors
def throw_out_pca_zerohops(data_in, metadata, df_meta, obs = 'ZeroHop'):

    data  = data_in.copy()
    adata = an.AnnData(X=data, obs=metadata)
    sc.pp.neighbors(adata, use_rep='X')

    # Get data subset
    constr_name = 'pca_'

    # Loop over all ZHs in order of samples per ZH
    zhs          = metadata[obs].unique()
    zh_n_samples = [df_meta.loc[z,'n_samples'] for z in zhs]
    sort_ind = sorted(range(len(zh_n_samples)), key=lambda k: zh_n_samples[k], reverse=False)
    zhs_sorted = [zhs[i] for i in sort_ind]
    for z in zhs_sorted:
        if z == 'nan':
            continue
        else:
            constr_name += z+'_'
            adata = reducePCAMetaData(z, obs, adata)

    data = pd.DataFrame(adata.X, columns=adata.var_names.values, index=adata.obs.index)
    return data

