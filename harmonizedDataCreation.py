################################################################################
#################         METADATA NEAREST NEIGHBOURS         ##################
################################################################################

# 1) Load the data including information on zero hop neighbourhoods
# 2) Filter out samples with no internal/external repeat experiment (no replica)
# 3) Run batch correction
# 4) Compare batch correction techniques by:
    # 4a) Visualization (tsne)
    # 4b) Evaluation metrics


###########################################################################################
# Libraries
###########################################################################################
import reComBat
from scipy.stats import mannwhitneyu
from pathlib import Path


# Additional fucntions
from evaluationMetrics import *
from batchCorrectionMethods import *
from visualization import *
from dataloading import *



###########################################################################################
# Selections - paths etc.
###########################################################################################

output = "output_TEST"
sc._settings.ScanpyConfig.figdir = Path(output)
Path(os.path.join(os.getcwd(), output)).mkdir(parents=True, exist_ok=True)

batch_field = 'gse'
obs_eval    = 'ZeroHop'

data_path         = os.path.join(os.getcwd(),'data/')
data_filename     = 'Data_from_CELs_R_exported_RMAnormalized.txt'#Expression data, preprocessed by RMA normalization
metadata_filename = 'metadata_GPL84_affymetrix_PA_array.csv'#All metadata anotations regarding growth conditions and PA strain
media_filename    = 'PA_culture_conditions.csv'# Categorization of the relevant growth media

excl2SampleBatches = True


# Possible Evaluation Metrics:
useDRS            = True
useClusterPurity  = True
useCrossDistance  = True
useMinSep         = True
useLDA            = True
useShannonEntropy = True
useLR             = True


# Get data
adata, metadata, data,metadata_cat = getArrayData(data_path,excl2SampleBatches,data_filename,metadata_filename)


# Create overview of all ZeroHops
df_meta = getZHoverview(metadata_cat, obs_eval)


# Check if there are some large ZeroHops and subdivide these based on the refined media definintions
metadata, df_meta = checkAndRefineZeroHops(data_path, metadata, metadata_cat, df_meta, obs_eval,media_filename)



# Uncorrected data
data_raw  = data.copy()
adata_raw = an.AnnData(X=data_raw,obs=metadata)
name = 'raw_raw'
plot_new(adata_raw,output, name)



# Batch correction methods as presented in paper:

# Standardization
data_standardised = ((data.T-data.T.mean())/data.T.std()).T
adata_standardised= an.AnnData(X=data_standardised,obs=metadata)
sc.pp.neighbors(adata_standardised, n_neighbors=10, n_pcs=40)
sc.tl.tsne(adata_standardised)
name = 'Standardised'
plot_new(adata_standardised,output, name)


# Marker gene elimination
data_throw_out_marker = throw_out_marker_genes(data_standardised,metadata,n_throw_out=8)
adata_throw_out_marker= an.AnnData(X=data_throw_out_marker,obs=metadata)
name = 'elMG'
plot_new(adata_throw_out_marker,output, name)


# PC elimination
data_PCel  = throw_out_pca_zerohops(data_standardised,metadata,df_meta, obs = 'ZeroHop')
adata_PCel = an.AnnData(X=data_PCel,obs=metadata)
df_PCel    = pd.DataFrame(data_PCel,columns=data_standardised.columns,index =data_standardised.index)
name       = 'elPC'
plot_new(adata_PCel,output, name)



# reComBat
reg          = 1e-10
model        = reComBat.reComBat(model='ridge', parametric=True, config={'alpha': reg})
data_combat  = model.fit_transform(data_standardised, metadata[batch_field], X=metadata.drop(batch_field, axis=1))
adata_combat = an.AnnData(X=data_combat, obs=metadata)
name         = 'reComBat \n \u03BB\u2081=0, \u03BB\u2082='+str(reg)
plot_new(adata_combat,output, name)


# Combine all
all_adata = [adata_raw,adata_standardised,adata_throw_out_marker,adata_PCel,adata_combat]
all_df    = [data_raw,data_standardised, data_throw_out_marker, df_PCel,data_combat]
all_meta  = [metadata,metadata, metadata, metadata,metadata]
all_names = ['Raw', 'Standardized', 'Eliminate\nMarker Genes','Eliminate PCs',
             'reComBat \n \u03BB\u2081=0, \u03BB\u2082='+str(reg)]




# Crossdistance
if useCrossDistance:

    # Get distance of Zero-hop clusters
    data_CRO_ZH = []
    data_CRO_batch = []
    for cor_meth in range(0, len(all_names)):
        mean_dists_max_raw, std_dists_max_raw, mean_dists_mean_raw, std_dists_mean_raw, mean_dists_med_raw, \
        std_dists_med_raw       = calc_MeanZeroHopDist(all_df[cor_meth], all_meta[cor_meth],obs_eval)
        data_CRO_ZH.append(mean_dists_med_raw)
        mean_dists_max_raw, std_dists_max_raw, mean_dists_mean_raw, std_dists_mean_raw, mean_dists_med_raw, \
        std_dists_med_raw       = calc_MeanZeroHopDist(all_df[cor_meth], all_meta[cor_meth],batch_field)
        data_CRO_batch.append(mean_dists_med_raw)

    # Plot
    label_a = 'Zero-Hop'
    label_b = 'Batch'
    makeBoxPlot(data_CRO_ZH,data_CRO_batch, all_names, label_a,label_b, output, ylabel='Median Pairwise Distance', title='')



# Cluster purity
if useClusterPurity:

    n_cluster = len(np.unique(metadata[obs_eval]))
    data_aGi = []
    data_bGi = []
    for cor_meth in range(0, len(all_names)):
        sc_raw_gini   = cluster_impurities(all_adata[cor_meth],datafield=obs_eval,n_clusters=n_cluster,measure='gini')
        data_aGi.append(sc_raw_gini)

        sc_raw_purity = cluster_impurities(all_adata[cor_meth], datafield=obs_eval, n_clusters=n_cluster, measure='purity')
        data_bGi.append(sc_raw_purity)

    label_a = 'Gini impurity'
    label_b = 'Absolute Cluster Purity'
    makeBoxPlot(data_aGi, data_bGi, all_names, label_a, label_b, output, ylabel='Cluster (Im)Purity',
                title='Cluster (Im)Purity')


#DRS
if useDRS:
    # DRS per cluster:
    data_aDRS = []
    data_bDRS = []
    for cor_meth in range(0, len(all_names)):

        drs_raw_exp, drs_raw_log = DRSperZH(all_adata[cor_meth])
        data_aDRS.append(drs_raw_log)
        data_bDRS.append(drs_raw_exp)
    label_aDRS = 'DRS Log'
    label_bDRS = 'DRS Exp'
    makeBoxPlot(data_aDRS, data_bDRS, all_names, label_aDRS, label_bDRS, output, ylabel='DRS',
                title='DRS per cluster')


# LDA
if useLDA:

    data_aLDA = []
    data_bLDA = []
    for cor_meth in range(0, len(all_names)):
        sc_raw = LDA_score(all_adata[cor_meth],datafield=obs_eval)
        data_aLDA.append(sc_raw)
        # data_a_std.append(std_LDA)

        sc_rawb= LDA_score(all_adata[cor_meth], datafield='gse')
        data_bLDA.append(sc_rawb)

    label_a = 'Zero-Hop'
    label_b = 'Batch'
    makeBoxPlot(data_aLDA, data_bLDA, all_names, label_a, label_b,output, ylabel='LDA Score',
                title='LDA Score')
    u_statistic, p_value = mannwhitneyu(data_aLDA[2], data_aLDA[3])


# Shannon Entropy
if useShannonEntropy:
    n = 14 # number of nearest neighbours to be included
    data_aSHA = []
    data_bSHA = []
    for cor_meth in range(0, len(all_names)):
        all_entr_zero_raw, all_entr_batch_raw = getnormShannonEntropy(all_df[cor_meth],all_meta[cor_meth],n=n, obs = obs_eval)
        data_aSHA.append(all_entr_zero_raw)
        data_bSHA.append(all_entr_batch_raw)
    label_a = 'Zero-Hop'
    label_b = 'Batch'
    makeBoxPlot(data_aSHA, data_bSHA, all_names, label_a, label_b, output, ylabel='Shannon Entropy', title='')

    from scipy.stats import mannwhitneyu
    u_statistic, p_value = mannwhitneyu(data_aSHA[3], data_aSHA[4])


# Minimum Cluster Separation
if useMinSep:
    data_aMIN = []
    for cor_meth in range(0, len(all_names)):
        sc_raw = min_speration_n_cluster(all_adata[cor_meth])
        data_aMIN.append(sc_raw)

    makeBoxPlotSingle(data_aMIN, all_names, 'Zero-Hop', output, ylabel='Minimum Speration Number', title='')



if useLR:
    all_names_LR = ['Raw',
 'Standardized',
 'Eliminate\nMarker GenesNEW',
 'Eliminate PCs',
 'reComBat l2(1e-9)',
 'reComBat EL(l11e-2 l21e-9)',
 'reComBatLASSO(l0.01)',
]

    # Calculate LR and RF prediction of batch and metadata subset
    data_bal_accs_batch = []
    data_bal_accs_zero = []
    data_f1_batch = []
    data_f1_zero = []
    for cor_meth in range(0, len(all_names)):

        bal_accs_batch_LR_raw, f1s_batch_LR_raw, \
        bal_accs_zeroH_LR_raw, f1s_zeroH_LR_raw = calc_LRtoPredBatch(all_df[cor_meth],all_meta[cor_meth], 'LogisticRegression',
                                                                     obs_batch = 'gse', obs_meta = obs_eval)

        # Save results
        save_LRres('LR', all_names[cor_meth], bal_accs_batch_LR_raw, f1s_batch_LR_raw,
                   bal_accs_zeroH_LR_raw, f1s_zeroH_LR_raw, output)

        data_bal_accs_batch.append(bal_accs_batch_LR_raw)
        data_bal_accs_zero.append(bal_accs_zeroH_LR_raw)
        data_f1_batch.append(f1s_batch_LR_raw)
        data_f1_zero.append(f1s_zeroH_LR_raw)



    # Plot: f1s and balanced ACC
    label_a = 'Zero-Hop'
    label_b = 'Batch'
    makeBoxPlot(data_bal_accs_zero,data_bal_accs_batch,  all_names, label_a, label_b, output, output, ylabel='Balanced Accuracy',
                title='')
    makeBoxPlot(data_f1_zero,data_f1_batch, all_names, label_a, label_b, output, output, ylabel='F1 Score', title='')