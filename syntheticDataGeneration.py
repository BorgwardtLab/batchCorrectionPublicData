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
import torch
import torch.nn as nn
import time
import argparse
import scanpy as sc
import anndata as an
from pathlib import Path
from reComBat import reComBat


# Additional fucntions
from evaluationMetrics import *
from batchCorrectionMethods import *
from visualization import *


class SynthicDataGenerator():
    '''
    This is a class to create an expression dataset with different experimental designs,
    additive and multiplicative batch effects.

    Public methods:
    - create_data(n_samples,kwargs): creates a dataset
    - create_more_data(n_samples,batch_list,zero_hops_list): creates a test dataset
                            from defined batches and zero hops
    - nonlinear_transform(data): randomly nonlinearly transforms the data
                    creates non-linear interaction between the genes

    Quasi Private methods (not actually private, but used only in class):
    Many; serve to create the design and batches matrices; stored as class attributes
        Potentially not memory friendly, could be optimised in the future

    Class methods:
    - estimate_from_data(data): NOT IMPLEMENTED!
                    estimates the batch and zero-hop parameters for data
    '''
    def __init__(self,
                 n_features,
                 n_design_features,
                 n_categories,
                 n_zero_hops,
                 mean_samples_per_batch,
                 relative_batch_beta=10.0,
                 one_seed_per_sample=False,
                 is_singular=True,
                 sparsity_percentage_additive=0.2,
                 mu_additive=0.0,
                 sigma_additive=1.0,
                 sparsity_percentage_multiplicative=0.8,
                 mu_multiplicative=0.0,
                 sigma_multiplicative=1.0,
                 poisson=False,
                 wanted_effects=True):
        self.n_features=n_features
        self.n_design_features=n_design_features
        self.n_categories=n_categories
        self.n_zero_hops=n_zero_hops
        self.mean_samples_per_batch=mean_samples_per_batch
        self.relative_batch_beta=relative_batch_beta
        self.one_seed_per_sample=one_seed_per_sample
        self.is_singular=is_singular
        self.sparsity_percentage_additive=sparsity_percentage_additive
        self.mu_additive=mu_additive
        self.sigma_additive=sigma_additive
        self.sparsity_percentage_multiplicative=sparsity_percentage_multiplicative
        self.mu_multiplicative=mu_multiplicative
        self.sigma_multiplicative=sigma_multiplicative
        self.poisson=poisson
        self.wanted_effects=wanted_effects

        self.model = None

    def create_data(self,n_samples,X_df=None,b_df=None,batch_indicator=None,additive_effect=None,multplicative_effect=None):
        '''
        Known issues:
            - Sometimes X is not singular
            This is likely due to not sampling all categories for each feature.
        '''
        if not self.one_seed_per_sample:
            self._create_base_gauss(1)
        else:
            self._create_base_gauss(n_samples)

        if X_df is None:
            self._create_X(n_samples)
        else:
            self.X_df = X_df
        if b_df is None:
            self._create_batches(n_samples)
        else:
            self.b_df = b_df
            assert batch_indicator is not None
            self.batch_indicator = batch_indicator

        if additive_effect is None:
            self._create_batch_additive()
        else:
            self.additive_effect = additive_effect
        if multplicative_effect is None:
            self._create_batch_multiplicative()
        else:
            self.multplicative_effect = multplicative_effect

        self.beta_x_ = np.random.randn(self.n_design_features*(self.n_categories-1),self.n_features)
        self.beta_b_ = self.relative_batch_beta*np.random.randn(np.max(self.batch_indicator),self.n_features)

        # add small random shift for each gene
        mu, sigma = 1, 0.001
        randShift = np.random.normal(mu, sigma,
                                     self._one_hot_X(self.X_df.drop('ZeroHop',axis=1)).to_numpy().dot(self.beta_x_).shape)

        # Get shifted genexpr.
        Y = self.seed +\
            self._one_hot_X(self.X_df.drop('ZeroHop',axis=1)).to_numpy().dot(self.beta_x_)*randShift +\
            pd.get_dummies(self.b_df,drop_first=True).to_numpy().dot(self.beta_b_) +\
            self.additive_effect +\
            self.multplicative_effect

        # Get GT
        if self.wanted_effects:
            Y_ground_truth = self.seed +\
                             self._one_hot_X(self.X_df.drop('ZeroHop',axis=1)).to_numpy().dot(self.beta_x_)*randShift
        else:
            if self.one_seed_per_sample:
                Y_ground_truth = self.seed
            else:
                Y_ground_truth = np.tile(self.seed,(n_samples,1))

        return {'Data':Y,'Data_GT':Y_ground_truth,'Design':self.X_df,'Batches':self.b_df}

    def create_more_data(self,n_samples,batch_list,zero_hops_list):
        if self.one_seed_per_sample:
            self._test_create_base_gauss(n_samples)
        else:
            self.test_seed = self.seed

        self._create_additional_X(n_samples,zero_hops_list)
        self._sample_more_from_same_batches(n_samples,batch_list)
        self._create_more_batch_additive()
        self._create_more_batch_multiplicative()

        Y = self.test_seed +\
            self._one_hot_X(self.test_X_df.drop('ZeroHop',axis=1)).to_numpy().dot(self.beta_x_) +\
            self._one_hot_test_b(self.test_b_df).to_numpy().dot(self.beta_b_) +\
            self.test_additive_effect +\
            self.test_multplicative_effect


        if self.wanted_effects:
            Y_ground_truth = self.test_seed +\
                             self._one_hot_X(self.test_X_df.drop('ZeroHop',axis=1)).to_numpy().dot(self.beta_x_)
        else:
            if self.one_seed_per_sample:
                Y_ground_truth = self.test_seed
            else:
                Y_ground_truth = np.tile(self.test_seed,(n_samples,1))

        return {'Data':Y,'Data_GT':Y_ground_truth,'Design':self.test_X_df,'Batches':self.test_b_df}

    def nonlinear_transform(self,data,n_features):
        starting_features = data.shape[1]

        if self.model==None:
            self.model = nn.Sequential(nn.Linear(starting_features,np.min([2*starting_features,n_features])), nn.ReLU(),
                                   nn.Linear(np.min([2*starting_features,n_features]),np.min([4*starting_features,n_features])), nn.ReLU(),
                                   nn.Linear(np.min([4*starting_features,n_features]),n_features)
                                   )
        return self.model.forward(torch.from_numpy(data)).detach().numpy()

    def _create_X(self,n_samples):
        design_features = [f'Feature_{i}' for i in range(self.n_design_features)]
        categories = [f'Cat_{i}' for i in range(self.n_categories)]

        self.embedding_dummy = pd.DataFrame({d:categories for d in design_features})

        design_matrix = []
        mean_samples_per_zh = int(np.ceil(n_samples/self.n_zero_hops))

        i = 0
        zero_hop_indicator = []
        if self.is_singular:
            while len(zero_hop_indicator) < n_samples:
                design_row = np.random.choice(categories,size=(1,self.n_design_features-1))
                samples_in_zh = mean_samples_per_zh#np.random.choice([mean_samples_per_zh-1,mean_samples_per_zh,mean_samples_per_zh+1],p=3*[1./3])
                design_matrix.append(list(np.tile(design_row,(samples_in_zh,1))))
                zero_hop_indicator.extend(samples_in_zh*[i])
                i += 1
            design_matrix = np.vstack(design_matrix)[:n_samples]
            zero_hop_indicator = np.array(zero_hop_indicator)[:n_samples].reshape(-1,1)
            last_col = self._permute_categories(categories,design_matrix[:,0])
            design_matrix = np.hstack([design_matrix,last_col,zero_hop_indicator])
            np.random.shuffle(design_matrix)
            self.X_df = pd.DataFrame(data=design_matrix,columns=design_features+['ZeroHop'])
            X = pd.get_dummies(self.X_df,drop_first=True).to_numpy()
            if X.shape[0] <= X.shape[1]:
                assert np.linalg.det(X.dot(X.T)) == 0
            elif X.shape[0] > X.shape[1]:
                assert np.linalg.det(X.T.dot(X)) == 0
        else:
            while len(zero_hop_indicator) < n_samples:
                design_row = np.random.choice(categories,size=(1,self.n_design_features))
                samples_in_zh = np.random.choice([mean_samples_per_zh-1,mean_samples_per_zh,mean_samples_per_zh+1],p=3*[1./3])
                design_matrix.append(list(np.tile(design_row,(samples_in_zh,1))))
                zero_hop_indicator.extend(samples_in_zh*[i])
                i += 1
            design_matrix = np.vstack(design_matrix)[:n_samples]
            zero_hop_indicator = np.array(zero_hop_indicator)[:n_samples].reshape(-1,1)
            design_matrix = np.hstack([design_matrix,zero_hop_indicator])
            np.random.shuffle(design_matrix)
            self.X_df = pd.DataFrame(data=design_matrix,columns=design_features+['ZeroHop'])

    def _create_additional_X(self,n_samples,zero_hops_list):
        mean_samples_per_zh = int(np.ceil(n_samples/len(zero_hops_list)))
        zero_hops_designs = self.X_df.loc[self.X_df.ZeroHop.astype(int).isin(zero_hops_list)].drop_duplicates().reset_index(drop=True)
        design_matrix = []

        i = 0
        total_samples = 0
        while total_samples < n_samples:
            design_row = zero_hops_designs.loc[i].values
            samples_in_zh = np.random.choice([mean_samples_per_zh-1,mean_samples_per_zh,mean_samples_per_zh+1],p=3*[1./3])
            design_matrix.append(list(np.tile(design_row,(samples_in_zh,1))))
            i += 1
            total_samples += samples_in_zh
            if i >= len(zero_hops_list):
                i = np.random.randint(low=0,high=len(zero_hops_list))
        design_matrix = np.vstack(design_matrix)[:n_samples]
        np.random.shuffle(design_matrix)
        self.test_X_df = pd.DataFrame(data=design_matrix,columns=self.X_df.columns)
        if self.is_singular:
            X = pd.get_dummies(self.test_X_df,drop_first=True).to_numpy()
            if X.shape[0] <= X.shape[1]:
                assert np.linalg.det(X.dot(X.T)) == 0
            elif X.shape[0] > X.shape[1]:
                assert np.linalg.det(X.T.dot(X)) == 0

    def _one_hot_X(self,X):
        tmp = pd.concat([X,self.embedding_dummy],axis=0,ignore_index=True)
        tmp = pd.get_dummies(tmp,drop_first=True)
        return tmp.loc[:X.shape[0]-1]

    def _one_hot_test_b(self,test_b):
        batches = self.b_df.drop_duplicates()
        tmp = pd.concat([test_b,batches],axis=0,ignore_index=True)
        tmp = pd.get_dummies(tmp,drop_first=True)
        return tmp.loc[:test_b.shape[0]-1]

    def _create_base_gauss(self,n_samples):
        self.seed = np.random.randn(n_samples,self.n_features)

    def _test_create_base_gauss(self,n_samples):
        self.test_seed = np.random.randn(n_samples,self.n_features)

    def _create_batch_additive(self):
        n_batches = np.max(self.batch_indicator)+1
        self.additive_effect = np.zeros((len(self.batch_indicator),self.n_features))
        for b in range(n_batches):
            batch_mask = self.batch_indicator == b
            samples_per_batch = batch_mask.sum()
            sparsity_mask = np.random.choice([0,1],size=(1,self.n_features),p=[1-self.sparsity_percentage_additive,self.sparsity_percentage_additive])
            self.additive_effect[batch_mask] =  self.mu_additive+self.sigma_additive*sparsity_mask*np.random.randn(*sparsity_mask.shape)

    def _create_more_batch_additive(self):
        n_batches = np.max(self.test_batch_indicator)+1
        self.test_additive_effect = np.zeros((len(self.test_batch_indicator),self.n_features))
        for b in np.unique(self.test_batch_indicator):
            effect = self.additive_effect[self.batch_indicator==b][0]
            self.test_additive_effect[self.test_batch_indicator==b] = effect

    def _create_batch_multiplicative(self):
        n_batches = len(np.unique(self.batch_indicator))
        sparsity_mask = np.random.choice([0,1],size=(n_batches,self.n_features),p=[1-self.sparsity_percentage_multiplicative,self.sparsity_percentage_multiplicative])
        self.multplicative_effect_kernel =  self.mu_multiplicative+self.sigma_multiplicative*sparsity_mask*np.random.randn(n_batches,self.n_features)
        self.multplicative_effect = np.zeros((len(self.batch_indicator),self.n_features))
        for b in range(n_batches):
            batch_mask = self.batch_indicator == b
            samples_per_batch = np.array(batch_mask).sum()
            self.multplicative_effect[batch_mask] =  np.random.randn(samples_per_batch,self.n_features)*self.multplicative_effect_kernel[b]

    def _create_more_batch_multiplicative(self):
        n_batches = len(np.unique(self.test_batch_indicator))
        self.test_multplicative_effect = np.zeros((len(self.test_batch_indicator),self.n_features))
        for b in np.unique(self.test_batch_indicator):
            batch_mask = self.test_batch_indicator == b
            samples_per_batch = np.array(batch_mask).sum()
            self.test_multplicative_effect[batch_mask] = np.random.randn(samples_per_batch,self.n_features)*self.multplicative_effect_kernel[b]

    def _create_batches(self,n_samples):
        batches = []
        batch_indicator = []
        i = 0
        while len(batches) < n_samples:
            if self.poisson:
                batch = np.random.poisson(lam=self.mean_samples_per_batch)
            else:
                batch = self.mean_samples_per_batch#np.random.choice([self.mean_samples_per_batch-1,self.mean_samples_per_batch,self.mean_samples_per_batch+1],p=3*[1./3])
            batches.extend(batch*[f'Batch_{i}'])
            batch_indicator.extend(batch*[i])
            i += 1
        self.b_df = pd.DataFrame(data=batches[:n_samples],columns=['Batch'])
        self.batch_indicator = np.array(batch_indicator[:n_samples])

    def _sample_more_from_same_batches(self,n_samples,batch_list):
        mean_samples_per_batch = int(np.ceil(n_samples/len(batch_list)))
        batches = []
        batch_indicator = []
        i = 0
        while len(batches) < n_samples:
            if self.poisson:
                batch = np.random.poisson(lam=mean_samples_per_batch)
            else:
                batch = np.random.choice([mean_samples_per_batch-1,mean_samples_per_batch,mean_samples_per_batch+1],p=3*[1./3])
            batches.extend(batch*[batch_list[i]])
            batch_indicator.extend(batch*[int(batch_list[i].split('_')[1])])
            i += 1
            if i >= len(batch_list):
                i = np.random.randint(low=0,high=len(batch_list))
        self.test_b_df = pd.DataFrame(data=batches[:n_samples],columns=['Batch'])
        self.test_batch_indicator = np.array(batch_indicator[:n_samples])

    @staticmethod
    def _permute_categories(categories,col):
        last_col = np.zeros_like(col)
        n_categories = len(categories)
        for i,c in enumerate(col):
            cat_idx = int(c.split('_')[1])
            last_col[i] = categories[(cat_idx+1) % n_categories]
        return last_col.reshape(-1,1)

    @classmethod
    def estimate_from_data(cls,data):
        '''
        Not sure what the best way of doing this is.
        Maybe fit a reComBat model to real data and analyse the fitted parameters.
        I'll leave this for now in the interest of time.
        '''
        pass
def getZHoverview(metadata_cat,obs_eval,batch_field):
    'Return a summary of the metadata categories for each ZH'

    n_samples = []
    n_batch = []
    n_cluster = len(np.unique(metadata_cat[obs_eval]))
    for i in range(0, n_cluster):
        zh = np.unique(metadata_cat[obs_eval])[i]#str(i)
        metasub = metadata_cat[metadata_cat[obs_eval] == zh]
        n_batch.append(len(np.unique(metasub[batch_field])))
        n_samples.append(len(metasub))

        if i == 0:
            df_meta = metasub.iloc[0, :].to_frame().transpose()
        else:
            df_meta = pd.concat([df_meta, metasub.iloc[0, :].to_frame().transpose()], axis=0)

    df_meta.set_index(obs_eval, drop=True, inplace=True)
    df_meta.drop(batch_field, axis=1, inplace=True)
    df_meta['n_samples'] = n_samples
    df_meta['n_gse'] = n_batch
    return df_meta
def getAllVars(run, res_all, n_samples, n_features,n_design_features,
                                                    n_categories,n_zero_hops,
                                                    mean_samples_per_batch,
                                                    relative_batch_beta,
                                                    sparsity_percentage_additive,
                                                    mu_additive,sigma_additive,
                                                    sparsity_percentage_multiplicative,
                                                    mu_multiplicative,sigma_multiplicative):

    res_all.loc[run, 'repeat'] = run
    res_all.loc[run, 'n_samples']= n_samples
    res_all.loc[run, 'n_features'] = n_features
    res_all.loc[run, 'n_design_features'] = n_design_features
    res_all.loc[run, 'n_categories'] = n_categories
    res_all.loc[run, 'n_zero_hops'] = n_zero_hops
    res_all.loc[run, 'mean_samples_per_batch'] = mean_samples_per_batch
    res_all.loc[run, 'relative_batch_beta'] = relative_batch_beta
    res_all.loc[run, 'sparsity_percentage_additive'] = sparsity_percentage_additive
    res_all.loc[run, 'mu_additive'] = mu_additive
    res_all.loc[run, 'sigma_additive'] = sigma_additive
    res_all.loc[run, 'sparsity_percentage_multiplicative'] = sparsity_percentage_multiplicative
    res_all.loc[run, 'mu_multiplicative'] = mu_multiplicative
    res_all.loc[run, 'sigma_multiplicative'] = sigma_multiplicative
    return res_all





###########################################################################################
# Selections - paths etc.
###########################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--uselocal', type=int, default=1, help='use local machine or cluster')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_features', type=int, default=5000)
    parser.add_argument('--n_design_features', type=int, default=5)
    parser.add_argument('--n_categories', type=int, default=3)
    parser.add_argument('--n_zero_hops', type=int, default=10)
    parser.add_argument('--mean_samples_per_batch', type=int, default=10)
    parser.add_argument('--relative_batch_beta', type=float, default=10.0)
    parser.add_argument('--sparsity_percentage_additive', type=float, default=0.2)
    parser.add_argument('--mu_additive', type=float, default=2.0)
    parser.add_argument('--sparsity_percentage_multiplicative', type=float, default=0.8)
    parser.add_argument('--mu_multiplicative', type=float, default=0.0)
    parser.add_argument('--sigma_multiplicative', type=float, default=1.0)
    defaults = [1000, 5000, 5, 3, 10, 3,10.0,0.2,2.0,1.0,0.8,0.0,1.0]
    args     = parser.parse_args()


    n_samples                    = args.n_samples
    n_features                   = args.n_features
    n_design_features            = args.n_design_features
    n_categories                 = args.n_categories
    n_zero_hops                  = args.n_zero_hops
    mean_samples_per_batch       = args.mean_samples_per_batch
    relative_batch_beta          = args.relative_batch_beta
    sparsity_percentage_additive = args.sparsity_percentage_additive
    mu_additive                        = args.mu_additive
    sparsity_percentage_multiplicative = args.sparsity_percentage_multiplicative
    mu_multiplicative                  = args.mu_multiplicative
    sigma_multiplicative               = args.sigma_multiplicative
    sigma_additive                     = args.sigma_multiplicative

    # Check which variable is not the default:
    all_vars  = [n_samples,n_features,n_design_features,n_categories,n_zero_hops,mean_samples_per_batch,
                 relative_batch_beta,sparsity_percentage_additive,mu_additive,sigma_additive,
                 sparsity_percentage_multiplicative,mu_multiplicative,sigma_multiplicative]
    variables = ['n_samples','n_features','n_design_features','n_categories','n_zero_hops','mean_samples_per_batch',
                 'relative_batch_beta','sparsity_percentage_additive','mu_additive','sigma_additive',
                 'sparsity_percentage_multiplicative','mu_multiplicative','sigma_multiplicative']
    name_run = ''
    for v in range(0,len(all_vars)):
        if all_vars[v]!=defaults[v]:
            name_run+=variables[v]+str(all_vars[v])+'_'
    if len(name_run)==0:
        name_run = 'default_'




    # Define this experiment
    doPlots  = False
    path     = os.getcwd()
    useGPU   = False

    n_repeats = 10
    ids = ['n_design', 'n_genes', 'n_gse_sim', 'shift_std_batch', 'shift_std_batch_to_meta', 'n_ind_conditions']

    # Define batch and wanted biological variation
    batch_field = 'Batch'
    obs_eval    = 'ZeroHop'

    # Select correction models
    models = ['GroundTruth','reComBat']
    # Options: 'GroundTruth', 'Raw', 'Standardised','elMG','elPC', 'reComBat', 'harmony', 'scGen','scGen_ZH'

    # reComBat regularization strengths
    regs    = [0,1e-1, 1e-3, 1e-6, 1e-9, 1e-12]
    metrics = ['ShEntr_ZH', 'ShEntr_Batch', 'LDA_ZH', 'LDA_Batch', 'Gini', 'Purity', 'runTime']

    # output
    output   =  os.path.join(path,"output_Synthetic_Systematic_LINreComBATHP")
    sc._settings.ScanpyConfig.figdir = Path(output)
    Path(output).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output,'Results')).mkdir(parents=True, exist_ok=True)

    # Check if run already exists
    res_all = pd.DataFrame(columns = ['n_samples','n_features','n_design_features','n_categories','n_zero_hops',
                                      'mean_samples_per_batch','relative_batch_beta','sparsity_percentage_additive',
                                      'mu_additive','sigma_additive','sparsity_percentage_multiplicative',
                                      'mu_multiplicative','sigma_multiplicative']
                                     +['ShEntr_'+ m+'_ZH' for m in models]+['ShEntr_'+ m+'_Batch' for m in models]
                                          + ['LDA_' + m + '_ZH' for m in models] + ['LDA_' + m + '_Batch' for m in models]
                                          + ['ACC_' + m + '_ZH' for m in models] + ['ACC_' + m + '_Batch' for m in models]
                                          +['Gini_'+m for m in models] + ['Purity_'+m  for m in models]
                                          +['runTime_'+m  for m in models])


    # Get data
    for run in range(0, n_repeats):

        # Get the data
        gotData = 0
        fail_count = 0
        while gotData == 0 & fail_count <5:
            try:
                generator = SynthicDataGenerator(
                    n_features=n_features,
                    n_design_features=n_design_features,
                    n_categories=n_categories,
                    n_zero_hops=n_zero_hops,
                    mean_samples_per_batch=mean_samples_per_batch,
                    relative_batch_beta=relative_batch_beta,
                    one_seed_per_sample=False,
                    is_singular=True,
                    sparsity_percentage_additive=sparsity_percentage_additive,
                    mu_additive=mu_additive,
                    sigma_additive=sigma_additive,
                    sparsity_percentage_multiplicative=sparsity_percentage_multiplicative,
                    mu_multiplicative=mu_multiplicative,
                    sigma_multiplicative=sigma_multiplicative,
                    poisson=False,
                    wanted_effects=True)

                # Create train set
                train_dict = generator.create_data(n_samples)
                gotData = 1
            except:
                print('Data generation failed')
                fail_count += 1


        # Uncorrected data
        batchdata   = train_dict['Batches']['Batch'].str.split('_', expand=True)[1].to_frame().rename(columns = {1:batch_field})
        metadata    = pd.concat([train_dict['Design'], batchdata],axis = 1)
        data_BE     = pd.DataFrame(train_dict['Data'], columns = np.arange(train_dict['Data'].shape[1]),
                                   index = metadata.index)
        data_GT     = pd.DataFrame(train_dict['Data_GT'], columns = np.arange(train_dict['Data'].shape[1]),
                                   index = metadata.index)
        df_meta_new = getZHoverview(metadata, obs_eval,batch_field)
        df_meta_new.to_csv(os.path.join(output, 'Results', name_run+ str(k)+'_df_meta.csv'))
        metadata.to_csv(os.path.join(output, 'Results', name_run + str(k)+ '_metadata.csv'))
        to_colour_by = [batch_field,obs_eval]

        # Check that design matrix is singular
        if np.any(df_meta_new.n_gse < 2):
            print('Single batch identified!')


        # Start correction of batch effects
        res_all = getAllVars(run, res_all, n_samples, n_features,n_design_features,
                                                n_categories,n_zero_hops,
                                                mean_samples_per_batch,
                                                relative_batch_beta,
                                                sparsity_percentage_additive,
                                                mu_additive,sigma_additive,
                                                sparsity_percentage_multiplicative,
                                                mu_multiplicative,sigma_multiplicative)

        all_names = []
        all_adata = []
        all_df = []
        all_meta = []
        if 'GroundTruth' in models:
            name = 'Ground Truth'

            adata_gt = an.AnnData(data_GT, obs=metadata)
            all_names.append(name)
            all_adata.append(adata_gt)
            all_df.append(data_GT)
            all_meta.append(metadata)
            if doPlots:
                plot_newSynthetic(adata_gt, output, name_run+'_'+name, to_colour_by=to_colour_by)

        adata_raw = an.AnnData(X=data_BE, obs=metadata)
        if 'Raw' in models:
            name = 'Raw'
            all_names.append(name)
            all_adata.append(adata_raw)
            all_df.append(data_BE)
            all_meta.append(metadata)
            if doPlots:
                plot_newSynthetic(adata_raw,output, name_run+'_'+name, to_colour_by= to_colour_by)

        # Standardization
        data_standardised = ((data_BE.T - data_BE.T.mean()) / data_BE.T.std()).T
        adata_standardised = an.AnnData(X=data_standardised, obs=metadata)
        if 'Standardised' in models:
            name = 'Standardised'
            all_names.append(name)
            all_adata.append(adata_standardised)
            all_df.append(data_standardised)
            all_meta.append(metadata)
            if doPlots:
                plot_newSynthetic(adata_standardised,output, name_run+'_'+name, to_colour_by= to_colour_by)

        # Marker gene elimination
        if 'elMG' in models:
            data_throw_out_marker = throw_out_marker_genes(data_standardised, metadata, n_throw_out=8,
                                                           batch_field = batch_field)
            adata_throw_out_marker = an.AnnData(X=data_throw_out_marker, obs=metadata)
            name = 'elMG'
            all_names.append('Eliminate\nMarker Genes')
            all_adata.append(adata_throw_out_marker)
            all_df.append(data_throw_out_marker)
            all_meta.append(metadata)
            if doPlots:
                plot_newSynthetic(adata_throw_out_marker,output, name_run+'_'+name, to_colour_by= to_colour_by)

        # PC elimination
        if 'elPC' in models:
            data_PCel = throw_out_pca_zerohops(data_standardised, metadata, df_meta_new, obs='ZeroHop')
            adata_PCel = an.AnnData(X=data_PCel, obs=metadata)
            df_PCel = pd.DataFrame(data_PCel, columns=data_standardised.columns, index=data_standardised.index)
            name = 'elPC'
            all_names.append('Eliminate PCs')
            all_adata.append(adata_PCel)
            all_df.append(data_PCel)
            all_meta.append(metadata)
            if doPlots:
                plot_newSynthetic(adata_PCel,output, name_run+'_'+name, to_colour_by= to_colour_by)

        # reComBat
        if 'reComBat' in models:

            for regs_L1 in regs:
                for regs_L2 in regs:

                    if (regs_L1 == 0):
                        if (regs_L2 == 0):
                            continue
                        else:
                            model = reComBat(model='ridge', parametric=True,
                                         config={'alpha': regs_L2,'max_iter':100})
                    else:
                        if (regs_L2 == 0):
                            model = reComBat(model='Lasso', parametric=True,
                                             config={'alpha': regs_L1, 'max_iter':100})
                        else:
                            model = reComBat(model='elastic_net', parametric=True,
                                             config={'alpha': regs_L2,'l1_ratio':regs_L1,'max_iter':100})
                    try:
                        t = time.time()
                        data_combat = model.fit_transform(data_standardised,
                                                                metadata[batch_field],
                                                                X=metadata.drop([batch_field, obs_eval],
                                                                                      axis=1))
                        elapsed = time.time() - t
                        # data_combat_test = model.transform(data_standardised_test, metadata_test[batch_field],
                        #                                    X=metadata_test.drop([batch_field, 'ZeroHop'], axis=1))

                    except:
                        print('Combat failed!')
                        continue
                    adata_combat = an.AnnData(X=data_combat, obs=metadata)
                    name = 'reComBat \n \u03BB\u2081='+ str(regs_L1) +', \u03BB\u2082='+ str(regs_L2)


                    all_names.append(name)
                    all_adata.append(adata_combat)
                    all_df.append(data_combat)
                    all_meta.append(metadata)
                    res_all.loc[run,'runTime_'+ name] = elapsed
                    if doPlots:
                        plot_newSynthetic(adata_combat,output, name_run+'_'+name, to_colour_by= to_colour_by)


        # Shannon Entropy
        n = 14  # number of nearest neighbours to be included
        for cor_meth in range(0, len(all_names)):
            all_entr_zero_raw, all_entr_batch_raw = getnormShannonEntropy(all_df[cor_meth], all_meta[cor_meth],
                                                                          n=n,
                                                                          obs=obs_eval,
                                                                          batch = batch_field)
            res_all.loc[run, 'ShEntr_'+ all_names[cor_meth] + '_ZH'] = np.median(all_entr_zero_raw)
            res_all.loc[run, 'ShEntr_'+ all_names[cor_meth] + '_Batch'] = np.median(all_entr_batch_raw)


        # LDA
        data_aLDA = []
        data_bLDA = []
        for cor_meth in range(0, len(all_names)):
            sc_raw = LDA_score(all_adata[cor_meth], datafield=obs_eval)
            data_aLDA.append(sc_raw)
            res_all.loc[run, 'LDA_'+ all_names[cor_meth] + '_ZH'] = np.median(sc_raw)

            sc_rawb = LDA_score(all_adata[cor_meth], datafield=batch_field)
            data_bLDA.append(sc_rawb)
            res_all.loc[run, 'LDA_'+all_names[cor_meth] + '_Batch'] = np.median(sc_rawb)

        n_cluster = len(np.unique(metadata[obs_eval]))
        for cor_meth in range(0, len(all_names)):
            sc_raw_gini = cluster_impurities(all_adata[cor_meth], datafield=obs_eval, n_clusters=n_cluster,
                                             measure='gini')

            sc_raw_purity = cluster_impurities(all_adata[cor_meth], datafield=obs_eval, n_clusters=n_cluster,
                                               measure='purity')

            res_all.loc[run, 'Gini_'+all_names[cor_meth] ] = np.median(sc_raw_gini)
            res_all.loc[run, 'Purity_'+all_names[cor_meth] ] = np.median(sc_raw_purity)

        res_all.to_csv(os.path.join(output,'Results',name_run+ '.csv'))







