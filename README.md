# batchCorrectionPublicData
Summary of the code published in 'reComBat: Batch effect removal in large-scale, multi-source omics data integration'.


## Installation of packages
All packages have been compiled in the provided requirements.txt file. 
Simply use this file to install all pachages via "pip install requirements.txt".


## Run example
We provide all data and code to reproduce Figures 1, 2 and S1-S10. This code is executed by running "harmonizedData.py" where options referring the specific batch correction methods, evaluation metrics and output folders are defined. This script comprises three main parts: 
1. Data loading and metadata preprocessing
2. Batch correction
3. Evaluation of the batch correction methods

### Data loading and preparation
The relevant data associated with this code is provided as a .zip file and needs to be extracted into the 'data' folder. It comprises >1000 micro array gene expressen samples extracted from the GEO database in October 2020 as indicated by the relevant GSE and GSM identifiers. All data was preprocessed using RMA normalization. 

The data annotation (referred to as "metadata") is categorized to reflect the specific PA strain, and culture conditions (temperature, growth medium, culture geometry, antibiotic treatment, growth phase) and each sample is assigned to one of 39 unique metadata subsets (ZeroHops). Only ZeroHops comprising at least 2 batches (GSEs) of at least two samples (GSMs) are kept. 

### Batch correction
We provide code for the following (optional) batch correction methods: 
1. Uncorrected data 
2. Standardized data (Z-scoreing to mean zero and unit variance was applied)
3. Marker gene elimination for each of the ZeroHop Clusters (default top 8 marker genes)
4. Principal component elimination for each of the ZeroHops 
5. reComBat
For each of the relevant methods overview fiures showing t-SNE embeddings of the corrected adata colored by all metadata categories are created to provide a visual inspection of the batch correction success. 

### Evaluation of the batch correction methods
We provide a range of custom evaluation metrics probing different aspects of a successful batch corrected dataframe. These include:

1. LDA score
2. DRS score
3. Cluster purity and Gini impurity
4. Minmum Cluster Separation number
5. Cluster Cross-distance
6. Logistic Regression (or other classifier) classification performance of batch and ZeroHop.




