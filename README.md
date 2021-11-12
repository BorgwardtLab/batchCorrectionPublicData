# batchCorrectionPublicData
Summary of the code published in 'reComBat: Batch effect removal in large-scale, multi-source omics data integration'.

The relevant data associated with this code is provided as a .zip file and needs to be extracted into the 'data' folder.

This code is executed by running harmonizedData.py where optional changes referring the specific batch correction methods, evaluation metrics and output folders are defined.

First the data is loaded. It comprises >1000 micro array gene expressen samples extracted from the GEO database in October 2020 as indicated by the relevant GSE and GSM identifiers. All data was preprocessed using RMA normalization.

The data annotation (referred to as 'metadata') is categorized to reflect the specific PA strain, and culture conditions (temperature, growth medium, culture geometry, antibiotic treatment, growth phase) and each sample was assigned to one of 39 unique metadata subsets (ZeroHops). Only ZeroHops comprising at least 2 batches (GSEs) of at least two samples (GSMs) are kept. 


It will create overview figures of the tsne representations of the data using the following (optional) batch correction methods: 
1. Uncorrected data 
2. Standardized data (Z-scoreing to mean zero and unit variance was applied)
3. Marker gene elimination for each of the ZeroHop Clusters (default top 8 marker genes)
4. Principal component elimination for each of the ZeroHops 
5. reComBat

Finally a range of evaluation metrics are calculated to quantify batch correction efficacy. These include: 
1. LDA score
2. DRS score
3. Cluster purity and Gini impurity
4. Minmum Cluster Separation number
5. Cluster Cross-distance
6. Logistic Regression (or other classifier) classification performance of batch and ZeroHop.




