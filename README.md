# AGEP_TWAS
Source code for AGEP_TWAS
## dataset
This repository contains a set of tools for performing Gene Expression Prediction and conducting Transcriptome-Wide Association Study (TWAS). The key steps in the process include identifying target genes, splitting tissue-specific datasets, landmark gene extraction, model training, and running the PrediXcan prediction. The tools in this repository are designed to help researchers in genetic and transcriptomic analysis.

Directory Structure
bash
复制代码
.
├── dataset/
│   ├── get_target_gene.py
│   └── tissuesplit.py
├── autoencoder_deeplift.py
├── train.py
├── Predict_for_Predixcan.py
├── inference.py
├── gene_summary.csv
Files and Their Functions
dataset/
get_target_gene.py
This script identifies the target genes for each fold in the dataset. It processes the data, filters out the top genes, and then selects the target genes that are shared across both the full gene set and the prediction gene set.

tissuesplit.py
This script is used to split the cattlegtex dataset into different tissue types. It helps organize the data for tissue-specific analysis, which is crucial for TWAS studies where tissue-specific gene expression is a key factor.

Main Directory
autoencoder_deeplift.py
This script applies a non-linear feature extraction method (Autoencoder with DeepLIFT) to extract a specified number of landmark genes. You can customize the number of landmark genes to extract using the input parameters in the script. The extracted genes are important for downstream analysis like TWAS.

train.py
This script is used for training the machine learning model. It handles the training pipeline, including the loading of datasets, training process, and model evaluation.

Predict_for_Predixcan.py
This script provides functionality for performing PrediXcan prediction. The PrediXcan method is based on the MetaXcan framework. For more information about MetaXcan, you can visit MetaXcan. This script is designed to perform predictions using gene expression data and genomic features for association studies.

inference.py
This script provides an interface for users to select their desired landmark genes. These selected landmark genes are then used to complete the gene expression profiles and prepare for further TWAS analysis.

gene_summary.csv
This file contains the results of the TWAS analysis. It includes a summary of the genes considered significant in the study. This file can be used to review the identified genes and their associated statistical metrics.
