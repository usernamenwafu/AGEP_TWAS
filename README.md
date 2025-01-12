# AGEP_TWAS
This repository provides tools for Gene Expression Prediction and conducting Transcriptome-Wide Association Study (TWAS). The main steps in the process include identifying target genes, splitting tissue-specific datasets, landmark gene extraction, model training, and running the PrediXcan prediction. These tools are designed to assist researchers in genetic and transcriptomic analysis.

## Directory Structure
```
. ├── dataset/ <br> │ ├── get_target_gene.py # Identifies target genes for each fold <br> │ └── tissuesplit.py # Splits cattlegtex dataset into tissue types <br> ├── autoencoder_deeplift.py # Extracts landmark genes using Autoencoder + DeepLIFT <br> ├── train.py # Train the model <br> ├── Predict_for_Predixcan.py # PrediXcan prediction based on MetaXcan <br> ├── inference.py # Interface for selecting landmark genes for TWAS <br> ├── gene_summary.csv # TWAS results with significant genes <br>
```
## Files and Their Functions

### `dataset/`

- **`get_target_gene.py`**  
  This script identifies the target genes for each fold in the dataset. It processes the data, filters out the top genes, and selects the target genes that are shared across both the full gene set and the prediction gene set.

- **`tissuesplit.py`**  
  This script splits the `cattlegtex` dataset into different tissue types, helping organize data for tissue-specific analysis, which is crucial for TWAS studies where tissue-specific gene expression is important.

### Main Directory

- **`autoencoder_deeplift.py`**  
  This script applies a non-linear feature extraction method (Autoencoder with DeepLIFT) to extract a specified number of landmark genes. The number of landmark genes can be customized via input parameters.

- **`train.py`**  
  This script is used for training the machine learning model. It includes the training pipeline, dataset loading, training process, and model evaluation.

- **`Predict_for_Predixcan.py`**  
  This script provides functionality for running PrediXcan predictions. PrediXcan is based on the MetaXcan framework, which can be found [here](https://github.com/hakyimlab/MetaXcan).

- **`inference.py`**  
  This script provides an interface for users to select landmark genes. The selected landmark genes are then used to complete gene expression profiles and prepare for further TWAS analysis.

### `gene_summary.csv`

This file contains the results of the TWAS analysis. It includes a summary of the genes considered significant in the study. This file can be used to review the identified genes and their associated statistical metrics.

---
