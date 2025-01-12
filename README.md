# AGEP_TWAS
This repository provides tools for Adaptive Gene Expression Predictor for TWAS(AGEP_TWAS). The main steps in the process include identifying target genes, splitting tissue-specific datasets, landmark gene extraction, model training. 

## Directory Structure
```
.
├── dataset/
│   ├── get_target_gene.py
│   └── tissuesplit.py
├── autoencoder_deeplift.py
├── train.py
├── Predict_for_Predixcan.py
├── inference.py
├── gene_summary.csv
```
## Files and Their Functions

### `dataset/`

- **`get_target_gene.py`**  
  This script identifies the target genes for each fold in the dataset. 

- **`tissuesplit.py`**  
  This script splits the `cattleGTEx` dataset into different tissue types and helps organize data for tissue-specific analysis.

### Main Directory

- **`autoencoder_deeplift.py`**  
  This script applies a non-linear feature extraction method (Autoencoder with DeepLIFT) to extract a specified number of landmark genes. The number of landmark genes can be customized via input parameters.

- **`train.py`**  
  It allows users to customize training by setting parameters such as the number of epochs, hidden layer size, and momentum size, among others.

- **`Predict_for_Predixcan.py`**  
  This script provides functionality for running PrediXcan predictions. PrediXcan is based on the MetaXcan framework, which can be found [here](https://github.com/hakyimlab/MetaXcan).

- **`inference.py`**  
  This script provides an interface for users to select landmark genes. The selected landmark genes are then used to complete gene expression profiles and prepare for further TWAS analysis.

### `gene_summary.csv`

This file contains the results of our TWAS analysis. It includes a summary of the genes considered significant in milk production. 

---
