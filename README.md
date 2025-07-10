## Overview
This repository provides code for constructing polygenic risk scores (PRS) for Alzheimer's disease and related dementias (ADRD) using a Bayesian variational autoencoder-based deep learning model (DDML). The model was developed and validated using data from the UK Biobank and compared against traditional PRS methods including SBayesR and Clumping + Thresholding (C+T). 

## Data or Resources
For more details on the genetic data in the UK Biobank, see [UK Biobank Genetic Data](https://biobank.ndph.ox.ac.uk/ukb/label.cgi?id=263).
For more details on the ADRD phenotypes in the UK Biobank, see [UK Biobank Dementia outcomes](https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=47).

## Input/Output 
Input: Imputed genotype data, GWAS summary statistics (e.g., Bellenguez et al. 2022), and phenotype labels.
Output: Individual-level PRS values and model evaluation metrics.

## Key Features 
- Implements a deep data-driven machine learning (DDML) approach using a Bayesian variational autoencoder.
- Incorporates GWAS summary statistics and individual-level genotype data.
- Supports polygenic hazard score (PHS) modeling for time-to-event analysis.
- Evaluates model performance using AUC, C-index, HRs, and confusion matrix metrics.
- Stratified analysis by age groups, APOE-ε4 carrier status, and ADRD subtypes. 
  
## Contact
For questions or contributions, please contact:
•	Dr. Shayan Mostafaei (shayan.mostafaei@ki.se) 
•	Dr. Sara Hägg (sara.hagg@ki.se)
• Dr. Daniel Wikström Shemer (daniel.wikstrom.shemer@ki.se) 

## Citation
Mostafaei S, et al. *Age- and APOE-Stratified Polygenic Risk Prediction of Alzheimer’s Disease and Related Dementias Using Machine Learning in the UK Biobank*. JAMA Network Open. 2025. (Manuscript in preparation)

## Requirements

See `requirements.txt` for the list of packages used in the analysis.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/shayanmostafaei/DDML_PRS_ADRD.git
cd DDML_PRS_ADRD
pip install -r requirements.txt

