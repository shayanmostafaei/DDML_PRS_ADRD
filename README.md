## Overview
This repository provides code for constructing polygenic risk scores (PRS) for Alzheimer's disease and related dementias (ADRD) using a Bayesian variational autoencoder-based deep learning model (DDML). The model was developed and validated using data from the UK Biobank and compared against traditional PRS methods including SBayesR and Clumping + Thresholding (C+T). 

## Data or Resources
For more details on the genetic data in the UK Biobank, see [UK Biobank Genetic Data](https://biobank.ndph.ox.ac.uk/ukb/label.cgi?id=263).
For more details on the ADRD phenotypes in the UK Biobank, see [UK Biobank Dementia outcomes](https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=47).

## Key Features
- Implements a deep data-driven machine learning (DDML) approach using a Bayesian variational autoencoder.
- Incorporates GWAS summary statistics and individual-level genotype data.
- Supports polygenic hazard score (PHS) modeling for time-to-event analysis.
- Evaluates model performance using AUC, C-index, HRs, and confusion matrix metrics.
- Stratified analysis by age groups, APOE-ε4 carrier status, and ADRD subtypes. 
  
## Contributors & Maintainers
This project is maintained by:
•	[Dr. Shayan Mostafaei] (shayan.mostafaei@ki.se) 
•	Collaborators/Contributors: Dr. Sara Hägg (sara.hagg@ki.se) and Dr. Daniel Wikström Shemer (daniel.wikstrom.shemer@ki.se)

## Citation
Mostafaei S, et al. *Age- and APOE-Stratified Polygenic Risk Prediction of Alzheimer’s Disease and Related Dementias Using Machine Learning in the UK Biobank*. JAMA Network Open. 2025. (Manuscript in preparation)

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/shayanmostafaei/DDML_PRS_ADRD.git
cd DDML_PRS_ADRD
pip install -r requirements.txt

