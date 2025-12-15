# DDML_PRS_ADRD  

Deep Data-Driven Machine Learning–based Polygenic Risk Score for Alzheimer’s Disease & Related Dementias

---

## 📖 Overview

This repository contains the implementation, documentation, and analysis code for **DDML_PRS**, a Bayesian variational autoencoder-based polygenic risk score (PRS) designed to improve prediction of Alzheimer’s disease and related dementias (ADRD) in the UK Biobank.  
The model is compared against classical PRS methods (SBayesR, clumping+thresholding).  

Key goals:
- Leverage GWAS summary statistics as Bayesian priors in VAE training  
- Produce a **continuous, standardized PRS** from latent features  
- Evaluate predictive performance (AUC, AUPRC) and stability across APOE and age strata  
- Provide full reproducibility (code, hyperparameters, docs)

---

## 🧰 Software & Tools

Analyses were carried out using both **R (v4.2.2)** and **Python (v3.10)** environments, employing a blend of statistical, survival, and machine learning packages.

### R packages
- `survival` (v3.8-3) — Cox proportional hazards modeling  
- `survminer` (v0.5.0) — visualization of survival / time-to-event curves  
- `pROC` (v1.18.0) — ROC curve analysis  
- `timeROC` (v0.4) — time-dependent ROC analysis  
- `ggplot2` (v3.4.4) — general data visualization  

### Python libraries
- `tensorflow` (v2.12.0)  
- `keras` (v2.12.0)  
- `scikit-learn` (v1.3.0)   

### Genetic / PRS tools
- **GCTB** (v2.0.3) — for implementing the SBayesR method  
- **PLINK** (v1.90b6.21) — for the clumping + thresholding (C+T) PRS  
- PRS values are standardized (Z-score) for comparability across methods  

All Python code (model training, evaluation) is open in this repository:  
https://github.com/shayanmostafaei/DDML_PRS_ADRD  

---

## 🏗️ Model Architecture & Method Description

The DDML_PRS is implemented via a **Bayesian Variational Autoencoder (VAE)** with the following structure:

- **Input layer**: genotype data (preselected SNPs, e.g. 80 SNPs) from UK Biobank (UKB) 
- **Encoder**: Dense layers 512 → 256 → 128 → latent mean & log-variance (dim = 50)  
- **Latent space**: 50 dimensions  
- **Decoder**: Mirror architecture 128 → 256 → 512, reconstructing genotypes  
- **KL regularization**: Divergence term computed using SNP-wise prior means & variances from GWAS summary statistics  
- **Inference → PRS**: The posterior mean of the latent vector is fed into a small regression head (dense layers → linear output) to yield a continuous PRS
- The VAE is trained on genotype inputs only (80 SNPs). Covariates (age, sex, PCs, and APOE genotype variables where applicable) are included only in downstream Cox/logistic models

**Hyperparameters and training settings**:
- Optimizer: Adam  
- Learning rate: 0.001  
- Batch size: 256  
- Epochs: up to 100 (with potential early stopping)  
- Latent dimension: 50  
- No dropout or weight decay  

For downstream evaluation, the continuous PRS is standardized (Z-score) and used in ROC / PR analyses, Cox models, and stratified analyses.

---

## Contact
For questions or contributions, please contact:
•	Dr. Shayan Mostafaei (shayan.mostafaei@ki.se) 
•	Dr. Sara Hägg (sara.hagg@ki.se)  

## Citation
Mostafaei S, et al. *Age- and APOE-Stratified Polygenic Risk Prediction of Alzheimer’s Disease and Related Dementias Using Machine Learning in the UK Biobank*. Alzheimer's Research & Therapy. 2025. (Manuscript in Peer review)

## 🧾 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with proper attribution.

---

## 🤝 How to Contribute / Update the Repository

We welcome contributions and updates that improve documentation, reproducibility, and analysis code.  
Please follow the steps below to safely create and submit updates.

### 🪄 1. Create a new branch
Always work on a separate branch rather than committing directly to `main`:
```bash
git checkout -b update/docs-and-model
