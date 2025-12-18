# DDML_PRS_ADRD

**Deep Data-Driven Machine Learning–based Polygenic Risk Score (DDML_PRS) for Alzheimer’s Disease & Related Dementias (ADRD)**

---

## 📖 Overview

This repository contains code and documentation for **DDML_PRS**, a **Bayesian variational autoencoder (VAE)**–based approach for constructing a continuous polygenic risk score (PRS) for ADRD in the **UK Biobank (UKB)**.

The study’s primary focus is **age- and APOE-stratified analysis**, and the main reported AUCs are **covariate-adjusted** (age/sex/10-PCs ± APOE genotype, where stated). For transparency, we also report **PRS-only** (genetics-only) performance separately.

**Comparators included in this project:**
- **DDML_PRS** (Bayesian VAE-derived PRS)
- **SBayesR PRS** (GCTB)
- **Clumping + Thresholding (C+T) PRS** (PLINK)
- Simple baselines (e.g., **logistic regression** on the same SNP set) for interpretability/context. 

---

## ✅ Key clarifications 

### 1) PRS-only vs PRS + covariates
We provide **two distinct evaluation settings**:

- **PRS-only (genetics-only):** PRS evaluated without age/sex/PCs/APOE genotype.
- **PRS + covariates (primary manuscript setting):** PRS combined **downstream** with covariates (age, sex, and genetic PCs; APOE genotype where stated) using logistic regression and/or survival models.

> The **primary study findings** correspond to the **PRS + covariates** setting, consistent with the study title (Age- and APOE-stratified analysis).

### 2) Covariates are NOT inputs to the VAE
The **Bayesian VAE is trained on genotype inputs only** (the selected SNP set).
Covariates (age, sex, PCs, and APOE genotype variables) are incorporated **only in downstream regression/survival models** and **not** as VAE input nodes.

### 3) APOE region definition (“with APOE region” vs “without APOE region”)
To avoid ambiguity, we explicitly distinguish:
- **With APOE region:** SNPs within the APOE region are included in PRS construction.
- **Without APOE region:** SNPs within the predefined APOE region are removed **prior** to PRS construction.

**Genome build:** GRCh37/hg19  
**APOE window:** `chr19: 44.4–46.5 Mb` 
This window is used to remove variants in LD with APOE ε2/ε3/ε4–defining SNPs prior to computing “no-APOE” PRS.

**APOE genotype variables:** ε2/ε3/ε4 allele counts are derived from `rs7412` and `rs429358` and are used **only as downstream covariates where stated**.

### 4) Leakage safeguards and evaluation
- Splits are performed to ensure **matched ADRD case/control proportion** between training and independent test sets (stratified by ADRD status).
- The **independent test set** is used **only for final evaluation** (not for training/early stopping).
- Model stability is assessed by repeating final training across multiple random seeds.

---

## 🧰 Software & Tools

Analyses were carried out using **R (v4.2.2)** and **Python (v3.10)**.

### R packages
- `survival` — Cox proportional hazards modeling  
- `survminer` — survival visualization  
- `pROC` — ROC curve analysis  
- `timeROC` — time-dependent ROC analysis  
- `ggplot2` — visualization  

### Python libraries
- `tensorflow` / `keras` 
- `scikit-learn` 
- `numpy`, `pandas`

### Genetic / PRS tools
- **GCTB** — SBayesR
- **PLINK** — clumping + thresholding PRS

---

## 🏗️ Model architecture & method description (DDML_PRS)

DDML_PRS is implemented via a **Bayesian Variational Autoencoder (VAE)**:

- **Input:** genotype matrix of the selected SNPs (e.g., 80 SNPs)  
- **Encoder:** Dense layers `512 → 256 → 128 → (z_mean, z_log_var)`  
- **Latent space:** 50 dimensions  
- **Decoder:** mirror network `128 → 256 → 512` reconstructing genotype input  
- **Bayesian regularization:** KL divergence term incorporates **GWAS-informed priors** (from external GWAS summary statistics)  
- **PRS derivation:** a continuous scalar **DDML_PRS** is derived from the latent posterior mean and standardized (Z-score) for downstream evaluation

**Training configuration:**
- Optimizer: Adam  
- Learning rate: 0.001  
- Batch size: 256  
- Epochs: up to 100  
- Early stopping: used on a held-out validation subset of the training data  
- Dropout: none  
- Explicit weight decay/L2: none  
> Regularization is provided implicitly via GWAS-informed priors and early stopping.

---

## 📊 Reported metrics 

We report:
- **AUC (ROC)** for classification performance
- **AUPRC** for precision–recall performance (useful under class imbalance)
- Survival outcomes (C-index / time-dependent ROC) where applicable

**Important:** AUC values differ depending on whether covariates are included downstream.
- **PRS-only AUCs** quantify genetics-only discrimination.
- **PRS + covariates AUCs** reflect combined predictive models (age/sex/PCs ± APOE genotype), which are the primary manuscript results.

---

## 🗂️ Data inputs and expected formats

Because UK Biobank data cannot be redistributed, this repository assumes you have access to UKB-approved genotype and phenotype data.

Typical inputs include:
- Genotype matrix for the selected SNPs (e.g., `n_samples × 80`)
- ADRD case/control label
- Covariates for downstream models (age, sex, genetic PCs; APOE genotype where applicable)
- External GWAS summary statistics used for priors (excluding UKB where applicable)

---

## ▶️ Reproducible workflow 

1) **Prepare genotype and phenotype arrays** (SNP matrix + labels; covariates for downstream models)  
2) **Train DDML VAE on training data only** (internal validation for early stopping)  
3) **Derive standardized DDML_PRS**  
4) Evaluate:
   - **PRS-only** AUC
   - **PRS + covariates** AUC via downstream regression/survival models  
5) Repeat training under multiple seeds for stability

> Scripts are organized to make it explicit which steps use genotype-only training vs downstream covariate-adjusted evaluation.

---

## 📌 Contact

- Dr. Shayan Mostafaei — shayan.mostafaei@ki.se  
- Dr. Sara Hägg — sara.hagg@ki.se  

---

## 📝 Citation

Mostafaei S, et al. *Improved Polygenic Risk Prediction for Alzheimer’s Disease and Related Dementias Using Deep Learning: Age and APOE-Stratified Analysis*. Alzheimer’s Research & Therapy. 2025. (Manuscript under peer review)

---

## 🧾 License

MIT License.

---

## 🤝 Contributing

Contributions improving documentation, reproducibility, and analysis code are welcome.

### 1) Create a new branch
```bash
git checkout -b update/docs-and-model
