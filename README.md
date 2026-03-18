# DDML_PRS_ADRD

**Deep Data-Driven Machine Learning–based Polygenic Risk Score (DDML_PRS) for Alzheimer’s Disease & Related Dementias (ADRD)**

This repository accompanies the manuscript: **“Improved Polygenic Risk Prediction for Alzheimer’s Disease and Related Dementias Using Deep Learning: Age and APOE-Stratified Analysis”**.

> **Key takeaway for readers:**  
> The article’s *primary* reported AUCs are **covariate-adjusted** (PRS combined downstream with age/sex/10 PCs ± APOE genotype, as stated). We also report **PRS-only (genetics-only)** performance separately to avoid ambiguity (especially around “no-APOE” wording).

---

## Repository contents (high level)

- `DDML_PRS_Model.py` — main DDML_PRS implementation (Bayesian VAE)
- `DDML_PRS_Model_Specification` — model specification details
- `priors/` — prior inputs / helper files (see “GWAS priors” below)
- `SBayesR_sumstats_without_ukb.snpRes.gz` — SBayesR-related summary-stat output (UKB-excluded context)
- `Requirements.txt` — Python dependencies

---

## Study snapshot 

- **Population:** 276,566 unrelated UK Biobank participants of White British/European ancestry, median follow-up 9.19 years.
- **Phenotype window:** follow-up to the earliest of ADRD diagnosis, death, or end of follow-up (Apr 1, 2018).
- **Train/test split (fixed):**
  - Train: 2/3 of sample, **N=184,378** (885 ADRD cases, 183,493 controls)
  - Test: 1/3 of sample, **N=92,188** (443 ADRD cases, 91,745 controls)
  - Split preserves case/control proportion (stratified by ADRD status).

---

## “With APOE region” vs “Without APOE region” (critical definitions)

To avoid any misunderstanding, we explicitly distinguish:

### A) APOE region inclusion/exclusion in PRS construction
- **Genome build:** GRCh37/hg19  
- **APOE region window:** `chr19:44–46 Mb`
  - **With APOE region:** variants in this window are retained during PRS construction.
  - **Without APOE region:** all SNPs within `chr19:44–46 Mb` are removed *prior* to PRS construction, and PRS are recomputed using remaining variants.

### B) APOE genotype variables as downstream covariates (NOT VAE inputs)
- APOE ε2/ε3/ε4 genotype is derived from **rs429358** and **rs7412**.
- These genotype variables are used **only** as covariates in downstream regression/survival models where stated.
- **They are not input nodes to the Bayesian VAE.**

---

## Variant set and GWAS priors (DDML_PRS)

### DDML_PRS SNP panel (80 SNPs)
- DDML_PRS uses **80 independent GWAS-significant SNPs** selected from Bellenguez et al. (2022).
- These 80 SNPs are the genotype input to the VAE.

### GWAS-informed priors (UKB-excluded)
- Prior means/variances are defined using **UK Biobank–excluded** GWAS summary statistics (Jansen et al., obtained directly from authors in the study).
- This UKB-excluded constraint is used to reduce sample overlap with UK Biobank during prior specification.
- If the exact author-provided UKB-excluded summary statistics cannot be redistributed, users must supply an equivalent priors file locally (or use what is provided under `priors/`, if applicable).

> Note: The 80-SNP panel was selected by Bellenguez et al. (2022). Because priors are taken from an independent (UKB-excluded) source, not all 80 variants remain genome-wide significant in that UKB-excluded dataset. In the Bayesian framework, variants with weaker evidence contribute less via smaller effects and larger uncertainty.

---

## DDML_PRS method (Bayesian VAE)

### Architecture (fixed a priori)
- Encoder: Dense layers **512 → 256 → 128** (ReLU)
- Latent space: **50 dimensions**
- Decoder: mirror network **128 → 256 → 512**
- Bayesian regularization: KL term incorporates GWAS-informed priors
- PRS derivation: posterior mean of latent variables aggregated to a single continuous scalar score (DDML_PRS), standardized for downstream evaluation

### Training (fixed a priori; no tuning)
- Optimizer: Adam (learning rate **0.001**)
- Batch size: **256**
- Early stopping on **internal validation** subset drawn from training data only
- No dropout / no explicit weight decay (regularization via priors + early stopping)
- **No architecture search and no hyperparameter tuning** were performed.
- Two diagnostic pilot runs (training data only) checked numerical stability and sensitivity to prior strength; they were not used for model selection.

### Reproducibility & robustness
- Final prespecified model retrained across **5 random seeds** using the same fixed split:
  - test-set AUC range: **0.832–0.845** (mean ± SD: **0.839 ± 0.005**).
- A bounded robustness analysis (training/validation only) checked learning rates and KL-annealing schedules; the independent test set was not used for this check.

---

## Results that often get misinterpreted (PRS-only vs covariate-adjusted)

### PRS-only (genetics-only) AUCs (no covariates)
These are standalone PRS models **without** age/sex/PCs/APOE genotype:

| PRS model     | With APOE region AUC (95% CI) | Without APOE region AUC (95% CI) |
|--------------|-------------------------------:|----------------------------------:|
| **DDML_PRS** | 0.6907 (0.68–0.70)             | 0.6542 (0.65–0.67)                |
| SBayesR_PRS  | 0.6604 (0.65–0.67)             | 0.6101 (0.60–0.62)                |
| C+T_PRS      | 0.6120 (0.60–0.62)             | 0.5805 (0.57–0.59)                |


---

## How to run (practical workflow)

Because UK Biobank data cannot be redistributed, you must provide:
- Genotypes for the selected SNP set (e.g., an `N × 80` matrix)
- ADRD case/control labels
- Covariates (age, sex, 10 genetic PCs, and APOE genotype where applicable) for downstream models

Suggested minimal workflow:
1. Prepare inputs:
   - `X_geno`: genotype matrix (N × 80)
   - `y`: ADRD status (0/1)
   - optional downstream covariates: age, sex, PCs, APOE genotype variables
2. Split data exactly as in paper (fixed split; stratified by `y`):
   - train: 2/3, test: 1/3
3. Train the Bayesian VAE on training data only (validation drawn from training only)
4. Export standardized DDML_PRS for all individuals
5. Evaluate:
   - PRS-only AUC (DDML_PRS only)
   - PRS + covariates AUC (primary manuscript setting)

> Entry points: see `DDML_PRS_Model.py` (implementation) and `DDML_PRS_Model_Specification` (settings).

---

## Notes on external validation (scope / limitation)

External replication was not attempted in the study due to practical constraints (cohort differences, phenotype definitions, and limited sample sizes for comparable age- and APOE-stratified analyses). External validation remains important future work.

---

## 📌 Contact

- Dr. Shayan Mostafaei — shayan.mostafaei@ki.se  
- Dr. Sara Hägg — sara.hagg@ki.se  

---

## 📝 Citation

Mostafaei S, Shemer DW, Mak JK, Karlsson IK, Hägg S. Improved polygenic risk prediction for alzheimer’s disease and related dementias using deep learning: age and APOE-stratified analysis. Alzheimer's Research & Therapy. 2026 Mar 12.

---

## 🧾 License

MIT License (see `LICENSE`)

---
