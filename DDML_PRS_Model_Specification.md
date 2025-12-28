# DDML_PRS Model Specification

**Model run ID**  
`DDML_PRS_final`

---

## 1) Model architecture (encoder / decoder / latent)

- **Input:** 80 GWAS-significant SNPs preselected as described in the main manuscript (Bellenguez et al., 2022).
- **Encoder:** Fully connected layers with ReLU activations: **512 → 256 → 128**.
- **Latent space:** **50-dimensional** latent representation.
  - Encoder outputs **z_mean** and **z_log_var**.
  - Latent sampling uses the **reparameterization trick**.
- **Decoder:** Symmetric mirror network: **128 → 256 → 512**, reconstructing the genotype input.
  - Output layer uses **sigmoid** for reconstruction in **[0,1]** (if genotypes are encoded {0,1,2}, they should be scaled to [0,1] before training).
- **PRS derivation (fixed):**
  - A single continuous scalar score (**DDML_PRS**) is derived deterministically from the latent posterior mean (**z_mean**).
  - Default mapping (as implemented): **DDML_PRS_raw = mean(z_mean across latent dimensions)**.
  - The final PRS is **standardized (Z-score)** using **training data only**, and the same standardization is applied to validation/test (no leakage).

> **Covariate handling:** Age, sex, genetic PCs, and APOE genotype variables are **not** inputs to the VAE. They are incorporated **only in downstream regression/survival models** where stated.

---

## 2) Bayesian priors & KL regularization

- **GWAS-informed priors (UKB-excluded):**
  - External GWAS summary statistics from **Jansen et al.** (excluding UK Biobank participants) were used to inform prior distributions, providing biologically motivated regularization.
- **Prior parameterization (implementation assumption):**
  - Priors are defined over **latent dimensions** as a diagonal Gaussian:
    - **p(z) = N(μ_prior, diag(σ²_prior))**, where priors are stored as *(latent_dim × 2)*: **[prior_mean, prior_var]**.
- **KL divergence:**
  - The objective includes **KL(q(z|x) || p(z))** where:
    - **q(z|x) = N(z_mean, diag(exp(z_log_var)))**
    - **p(z)   = N(prior_mean, diag(prior_var))**
- **KL annealing:** Linear annealing of KL weight from **0 → 1** over the first **20 epochs**.

> If priors are available per-SNP (e.g., 80×2), they must be mapped to latent priors or the KL term must be redesigned to operate in SNP/weight space. Any such mapping must be explicitly documented and kept fixed.

---

## 3) Training, validation, and data splitting (leakage safeguards)

- **Cohort size:** n ≈ **276,566** unrelated UK Biobank participants (White British/European ancestry).
- **Training vs independent test split:**
  - Random split with **stratification by ADRD case/control status**.
  - Training = **66.7%**, Test = **33.3%**.
  - Paper split counts:
    - Train: **184,378** (885 cases, 183,493 controls)
    - Test: **92,188** (443 cases, 91,745 controls)
- **Internal validation:**
  - **10% of the training set** held out as an internal validation subset (stratified).
  - Used for monitoring convergence and early stopping only.
- **Test-set usage:**
  - The independent test set is used **only for final PRS-only evaluation**.
  - The test set is **never** used for training, early stopping, validation, hyperparameter selection, or standardization/scaling.

- **Early stopping:**
  - Monitored on **validation loss (ELBO proxy)**.
  - **Patience = 10 epochs**, with **restore best weights**.

---

## 4) Final hyperparameters (fixed a priori)

- **Optimizer:** Adam
- **Learning rate:** **0.001**
- **Batch size:** **256**
- **Epochs (max):** **100**
- **Latent dimension:** **50**
- **KL anneal epochs:** **20**
- **Dropout:** none
- **Explicit weight decay / L2 regularization:** none

> Regularization is provided via GWAS-informed priors, KL regularization (with annealing), and early stopping.

---

## 5) APOE region and APOE genotype handling (distinct concepts)

### APOE region inclusion/exclusion in PRS construction
- **Genome build:** hg19 (GRCh37)
- **APOE region definition:** **chr19: 44–46 Mb**
- **With APOE region:** variants in this window are retained during PRS construction.
- **Without APOE region:** variants in this window are removed prior to PRS construction, and PRS are recomputed using remaining variants.

### APOE genotype variables as downstream covariates (NOT VAE inputs)
- APOE ε2/ε3/ε4 is derived from **rs429358** and **rs7412**.
- These genotype variables are used **only as covariates** in downstream regression/survival analyses where stated.

---

## 6) Performance metrics (independent test set)

### PRS-only models (no covariates)
- **DDML_PRS (with APOE region):** AUC = **0.6907** (95% CI: 0.68–0.70)
- **DDML_PRS (without APOE region):** AUC = **0.6542** (95% CI: 0.65–0.67)

### PRS + covariates models (downstream regression; main manuscript setting)
- **DDML_PRS + age + sex + 10 PCs (with APOE region):** AUC = **0.83** (95% CI: 0.83–0.85)

> **Interpretation note:** The higher AUCs emphasized in the main manuscript correspond to the **PRS + covariates** setting (and stratified analyses), not PRS-only models.

---

## 7) Subgroup and time-dependent analyses (as reported)

- **Age-stratified performance:** Peak subgroup AUCs up to ~**0.85** reported in subgroup/sensitivity analyses (PRS + covariates setting).
- **Cox proportional hazards models:** Overall C-index (PRS + covariates setting): **0.840**.

---

## 8) Reproducibility and stability (multi-seed)

- **Number of independent runs:** **5** (fixed architecture/hyperparameters and fixed split strategy).
- **Per-seed test AUCs (DDML_PRS + covariates; reported):**
  - Seed 123: 0.832
  - Seed 12321: 0.835
  - Seed 42: 0.843
  - Seed 2024: 0.839
  - Seed 99: 0.845
- **Mean ± SD:** **0.839 ± 0.005**

> PRS-only AUCs are expected to be lower than covariate-adjusted models; both are reported for transparency.

---

## 9) Notes on pre-specification and pilot checks

- Model architecture and hyperparameters were fixed *a priori* based on prior Bayesian VAE applications in genomics and stability considerations.
- Limited exploratory runs restricted to the training data were used only to verify numerical stability and convergence.
- No hyperparameter optimization and no model selection were performed using the independent test set.
