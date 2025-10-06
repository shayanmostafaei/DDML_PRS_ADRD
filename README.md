## DDML_PRS_ADRD

Polygenic Risk Scores (PRS) for Predicting Alzheimer's Disease and Related Dementias (ADRD) in UK Biobank

## Overview
This repository contains code, model specifications and documentation for "Improved Polygenic Risk Prediction for Alzheimer’s Disease and Related Dementias Using Deep Learning: Age and APOE-Stratified Analysis" (Mostafaei et al.). The project compares three PRS construction methods and documents a Bayesian variational autoencoder–based PRS (DDML_PRS).

**Key contents**
- `DDML_PRS_Model_Specification.md` — detailed model architecture & hyperparameters. 

## Purpose
- Benchmark a Bayesian variational autoencoder (VAE) PRS (DDML_PRS) against SBayesR and clumping+thresholding (C+T) approaches.
- Evaluate model performance across APOE strata and age ranges, and report reproducibility and stability metrics.

## Reproducibility & provenance
- Model architecture, final hyperparameters, and per-seed stability results are described in `DDML_PRS_Model_Specification.md`.
- All preprocessing scripts, analysis notebooks and run logs are placed in "DDML_PRS_Model.py".

## Clarifications included in this update
- Clarified how the 50-dimensional latent vector is turned into a scalar PRS (posterior mean of latent variables). See model spec. 
- Made training hyperparameters explicit (optimizer, learning rate, batch size, KL annealing schedule, early stopping criteria).

## How to contribute / update the repository
1. Create a branch: `git checkout -b update/readme-and-supplementary`
2. Add the updated files (see instructions below).
3. Commit and push, then open a pull request with the suggested PR text below.
 
## Contact
For questions or contributions, please contact:
•	Dr. Shayan Mostafaei (shayan.mostafaei@ki.se) 
•	Dr. Sara Hägg (sara.hagg@ki.se)  

## Citation
Mostafaei S, et al. *Age- and APOE-Stratified Polygenic Risk Prediction of Alzheimer’s Disease and Related Dementias Using Machine Learning in the UK Biobank*. Alzheimer's Research & Therapy. 2025. (Manuscript in Peer review)




