"""
DDML_PRS: Deep Data-Driven Machine Learning–based Polygenic Risk Score for ADRD
Repository: https://github.com/shayanmostafaei/DDML_PRS_ADRD

Purpose
-------
This script trains a Bayesian Variational Autoencoder (VAE) on genotype inputs only
(e.g., 80 preselected SNPs) and derives a continuous, standardized DDML_PRS score.
It then evaluates PRS-only discrimination (no covariates) on an independent test set
and saves the standardized DDML_PRS for downstream covariate-adjusted models.

Key methodological safeguards 
-----------------------------------------------
1) Covariates are NOT inputs to the VAE:
   Age, sex, genetic PCs, and APOE genotype variables are incorporated only in
   downstream regression/survival models (outside this script).

2) No data leakage:
   The independent test set is used ONLY for final PRS-only evaluation and is never
   used for training, early stopping, or validation. 

3) Matched ADRD prevalence:
   Train/test and train/validation splits are stratified by ADRD case/control status
   to preserve class proportions.

Environment (tested)
--------------------
- Python 3.10
- TensorFlow/Keras 2.12.x
- scikit-learn 1.3.x
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, average_precision_score
import warnings

warnings.filterwarnings("ignore")


# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_seeds(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Data loading
# -----------------------------
def load_data(data_path: str):
    """
    Load genotype and label arrays from preprocessed .npy files.

    Expected files:
      - genotype_data.npy : (n_samples, n_snps) float32/float64
      - labels.npy        : (n_samples,) int {0,1}
      - gwas_summary.npy  : (latent_dim, 2) prior_mean and prior_var for latent dimensions

    Notes:
      - This script uses genotype + labels only.
      - Covariates are handled downstream (outside the VAE), as described in the study's manuscript.
    """
    X = np.load(os.path.join(data_path, "genotype_data.npy")).astype(np.float32)
    y = np.load(os.path.join(data_path, "labels.npy")).astype(np.int32)
    gwas_summary = np.load(os.path.join(data_path, "gwas_summary.npy")).astype(np.float32)
    return X, y, gwas_summary


# -----------------------------
# VAE components
# -----------------------------
def sampling(args):
    """Reparameterization trick: z = mu + sigma * epsilon."""
    z_mean, z_log_var = args
    eps = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_encoder(input_dim: int, latent_dim: int) -> keras.Model:
    """Build encoder network."""
    inputs = keras.Input(shape=(input_dim,), name="genotype_input")
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


def build_decoder(output_dim: int, latent_dim: int) -> keras.Model:
    """Build decoder network."""
    latent_inputs = keras.Input(shape=(latent_dim,), name="latent_input")
    x = layers.Dense(128, activation="relu")(latent_inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="sigmoid", name="reconstruction")(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


class BayesianVAE(keras.Model):
    """
    Bayesian VAE incorporating GWAS-informed priors via a KL term.

    IMPORTANT:
      This implementation assumes gwas_summary.npy provides priors over the LATENT dimensions,
      i.e., shape (latent_dim, 2): [prior_mean, prior_var].

      If your available priors are per-SNP (e.g., 80 x 2), you need to map them to latent priors
      (or redesign the KL term to operate in SNP/weight space) and document that mapping.
    """

    def __init__(self, encoder: keras.Model, decoder: keras.Model,
                 prior_mean: np.ndarray, prior_var: np.ndarray,
                 kl_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.prior_mean = tf.constant(prior_mean, dtype=tf.float32)
        self.prior_var = tf.constant(prior_var, dtype=tf.float32)
        self.kl_weight = tf.Variable(kl_weight, trainable=False, dtype=tf.float32)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Reconstruction loss (MSE)
            recon_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstruction))

            # KL divergence to diagonal Gaussian prior N(prior_mean, prior_var)
            kl = -0.5 * tf.reduce_mean(
                1.0
                + z_log_var
                - tf.square(z_mean - self.prior_mean) / self.prior_var
                - tf.exp(z_log_var)
            )

            total_loss = recon_loss + self.kl_weight * kl

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_weight": self.kl_weight,
        }


class KLAnnealingCallback(keras.callbacks.Callback):
    """Linearly anneal KL weight from 0 to 1 over `anneal_epochs`."""
    def __init__(self, vae: BayesianVAE, anneal_epochs: int = 20):
        super().__init__()
        self.vae = vae
        self.anneal_epochs = max(1, anneal_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        weight = min(1.0, float(epoch) / float(self.anneal_epochs))
        self.vae.kl_weight.assign(weight)


# -----------------------------
# PRS derivation + evaluation
# -----------------------------
def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score standardization."""
    return (x - x.mean()) / (x.std() + 1e-8)


def derive_ddml_prs(z_mean: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Derive a 1D continuous score from latent posterior mean (z_mean).

    By default, uses the mean across latent dimensions to obtain a deterministic scalar.
    If your manuscript uses a different mapping, keep it fixed and document it consistently
    in the README and model specification.
    """
    if method == "mean":
        score = z_mean.mean(axis=1)
    else:
        raise ValueError(f"Unknown PRS derivation method: {method}")
    return standardize(score)


def main():
    parser = argparse.ArgumentParser(
        description="Train Bayesian VAE (genotype-only) and derive DDML_PRS (PRS-only evaluation)."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path containing genotype_data.npy, labels.npy, gwas_summary.npy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--latent_dim", type=int, default=50, help="Latent dimension (per manuscript).")
    parser.add_argument("--test_size", type=float, default=1/3,
                        help="Independent test split proportion (default 1/3).")
    parser.add_argument("--val_size", type=float, default=0.10,
                        help="Validation split proportion within the training data (default 10%).")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--kl_anneal_epochs", type=int, default=20, help="KL annealing epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--prs_method", type=str, default="mean", choices=["mean"],
                        help="How to map latent mean to scalar PRS (default: mean).")
    args = parser.parse_args()

    set_seeds(args.seed)

    # Load data
    X, y, gwas_summary = load_data(args.data_path)
    _, input_dim = X.shape

    # Split: train vs independent test (stratify by ADRD status for matched ADRD prevalence)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    # Split: internal validation from training only 
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train_full
    )

    # Validate prior dimensions
    if gwas_summary.shape[0] != args.latent_dim or gwas_summary.shape[1] < 2:
        raise ValueError(
            f"gwas_summary.npy must have shape (latent_dim, 2). "
            f"Got {gwas_summary.shape}, latent_dim={args.latent_dim}. "
            f"If your priors are per-SNP (e.g., 80x2), map them to latent priors or "
            f"redesign the KL term to operate in SNP/weight space."
        )

    prior_mean = gwas_summary[:, 0]
    prior_var = np.clip(gwas_summary[:, 1], 1e-8, None)  

    # Build and train Bayesian VAE (genotype-only)
    encoder = build_encoder(input_dim=input_dim, latent_dim=args.latent_dim)
    decoder = build_decoder(output_dim=input_dim, latent_dim=args.latent_dim)

    vae = BayesianVAE(encoder, decoder, prior_mean=prior_mean, prior_var=prior_var, kl_weight=0.0)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr))

    callbacks = [
        KLAnnealingCallback(vae, anneal_epochs=args.kl_anneal_epochs),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stop_patience,
            restore_best_weights=True
        ),
    ]

    # IMPORTANT: validation uses TRAINING ONLY (NO TEST leakage)
    vae.fit(
        X_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, None),
        callbacks=callbacks,
        verbose=2
    )

    # Derive PRS from latent posterior mean (z_mean) on test set
    z_mean_test, _, _ = encoder.predict(X_test, verbose=0)
    ddml_prs = derive_ddml_prs(z_mean_test, method=args.prs_method)

    # PRS-only evaluation on independent test set
    auc = roc_auc_score(y_test, ddml_prs)
    fpr, tpr, thr = roc_curve(y_test, ddml_prs)
    youden = np.argmax(tpr - fpr)
    opt_thr = thr[youden]
    cm = confusion_matrix(y_test, ddml_prs > opt_thr)
    auprc = average_precision_score(y_test, ddml_prs)

    print("\n--- DDML_PRS (PRS-only) Evaluation on Independent Test Set ---")
    print(f"ROC AUC: {auc:.3f}")
    print(f"AUPRC:  {auprc:.3f}")
    print(f"Optimal threshold (Youden index): {opt_thr:.3f}")
    print("Confusion matrix:")
    print(cm)
    print("\nNote: Covariate-adjusted performance (age/sex/PCs/APOE genotype) is evaluated "
          "in downstream regression/survival models, not within this script.")

    # Save standardized PRS for downstream models
    out_path = os.path.join(args.data_path, f"DDML_PRS_test_seed{args.seed}.npy")
    np.save(out_path, ddml_prs)
    print(f"\nSaved standardized DDML_PRS (test set) to: {out_path}")


if __name__ == "__main__":
    main()
# --------------------------------------------------------------------------
# End
# --------------------------------------------------------------------------
