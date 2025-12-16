"""
DDML_PRS: Deep Data-Driven Machine Learning-based Polygenic Risk Score for ADRD
Author: Shayan Mostafaei et al.
Repository: https://github.com/shayanmostafaei/DDML_PRS_ADRD

This script:
  1) Trains a Bayesian VAE on genotype inputs only (80 SNPs).
  2) Derives a continuous DDML_PRS score from the latent representation.
  3) Evaluates PRS-only discrimination on an independent test set.

Important methodological safeguards:
  - The independent test set is NEVER used during VAE training, validation, early stopping,
    pilot checks, or model selection.
  - Validation is performed only on a held-out subset of the training data.

Environment (tested):
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
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Data loading
# -----------------------------
def load_data(data_path: str):
    """
    Loads preprocessed arrays saved as .npy.

    Expected files:
      - genotype_data.npy  (n_samples, n_snps=80)
      - labels.npy         (n_samples,) binary ADRD status (0/1)
      - gwas_summary.npy   (n_prior_dims, 2) prior mean/variance

    Notes:
      - This script uses genotype + labels only.
      - Covariates (age/sex/10-PCs/APOE genotype) are used in downstream models elsewhere,
        not as inputs to the VAE.
    """
    X = np.load(os.path.join(data_path, "genotype_data.npy")).astype(np.float32)
    y = np.load(os.path.join(data_path, "labels.npy")).astype(np.int32)
    gwas_summary = np.load(os.path.join(data_path, "gwas_summary.npy")).astype(np.float32)
    return X, y, gwas_summary


# -----------------------------
# VAE components
# -----------------------------
def sampling(args):
    """Reparameterization trick."""
    z_mean, z_log_var = args
    eps = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_encoder(input_dim: int, latent_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="genotype_input")
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


def build_decoder(output_dim: int, latent_dim: int) -> keras.Model:
    latent_inputs = keras.Input(shape=(latent_dim,), name="latent_input")
    x = layers.Dense(128, activation="relu")(latent_inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="sigmoid", name="reconstruction")(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


class BayesianVAE(keras.Model):
    """
    Bayesian VAE incorporating priors in the KL term.
    """

    def __init__(self, encoder, decoder, prior_mean, prior_var, kl_weight=1.0, **kwargs):
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

            # Reconstruction loss: MSE over SNPs
            recon_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstruction))

            # KL divergence to latent prior
            # KL(q(z|x) || p(z)) with diagonal Gaussian p(z)=N(prior_mean, prior_var)
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
        w = min(1.0, float(epoch) / float(self.anneal_epochs))
        self.vae.kl_weight.assign(w)


# -----------------------------
# PRS derivation + evaluation
# -----------------------------
def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Train Bayesian VAE and derive DDML_PRS (PRS-only).")
    parser.add_argument("--data_path", type=str, required=True, help="Path containing .npy input files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--latent_dim", type=int, default=50, help="Latent dimension (per manuscript).")
    parser.add_argument("--test_size", type=float, default=1/3, help="Independent test split proportion.")
    parser.add_argument("--val_size", type=float, default=0.10, help="Validation split proportion inside training.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()

    set_seeds(args.seed)

    # Load
    X, y, gwas_summary = load_data(args.data_path)
    n_samples, input_dim = X.shape

    # Split: train vs independent test (stratify by labels only)
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

    if gwas_summary.shape[0] != args.latent_dim or gwas_summary.shape[1] < 2:
        raise ValueError(
            f"gwas_summary.npy must have shape (latent_dim, 2). "
            f"Got {gwas_summary.shape}, latent_dim={args.latent_dim}. "
            f"If your priors are per-SNP (80x2), map them to latent priors outside this script."
        )

    prior_mean = gwas_summary[:, 0]
    prior_var = gwas_summary[:, 1]
    prior_var = np.clip(prior_var, 1e-8, None)  # numerical safety

    # Build and train VAE (genotype-only)
    encoder = build_encoder(input_dim=input_dim, latent_dim=args.latent_dim)
    decoder = build_decoder(output_dim=input_dim, latent_dim=args.latent_dim)

    vae = BayesianVAE(encoder, decoder, prior_mean=prior_mean, prior_var=prior_var, kl_weight=0.0)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr))

    callbacks = [
        KLAnnealingCallback(vae, anneal_epochs=20),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
    ]

    vae.fit(
        X_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, None),  # validation from TRAINING ONLY
        callbacks=callbacks,
        verbose=2
    )

    # Derive DDML_PRS: use latent posterior mean; collapse to scalar by averaging (simple, deterministic)
    z_mean_test, _, _ = encoder.predict(X_test, verbose=0)
    ddml_prs = z_mean_test.mean(axis=1)
    ddml_prs = standardize(ddml_prs)

    # PRS-only evaluation
    auc = roc_auc_score(y_test, ddml_prs)
    fpr, tpr, thr = roc_curve(y_test, ddml_prs)
    youden = np.argmax(tpr - fpr)
    opt_thr = thr[youden]
    cm = confusion_matrix(y_test, ddml_prs > opt_thr)
    auprc = average_precision_score(y_test, ddml_prs)

    print("\n--- DDML_PRS (PRS-only) Evaluation on Independent Test Set ---")
    print(f"ROC AUC: {auc:.3f}")
    print(f"AUPRC:  {auprc:.3f}")
    print(f"Optimal threshold (Youden): {opt_thr:.3f}")
    print("Confusion matrix:")
    print(cm)

    # Save PRS for downstream covariate models
    out_path = os.path.join(args.data_path, f"DDML_PRS_test_seed{args.seed}.npy")
    np.save(out_path, ddml_prs)
    print(f"\nSaved standardized DDML_PRS for test set to: {out_path}")
    print("Note: Age/sex/PCs/APOE genotype are incorporated only in downstream models (not in the VAE).")


if __name__ == "__main__":
    main()
# --------------------------------------------------------------------------
# End
# --------------------------------------------------------------------------

