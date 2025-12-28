"""
DDML_PRS: Deep Data-Driven Machine Learning–based Polygenic Risk Score for ADRD

Purpose
-------
This script trains a Bayesian Variational Autoencoder (VAE) on genotype inputs only
(e.g., 80 preselected SNPs) and derives a continuous, standardized DDML_PRS score.
It evaluates PRS-only discrimination (no covariates) on an independent test set and
saves standardized DDML_PRS outputs for downstream covariate-adjusted models.

Key methodological safeguards
-----------------------------
1) Covariates are NOT inputs to the VAE:
   Age, sex, genetic PCs, and APOE genotype variables are incorporated only in
   downstream regression/survival models (outside this script).

2) No data leakage:
   - The independent test set is used ONLY for final PRS-only evaluation.
   - The test set is never used for training, early stopping, validation, or scaling.
   - Standardization parameters (mean/std) are fit on TRAINING ONLY, then applied to
     validation/test (and optionally full cohort outputs).

3) Matched ADRD prevalence:
   Train/test and train/validation splits are stratified by ADRD case/control status
   to preserve class proportions.

4) Fixed data split across multi-seed runs:
   - When running multiple seeds (robustness), the TRAIN/VAL/TEST split is held FIXED
     using --split_seed, while --seed controls RNG initialization for training only.

Environment (tested)
--------------------
- Python 3.10+
- TensorFlow/Keras 2.12+
- scikit-learn 1.3+

Inputs (expected in --data_path)
--------------------------------
- genotype_data.npy : (n_samples, n_snps) float32
    * Recommended encoding: {0,1,2} allele counts OR already scaled to [0,1].
    * If values exceed 1.0 and --auto_scale_genotypes=1 (default), the script divides by 2.0.
- labels.npy        : (n_samples,) int {0,1}
- gwas_summary.npy  : (latent_dim, 2) float32
    * Column 0: prior_mean
    * Column 1: prior_var  (must be > 0; will be clipped)

Notes on priors
---------------
The paper describes GWAS-informed priors derived from GWAS effect sizes and standard
errors. This script assumes you have already produced priors over the LATENT dimensions
(i.e., latent_dim x 2). If you have per-SNP priors (e.g., 80x2), you must map them to
latent priors (or redesign the KL term to operate in SNP/weight space) and document the mapping.

Implementation note for validation
---------------------------------
This model subclasses keras.Model and defines both train_step and test_step so that
validation loss (val_loss) is computed correctly for early stopping.
"""

from __future__ import annotations

import os
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, average_precision_score

warnings.filterwarnings("ignore")


# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_seeds(seed: int, deterministic: bool = True) -> None:
    """Set RNG seeds for reproducibility."""
    try:
        tf.keras.utils.set_random_seed(seed)  # covers Python, NumPy, TF
    except Exception:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


# -----------------------------
# Data loading
# -----------------------------
def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load genotype, labels, and latent priors arrays from preprocessed .npy files."""
    X = np.load(os.path.join(data_path, "genotype_data.npy")).astype(np.float32)
    y = np.load(os.path.join(data_path, "labels.npy")).astype(np.int32)
    gwas_summary = np.load(os.path.join(data_path, "gwas_summary.npy")).astype(np.float32)
    return X, y, gwas_summary


def maybe_scale_genotypes(X: np.ndarray, auto_scale: bool = True) -> np.ndarray:
    """
    If genotypes are allele counts {0,1,2}, map to [0,1] by dividing by 2.0.
    This helps match the sigmoid decoder output.
    """
    if not auto_scale:
        return X
    xmax = float(np.nanmax(X))
    if xmax > 1.0:
        return X / 2.0
    return X


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

    KL term computes KL( q(z|x) || p(z) ) where:
      q(z|x) = N(z_mean, diag(exp(z_log_var)))
      p(z)   = N(prior_mean, diag(prior_var))
    """

    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        prior_mean: np.ndarray,
        prior_var: np.ndarray,
        kl_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.prior_mean = tf.constant(prior_mean, dtype=tf.float32)
        self.prior_var = tf.constant(prior_var, dtype=tf.float32)
        self.kl_weight = tf.Variable(kl_weight, trainable=False, dtype=tf.float32)

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def _compute_losses(self, x: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        z_mean, z_log_var, z = self.encoder(x, training=training)
        reconstruction = self.decoder(z, training=training)

        # Reconstruction loss (MSE over inputs)
        recon_loss = tf.reduce_mean(tf.keras.losses.mse(x, reconstruction))

        # KL divergence KL(q||p) for diagonal Gaussians
        var_q = tf.exp(z_log_var)
        kl_per_dim = 0.5 * (
            tf.math.log(self.prior_var)
            - z_log_var
            + (var_q + tf.square(z_mean - self.prior_mean)) / self.prior_var
            - 1.0
        )
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=1))

        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data

        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = self._compute_losses(x, training=True)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_weight": self.kl_weight,
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        total_loss, recon_loss, kl_loss = self._compute_losses(x, training=False)

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

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
        self.anneal_epochs = max(1, int(anneal_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        weight = min(1.0, float(epoch) / float(self.anneal_epochs))
        self.vae.kl_weight.assign(weight)


# -----------------------------
# PRS derivation + scaling
# -----------------------------
def derive_raw_ddml_prs(z_mean: np.ndarray, method: str = "mean") -> np.ndarray:
    """Derive a 1D continuous score from latent posterior mean (z_mean)."""
    if method == "mean":
        return z_mean.mean(axis=1)
    raise ValueError(f"Unknown PRS derivation method: {method}")


@dataclass
class Standardizer:
    mean_: float
    std_: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / (self.std_ + 1e-8)


def fit_standardizer(train_scores: np.ndarray) -> Standardizer:
    return Standardizer(mean_=float(train_scores.mean()), std_=float(train_scores.std()))


# -----------------------------
# Split helper (fixed across seeds)
# -----------------------------
def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    split_seed: int,
) -> Dict[str, np.ndarray]:
    """
    Create stratified TRAIN/VAL/TEST splits using a fixed split_seed.
    Returns dict containing arrays and indices for reproducibility.
    """
    n_samples = X.shape[0]
    idx_all = np.arange(n_samples)

    X_train_full, X_test, y_train_full, y_test, idx_train_full, idx_test = train_test_split(
        X, y, idx_all,
        test_size=test_size,
        random_state=split_seed,
        stratify=y
    )

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_full, y_train_full, idx_train_full,
        test_size=val_size,
        random_state=split_seed,
        stratify=y_train_full
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "idx_train": idx_train,
        "X_val": X_val,
        "y_val": y_val,
        "idx_val": idx_val,
        "X_test": X_test,
        "y_test": y_test,
        "idx_test": idx_test,
    }


# -----------------------------
# One run (one seed)
# -----------------------------
def run_one_seed(args, X: np.ndarray, y: np.ndarray, gwas_summary: np.ndarray, seed: int) -> Dict:
    # Seed controls model initialization/training randomness ONLY
    set_seeds(seed, deterministic=bool(args.deterministic))

    n_samples, input_dim = X.shape

    # Fixed split across all seeds (paper alignment)
    splits = make_splits(
        X=X,
        y=y,
        test_size=args.test_size,
        val_size=args.val_size,
        split_seed=args.split_seed,
    )

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    idx_train = splits["idx_train"]

    X_val = splits["X_val"]
    y_val = splits["y_val"]
    idx_val = splits["idx_val"]

    X_test = splits["X_test"]
    y_test = splits["y_test"]
    idx_test = splits["idx_test"]

    # Validate prior dimensions
    if gwas_summary.shape[0] != args.latent_dim or gwas_summary.shape[1] < 2:
        raise ValueError(
            "gwas_summary.npy must have shape (latent_dim, 2).\n"
            f"Got {gwas_summary.shape}, latent_dim={args.latent_dim}.\n"
            "If you have SNP-level priors (e.g., 80x2), you must map them to latent priors "
            "(latent_dim x 2) and document the mapping."
        )

    prior_mean = gwas_summary[:, 0]
    prior_var = np.clip(gwas_summary[:, 1], 1e-8, None)

    # Build and train Bayesian VAE (genotype-only)
    encoder = build_encoder(input_dim=input_dim, latent_dim=args.latent_dim)
    decoder = build_decoder(output_dim=input_dim, latent_dim=args.latent_dim)

    vae = BayesianVAE(
        encoder=encoder,
        decoder=decoder,
        prior_mean=prior_mean,
        prior_var=prior_var,
        kl_weight=0.0,
        name=f"BayesianVAE_seed{seed}",
    )
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
    # Use validation_data=X_val (not (X_val, None)) for clean Keras behavior.
    vae.fit(
        X_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=X_val,
        callbacks=callbacks,
        verbose=args.verbose,
    )

    # --- Derive RAW PRS for train/val/test (deterministic from z_mean) ---
    z_mean_train, _, _ = encoder.predict(X_train, verbose=0)
    z_mean_val, _, _ = encoder.predict(X_val, verbose=0)
    z_mean_test, _, _ = encoder.predict(X_test, verbose=0)

    raw_train = derive_raw_ddml_prs(z_mean_train, method=args.prs_method)
    raw_val = derive_raw_ddml_prs(z_mean_val, method=args.prs_method)
    raw_test = derive_raw_ddml_prs(z_mean_test, method=args.prs_method)

    # --- Standardize using TRAINING ONLY (no leakage) ---
    scaler = fit_standardizer(raw_train)
    ddml_prs_train = scaler.transform(raw_train)
    ddml_prs_val = scaler.transform(raw_val)
    ddml_prs_test = scaler.transform(raw_test)

    # PRS-only evaluation on independent test set
    auc = roc_auc_score(y_test, ddml_prs_test)
    auprc = average_precision_score(y_test, ddml_prs_test)

    fpr, tpr, thr = roc_curve(y_test, ddml_prs_test)
    youden = np.argmax(tpr - fpr)
    opt_thr = float(thr[youden])
    cm = confusion_matrix(y_test, ddml_prs_test > opt_thr)

    def _cc(yy):
        cases = int(np.sum(yy == 1))
        ctrls = int(np.sum(yy == 0))
        return cases, ctrls

    tr_cases, tr_ctrls = _cc(y_train)
    va_cases, va_ctrls = _cc(y_val)
    te_cases, te_ctrls = _cc(y_test)

    print("\n============================================================")
    print(f"Seed (training RNG): {seed}")
    print(f"Split seed (fixed):  {args.split_seed}")
    print("Split sizes (stratified by ADRD status):")
    print(f"  Train: {len(y_train):,} (cases={tr_cases:,}, controls={tr_ctrls:,})")
    print(f"  Val:   {len(y_val):,} (cases={va_cases:,}, controls={va_ctrls:,})")
    print(f"  Test:  {len(y_test):,} (cases={te_cases:,}, controls={te_ctrls:,})")
    print("\n--- DDML_PRS (PRS-only) Evaluation on Independent Test Set ---")
    print(f"ROC AUC: {auc:.4f}")
    print(f"AUPRC:   {auprc:.4f}")
    print(f"Optimal threshold (Youden index): {opt_thr:.4f}")
    print("Confusion matrix (thresholded on standardized PRS):")
    print(cm)
    print("Note: Covariate-adjusted performance (age/sex/PCs/APOE genotype) is evaluated downstream.")

    # Save outputs
    out_dir = os.path.join(args.data_path, "ddml_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Save standardized PRS aligned to original sample indices
    prs_full = np.full((n_samples,), np.nan, dtype=np.float32)
    prs_full[idx_train] = ddml_prs_train.astype(np.float32)
    prs_full[idx_val] = ddml_prs_val.astype(np.float32)
    prs_full[idx_test] = ddml_prs_test.astype(np.float32)

    np.save(os.path.join(out_dir, f"DDML_PRS_standardized_full_seed{seed}.npy"), prs_full)
    np.save(os.path.join(out_dir, f"DDML_PRS_standardized_test_seed{seed}.npy"), ddml_prs_test.astype(np.float32))

    # Save split indices once (fixed), but keep per-seed file for convenience
    run_summary = {
        "seed_training_rng": seed,
        "seed_split_fixed": int(args.split_seed),
        "idx_train": idx_train.tolist(),
        "idx_val": idx_val.tolist(),
        "idx_test": idx_test.tolist(),
        "standardizer": {"mean": scaler.mean_, "std": scaler.std_},
        "metrics": {"roc_auc": float(auc), "auprc": float(auprc), "opt_threshold_youden": opt_thr},
        "confusion_matrix": cm.tolist(),
        "args": vars(args),
    }
    with open(os.path.join(out_dir, f"run_summary_seed{seed}.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    return {
        "seed": seed,
        "split_seed": int(args.split_seed),
        "roc_auc": float(auc),
        "auprc": float(auprc),
        "opt_threshold": float(opt_thr),
        "train_n": int(len(y_train)),
        "val_n": int(len(y_val)),
        "test_n": int(len(y_test)),
        "train_cases": tr_cases,
        "val_cases": va_cases,
        "test_cases": te_cases,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Bayesian VAE (genotype-only) and derive DDML_PRS (PRS-only evaluation)."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path containing genotype_data.npy, labels.npy, gwas_summary.npy")

    # Seed for training randomness (weights init, minibatch order, etc.)
    parser.add_argument("--seed", type=int, default=42,
                        help="Training RNG seed (used if --seeds not provided).")

    # Fixed split seed (must remain constant across multi-seed runs to match manuscript)
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Fixed seed for train/val/test splitting (kept constant across seeds).")

    parser.add_argument("--seeds", type=str, default="",
                        help='Optional comma-separated list of training seeds (e.g., "1,2,3,4,5").')

    parser.add_argument("--latent_dim", type=int, default=50, help="Latent dimension (per manuscript).")
    parser.add_argument("--test_size", type=float, default=1/3,
                        help="Independent test split proportion (default 1/3).")
    parser.add_argument("--val_size", type=float, default=0.10,
                        help="Validation split proportion within the training data (default 10%).")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 0.001).")
    parser.add_argument("--kl_anneal_epochs", type=int, default=20, help="KL annealing epochs (default 20).")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience (default 10).")
    parser.add_argument("--prs_method", type=str, default="mean", choices=["mean"],
                        help="How to map latent mean to scalar PRS (default: mean).")
    parser.add_argument("--auto_scale_genotypes", type=int, default=1,
                        help="If 1 (default), scale allele-count genotypes {0,1,2} to [0,1] by dividing by 2.")
    parser.add_argument("--deterministic", type=int, default=1,
                        help="Best-effort TF determinism for reproducibility (default 1).")
    parser.add_argument("--verbose", type=int, default=2, help="Keras fit verbosity (0/1/2).")
    args = parser.parse_args()

    # Load data (genotype + labels only)
    X, y, gwas_summary = load_data(args.data_path)
    X = maybe_scale_genotypes(X, auto_scale=bool(args.auto_scale_genotypes))

    # Parse training seeds
    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [int(args.seed)]

    all_runs: List[Dict] = []
    for s in seeds:
        out = run_one_seed(args, X, y, gwas_summary, seed=s)
        all_runs.append(out)

    # Multi-run summary 
    if len(all_runs) > 1:
        aucs = np.array([r["roc_auc"] for r in all_runs], dtype=float)
        auprcs = np.array([r["auprc"] for r in all_runs], dtype=float)
        print("\n================ Multi-run summary ================")
        print(f"Training seeds: {seeds}")
        print(f"Fixed split seed: {args.split_seed}")
        print(f"AUC range: {aucs.min():.4f}–{aucs.max():.4f}  (mean ± SD: {aucs.mean():.4f} ± {aucs.std(ddof=1):.4f})")
        print(f"AUPRC range: {auprcs.min():.4f}–{auprcs.max():.4f} (mean ± SD: {auprcs.mean():.4f} ± {auprcs.std(ddof=1):.4f})")

        out_dir = os.path.join(args.data_path, "ddml_outputs")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "multi_run_summary.json"), "w") as f:
            json.dump({"runs": all_runs}, f, indent=2)


if __name__ == "__main__":
    main()
# --------------------------------------------------------------------------
# End
# --------------------------------------------------------------------------

