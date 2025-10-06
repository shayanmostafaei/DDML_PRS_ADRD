"""
DDML_PRS: Deep Data-Driven Machine Learning-based Polygenic Risk Score for ADRD
Author: Shayan Mostafaei et al.
Repository: https://github.com/shayanmostafaei/DDML_PRS_ADRD
Environment: Python 3.10, TensorFlow 2.12.0, Keras 2.12.0, scikit-learn 1.3.0
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Project configuration and data paths
# --------------------------------------------------------------------------

PROJECT_ID = "sens2017519"
DATA_PATH = "/proj/sens2017519/nobackup/b2016326_nobackup/UKBB/"

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_data(data_path):
    """Load genotype and phenotype data from preprocessed .npy files."""
    genotype_data = np.load(os.path.join(data_path, "genotype_data.npy"))
    labels = np.load(os.path.join(data_path, "labels.npy"))                 # ADRD status
    adrd_subtypes = np.load(os.path.join(data_path, "adrd_subtypes.npy"))   # AD, VaD, mixed, etc.
    age = np.load(os.path.join(data_path, "age.npy"))                       # Age at baseline
    apoe4 = np.load(os.path.join(data_path, "apoe4.npy"))                   # APOE4 status
    ancestry = np.load(os.path.join(data_path, "ancestry.npy"))             # Ancestry (e.g., European)
    gwas_summary = np.load(os.path.join(data_path, "gwas_summary.npy"))     # Mean & variance of prior (80 SNPs)
    return genotype_data, labels, adrd_subtypes, age, apoe4, ancestry, gwas_summary


genotypes, labels, adrd_subtypes, age, apoe4, ancestry, gwas_summary = load_data(DATA_PATH)

# --------------------------------------------------------------------------
# Stratified split
# --------------------------------------------------------------------------

def stratified_split(genotypes, labels, adrd_subtypes, age, apoe4, ancestry, test_size=1/3):
    """Split data ensuring balanced distribution of key covariates."""
    combined = np.column_stack((labels, adrd_subtypes, age, apoe4, ancestry))
    return train_test_split(
        genotypes, labels, adrd_subtypes, age, apoe4, ancestry,
        test_size=test_size, stratify=combined, random_state=42
    )

X_train, X_test, y_train, y_test, subtypes_train, subtypes_test, age_train, age_test, apoe4_train, apoe4_test, ancestry_train, ancestry_test = stratified_split(
    genotypes, labels, adrd_subtypes, age, apoe4, ancestry
)

# --------------------------------------------------------------------------
# Bayesian Variational Autoencoder (VAE)
# --------------------------------------------------------------------------

latent_dim = 50  # per manuscript specification

# Priors from GWAS summary statistics (mean, variance)
snp_prior_mean = gwas_summary[:, 0]
snp_prior_var = gwas_summary[:, 1]

def sampling(args):
    """Reparameterization trick."""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder network
inputs = keras.Input(shape=(genotypes.shape[1],))
x = layers.Dense(512, activation="relu")(inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder network
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(512, activation="relu")(x)
outputs = layers.Dense(genotypes.shape[1], activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, outputs, name="decoder")

# --------------------------------------------------------------------------
# Custom VAE model
# --------------------------------------------------------------------------

class BayesianVAE(keras.Model):
    """Bayesian VAE incorporating GWAS priors in KL divergence."""

    def __init__(self, encoder, decoder, snp_prior_mean, snp_prior_var, **kwargs):
        super(BayesianVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.snp_prior_mean = tf.constant(snp_prior_mean, dtype=tf.float32)
        self.snp_prior_var = tf.constant(snp_prior_var, dtype=tf.float32)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(data, reconstruction)
            )

            # KL divergence term using GWAS prior (Bayesian regularization)
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var
                - tf.square(z_mean - self.snp_prior_mean) / self.snp_prior_var
                - tf.exp(z_log_var)
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Instantiate model
vae = BayesianVAE(encoder, decoder, snp_prior_mean, snp_prior_var)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# --------------------------------------------------------------------------
# Training the VAE
# --------------------------------------------------------------------------

vae.fit(
    X_train,
    epochs=100,
    batch_size=256,
    validation_data=(X_test, None),
    verbose=2
)

# --------------------------------------------------------------------------
# Extract latent features and train classifier
# --------------------------------------------------------------------------

"""
DDML_PRS: Deep Data-Driven Machine Learning-based Polygenic Risk Score for ADRD
Author: Shayan Mostafaei et al.
Repository: https://github.com/shayanmostafaei/DDML_PRS_ADRD
Environment: Python 3.10, TensorFlow 2.12.0, Keras 2.12.0, scikit-learn 1.3.0
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Project configuration and data paths
# --------------------------------------------------------------------------

PROJECT_ID = "sens2017519"
DATA_PATH = "/proj/sens2017519/nobackup/b2016326_nobackup/UKBB/"

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_data(data_path):
    """Load genotype and phenotype data from preprocessed .npy files."""
    genotype_data = np.load(os.path.join(data_path, "genotype_data.npy"))
    labels = np.load(os.path.join(data_path, "labels.npy"))                 # ADRD status
    adrd_subtypes = np.load(os.path.join(data_path, "adrd_subtypes.npy"))   # AD, VaD, mixed, etc.
    age = np.load(os.path.join(data_path, "age.npy"))                       # Age at baseline
    apoe4 = np.load(os.path.join(data_path, "apoe4.npy"))                   # APOE4 status
    ancestry = np.load(os.path.join(data_path, "ancestry.npy"))             # Ancestry (e.g., European)
    gwas_summary = np.load(os.path.join(data_path, "gwas_summary.npy"))     # Mean & variance of prior (80 SNPs)
    return genotype_data, labels, adrd_subtypes, age, apoe4, ancestry, gwas_summary


genotypes, labels, adrd_subtypes, age, apoe4, ancestry, gwas_summary = load_data(DATA_PATH)

# --------------------------------------------------------------------------
# Stratified split (prespecified covariates)
# --------------------------------------------------------------------------

def stratified_split(genotypes, labels, adrd_subtypes, age, apoe4, ancestry, test_size=1/3):
    """Split data ensuring balanced distribution of key covariates."""
    combined = np.column_stack((labels, adrd_subtypes, age, apoe4, ancestry))
    return train_test_split(
        genotypes, labels, adrd_subtypes, age, apoe4, ancestry,
        test_size=test_size, stratify=combined, random_state=42
    )

X_train, X_test, y_train, y_test, subtypes_train, subtypes_test, age_train, age_test, apoe4_train, apoe4_test, ancestry_train, ancestry_test = stratified_split(
    genotypes, labels, adrd_subtypes, age, apoe4, ancestry
)

# --------------------------------------------------------------------------
# Bayesian Variational Autoencoder (VAE)
# --------------------------------------------------------------------------

latent_dim = 50  # per manuscript specification

# Priors from GWAS summary statistics (mean, variance)
snp_prior_mean = gwas_summary[:, 0]
snp_prior_var = gwas_summary[:, 1]

def sampling(args):
    """Reparameterization trick."""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder network
inputs = keras.Input(shape=(genotypes.shape[1],))
x = layers.Dense(512, activation="relu")(inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder network
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(512, activation="relu")(x)
outputs = layers.Dense(genotypes.shape[1], activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, outputs, name="decoder")

# --------------------------------------------------------------------------
# Custom VAE model
# --------------------------------------------------------------------------

class BayesianVAE(keras.Model):
    """Bayesian VAE incorporating GWAS priors in KL divergence."""

    def __init__(self, encoder, decoder, snp_prior_mean, snp_prior_var, **kwargs):
        super(BayesianVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.snp_prior_mean = tf.constant(snp_prior_mean, dtype=tf.float32)
        self.snp_prior_var = tf.constant(snp_prior_var, dtype=tf.float32)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(data, reconstruction)
            )

            # KL divergence term using GWAS prior (Bayesian regularization)
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var
                - tf.square(z_mean - self.snp_prior_mean) / self.snp_prior_var
                - tf.exp(z_log_var)
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Instantiate model
vae = BayesianVAE(encoder, decoder, snp_prior_mean, snp_prior_var)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# --------------------------------------------------------------------------
# Training the VAE
# --------------------------------------------------------------------------

vae.fit(
    X_train,
    epochs=100,
    batch_size=256,
    validation_data=(X_test, None),
    verbose=2
)

# --------------------------------------------------------------------------
# Extract latent features and train PRS regression model
# --------------------------------------------------------------------------

z_mean_train, _, _ = encoder.predict(X_train)
z_mean_test, _, _ = encoder.predict(X_test)

# Map 50D latent mean vector → continuous scalar PRS via regression head
# Using linear activation to produce continuous PRS values (not probabilities)
classifier = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="linear", name="DDML_PRS")
])

classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mae"]
)

classifier.fit(
    z_mean_train, y_train,
    epochs=50,
    batch_size=256,
    validation_data=(z_mean_test, y_test),
    verbose=2
)

# --------------------------------------------------------------------------
# Evaluation metrics (AUC, AUPRC, Youden Index)
# --------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve

# Continuous PRS values (regression outputs)
prs_continuous = classifier.predict(z_mean_test).flatten()

# Standardize PRS (Z-score normalization)
prs_standardized = (prs_continuous - np.mean(prs_continuous)) / np.std(prs_continuous)

# For diagnostic metrics, treat higher PRS = higher ADRD risk
fpr, tpr, thresholds = roc_curve(y_test, prs_standardized)
auc_score = roc_auc_score(y_test, prs_standardized)

# Precision-Recall and AUPRC
precision, recall, pr_thresholds = precision_recall_curve(y_test, prs_standardized)
auprc = average_precision_score(y_test, prs_standardized)

# Compute Youden Index (optimal cutoff)
youden_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds[youden_index]
conf_matrix = confusion_matrix(y_test, prs_standardized > optimal_threshold)

print("\n--- DDML_PRS Model Evaluation ---")
print(f"AUC (ROC): {auc_score:.3f}")
print(f"AUPRC: {auprc:.3f}")
print(f"Optimal Threshold (Youden Index): {optimal_threshold:.3f}")
print("Confusion Matrix:")
print(conf_matrix)

# --------------------------------------------------------------------------
# End
# --------------------------------------------------------------------------
