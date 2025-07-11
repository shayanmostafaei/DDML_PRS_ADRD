import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import statsmodels.api as sm

# Files path on the UPPMAX/Bianca server  
PROJECT_ID = "sens2017519"
DATA_PATH = "/proj/sens2017519/nobackup/b2016326_nobackup/UKBB/"

# Access and load needed data  
def load_data(data_path):
    genotype_data = np.load(os.path.join(data_path, "genotype_data.npy"))  
    labels = np.load(os.path.join(data_path, "labels.npy"))  # ADRD status
    adrd_subtypes = np.load(os.path.join(data_path, "adrd_subtypes.npy"))  # AD, Vascular Dementia, Mixed/unspecified Dementia
    age = np.load(os.path.join(data_path, "age.npy"))  # Age at baseline 
    apoe4 = np.load(os.path.join(data_path, "apoe4.npy"))  # APOE4 status 
    ancestry = np.load(os.path.join(data_path, "ancestry.npy"))  # Ancestry (European vs. Non-European) 
    gwas_summary = np.load(os.path.join(data_path, "gwas_summary.npy"))  # Mean and variance of prior distributions for 80 included SNPs
    return genotype_data, labels, adrd_subtypes, age, apoe4, ancestry, gwas_summary

genotypes, labels, adrd_subtypes, age, apoe4, ancestry, gwas_summary = load_data(DATA_PATH)

# Stratified splitting by prespecified covariates 
def stratified_split(genotypes, labels, adrd_subtypes, age, apoe4, ancestry, test_size=1/3):
    combined = np.column_stack((labels, adrd_subtypes, age, apoe4, ancestry))
    return train_test_split(genotypes, labels, adrd_subtypes, age, apoe4, ancestry, test_size=test_size, stratify=combined, random_state=42)

X_train, X_test, y_train, y_test, subtypes_train, subtypes_test, age_train, age_test, apoe4_train, apoe4_test, ancestry_train, ancestry_test = stratified_split(
    genotypes, labels, adrd_subtypes, age, apoe4, ancestry
)

# Bayesian Variational Autoencoder (Bayesian_VAE)
latent_dim = 50

# Define prior distribution for SNPs using GWAS summary statistics
snp_prior_mean = gwas_summary[:, 0]  # Mean from GWAS summary
snp_prior_var = gwas_summary[:, 1]  # Variance from GWAS summary

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder
inputs = keras.Input(shape=(genotypes.shape[1],))
x = layers.Dense(512, activation="relu")(inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
z = layers.Lambda(sampling)([z_mean, z_log_var])

encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(512, activation="relu")(x)
outputs = layers.Dense(genotypes.shape[1], activation="sigmoid")(x)

decoder = keras.Model(latent_inputs, outputs, name="decoder")

# VAE Model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, snp_prior_mean, snp_prior_var, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.snp_prior_mean = snp_prior_mean
        self.snp_prior_var = snp_prior_var
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(data, reconstruction))
            
            # KL divergence using GWAS prior
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean - self.snp_prior_mean) / self.snp_prior_var - tf.exp(z_log_var)
            )
            
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(), "reconstruction_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}

vae = VAE(encoder, decoder, snp_prior_mean, snp_prior_var)
vae.compile(optimizer=keras.optimizers.Adam())

# Train model
vae.fit(X_train, epochs=100, batch_size=256, validation_data=(X_test, None))

# Extract latent space representations from the encoder
z_mean_train, _, _ = encoder.predict(X_train)
z_mean_test, _, _ = encoder.predict(X_test)

# Fully connected classification layer
classifier = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

classifier.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
classifier.fit(z_mean_train, y_train, epochs=50, batch_size=256, validation_data=(z_mean_test, y_test))

# Compute Youden index for optimal threshold
prs_predictions = classifier.predict(z_mean_test)
fpr, tpr, thresholds = roc_curve(y_test, prs_predictions)
youden_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds[youden_index]
conf_matrix = confusion_matrix(y_test, prs_predictions > optimal_threshold)

print("AUC:", roc_auc_score(y_test, prs_predictions))
print("Optimal Threshold (Youden Index):", optimal_threshold)
print("Confusion Matrix:", conf_matrix)
