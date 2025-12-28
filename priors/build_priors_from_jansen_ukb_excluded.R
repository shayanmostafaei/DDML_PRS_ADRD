# ------------------------------------------------------------
# build_priors_from_jansen_ukb_excluded.R
#
# Build SNP-level GWAS prior table for the 80-SNP DDML_PRS panel
# Source: Jansen et al. GWAS summary statistics provided after excluding UK Biobank (UKB)
#
# Conventions:
#   - Genome build: GRCh37 / hg19
#   - A1 is effect allele
#   - BETA is log-odds effect aligned to A1
#   - SE is the standard error of BETA
#   - VAR (prior variance) = SE^2
#
# IMPORTANT (alignment with this repository's Python VAE):
#   - This script produces *SNP-level* priors (80 SNPs).
#   - The current DDML_PRS Python implementation expects *latent-dimension* priors
#     (latent_dim × 2) in gwas_summary.npy.
#   - Therefore, the CSV created here is primarily for transparency, record-keeping,
#     and for pipelines that use SNP-level priors directly (or if you implement an
#     explicit mapping from SNP priors → latent priors and document it).
# ------------------------------------------------------------

options(stringsAsFactors = FALSE)

# -----------------------------
# User-configurable file paths
# -----------------------------
INPUT_RDATA <- "Jansen_UKB_excluded_sumstats.RData"
SNP_LIST    <- "snps_80.txt"
OUT_CSV     <- "GWAS_sumstate_without_ukb_80snps.csv"

# APOE window used throughout this repo (hg19/GRCh37): chr19:44–46 Mb
APOE_CHR <- 19
APOE_START_BP <- 44e6
APOE_END_BP   <- 46e6

# -----------------------------
# Load and validate inputs
# -----------------------------
if (!file.exists(INPUT_RDATA)) stop("Missing input: ", INPUT_RDATA)
if (!file.exists(SNP_LIST)) stop("Missing SNP list: ", SNP_LIST)

load(INPUT_RDATA)

# Expect a data.frame named data_P_value (as in your original script)
if (!exists("data_P_value")) stop("Expected object 'data_P_value' not found in ", INPUT_RDATA)

required_cols <- c("CHR","BP","SNP","A1","A2","EAF","BETA","SE","P","Nsum","Neff")
missing_cols <- setdiff(required_cols, names(data_P_value))
if (length(missing_cols) > 0) {
  stop("data_P_value is missing required columns: ", paste(missing_cols, collapse = ", "))
}

# Read 80-SNP list (Bellenguez-derived panel used as genotype input)
snps80 <- read.table(SNP_LIST, header = TRUE)$SNP
snps80 <- unique(snps80)

if (length(snps80) != 80) {
  message("NOTE: snps_80.txt contains ", length(snps80), " unique SNP IDs (expected 80). Continuing.")
}

# -----------------------------
# Subset to the 80 SNPs
# -----------------------------
priors <- subset(data_P_value, SNP %in% snps80)

# Report missing SNPs (if any)
missing_snps <- setdiff(snps80, priors$SNP)
if (length(missing_snps) > 0) {
  message("WARNING: The following SNPs from snps_80.txt were not found in data_P_value (UKB-excluded Jansen):")
  message(paste(missing_snps, collapse = ", "))
}

# If duplicates exist per SNP, keep the first after ordering by smallest P (most significant)
# (this is a conservative de-duplication choice; adjust if your data structure differs)
priors <- priors[order(priors$SNP, priors$P), ]
priors <- priors[!duplicated(priors$SNP), ]

# Keep required fields for priors
priors_out <- priors[, required_cols]

# -----------------------------
# Derived prior variance and optional conservative mean
# -----------------------------
priors_out$VAR <- priors_out$SE^2

# NOTE (paper alignment):
# DDML_PRS uses BETA as prior mean in main analyses.
# BETA_CONS (below) is OPTIONAL and can be used in sensitivity analyses.
priors_out$BETA_CONS <- ifelse(priors_out$P < 5e-8, priors_out$BETA, 0)

# -----------------------------
# Add transparent metadata fields
# -----------------------------
priors_out$BUILD <- "hg19"
priors_out$BETA_SCALE <- "log_odds"
priors_out$EFFECT_ALLELE <- "A1"
priors_out$VAR_DEF <- "SE^2"
priors_out$SOURCE <- "Jansen et al. (2019) UKB-excluded summary statistics (author-provided)"

# Flag APOE region membership using repo-wide definition
priors_out$IN_APOE_REGION_44_46MB <- (
  priors_out$CHR == APOE_CHR &
    priors_out$BP >= APOE_START_BP &
    priors_out$BP <= APOE_END_BP
)

# -----------------------------
# Order output
# -----------------------------
# Prefer ordering by chromosome and position (stable, interpretable)
priors_out <- priors_out[order(priors_out$CHR, priors_out$BP), ]

# -----------------------------
# Save
# -----------------------------
write.csv(priors_out, OUT_CSV, row.names = FALSE)
message("Saved: ", OUT_CSV)

# Brief reminder printed to console (prevents future confusion)
message("\nReminder:")
message(" - This CSV is SNP-level (80 SNPs).")
message(" - The current DDML_PRS Python VAE expects latent-dimension priors (latent_dim×2) in gwas_summary.npy.")
message(" - If you implement an SNP→latent prior mapping, document it and keep it fixed for reproducibility.")
