# ------------------------------------------------------------
# Build GWAS prior table for the 80-SNP DDML_PRS panel
# Source: Jansen et al. summary statistics provided after excluding UKB
# Conventions:
#   - Genome build: GRCh37/hg19
#   - A1 is effect allele
#   - BETA is log-odds effect aligned to A1
#   - VAR = SE^2
# ------------------------------------------------------------

# Load the RData containing GWAS summary statistics 
load("Jansen_UKB_excluded_sumstats.RData") 

stopifnot(exists("data_P_value"))
stopifnot(all(c("CHR","BP","SNP","A1","A2","EAF","BETA","SE","P","Nsum","Neff") %in% names(data_P_value)))

# SNP list (80 SNPs) selected from Bellenguez-derived panel
snps80 <- read.table("snps_80.txt", header=TRUE, stringsAsFactors=FALSE)$SNP
snps80 <- unique(snps80)

# Subset to 80 SNPs
priors <- subset(data_P_value, SNP %in% snps80)

# Report missing SNPs (if any)
missing_snps <- setdiff(snps80, priors$SNP)
if (length(missing_snps) > 0) {
  message("WARNING: The following SNPs from snps_80.txt were not found in data_P_value:")
  message(paste(missing_snps, collapse = ", "))
}

# Keep required fields for priors
priors_out <- priors[, c("CHR","BP","SNP","A1","A2","EAF","BETA","SE","P","Nsum","Neff")]

# Derived variance for Bayesian priors
priors_out$VAR <- priors_out$SE^2

# NOTE: DDML_PRS uses BETA (not BETA_CONS) as the prior mean in the main analyses.
# BETA_CONS is provided for optional sensitivity analyses only and is not used by default.

# Optional conservative prior (for sensitivity analyses)
# Set mean=0 for non-genome-wide-significant SNPs in the prior source
priors_out$BETA_CONS <- ifelse(priors_out$P < 5e-8, priors_out$BETA, 0)

# Add metadata columns (helps transparency when shared)
priors_out$BUILD <- "hg19"
priors_out$BETA_SCALE <- "log_odds"
priors_out$EFFECT_ALLELE <- "A1"

# Order by chromosome/position
priors_out <- priors_out[order(priors_out$CHR, priors_out$BP), ]

# Save
write.csv(priors_out, "GWAS_sumstate_without_ukb.csv", row.names=FALSE)
message("Saved: GWAS_sumstate_without_ukb.csv")
