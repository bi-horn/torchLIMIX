# TorchLIMIX

GPU-accelerated multivariate GWAS, variance decomposition, and prediction using PyTorch.

TorchLIMIX is a PyTorch implementation of [LIMIX](https://github.com/limix/limix) and [GLIMIX-core](https://github.com/limix/glimix-core), optimized for multivariate genetic analyses on GPU (see [NOTICE](./NOTICE) for more details).

## Features

- **GPU acceleration** — PyTorch backend for large-scale datasets
- **Flexible input formats** — PLINK binary, HDF5, CSV, TSV, TXT with automatic delimiter detection
- **Command-line interface** — Run analyses directly from the terminal
- **Multivariate GWAS** — Test for common, any-effect, or phenotype-specific genetic effects
- **Variance decomposition** — Partition phenotypic variance into genetic and environmental components
- **Prediction** — Genomic prediction via multi-trait BLUP, with internal cross-validation or external genotype files

## Installation

`torchlimix` requires PyTorch. Because PyTorch has specific hardware and CUDA dependencies, we highly recommend installing via **Conda** to manage these dependencies automatically.

> **No CUDA GPU?** torchlimix runs on CPU but is faster with a CUDA-capable GPU. For free GPU access, see the [Google Colab](#running-on-google-colab) section below.

### Option 1: Installation via Conda (Recommended)

Conda will automatically handle installing Python, PyTorch (with the correct binaries), and `torchlimix`.

```bash
# 1. Clone the repository and navigate into it
git clone https://github.com/bi-horn/torchlimix.git
cd torchlimix

# 2. Create the environment (pick the file for your platform)
conda env create -f environment.yml          # Linux / Windows (with CUDA support)
conda env create -f environment_macos.yml    # macOS (Apple Silicon / Intel)

# 3. Activate the new environment
conda activate torchlimix-env

# 4. Test the installation
torchlimix --help
```

> **Which file?** The Linux/Windows environment includes `pytorch-cuda`, which
> enables GPU acceleration via CUDA. The macOS environment omits it since macOS
> does not support CUDA — PyTorch will run on CPU by default.

### Option 2: Installation via pip

If you prefer to manage your own environments and install PyTorch manually, you can install torchlimix directly using pip.

```bash
# 1. Create and activate a fresh environment
conda create -n torchlimix-env python=3.11 -y
conda activate torchlimix-env

# 2. Install PyTorch for your platform
pip install torch --index-url https://download.pytorch.org/whl/cu121  # Linux/Windows (CUDA 12.1)
pip install torch                                                      # macOS

# See https://pytorch.org/get-started/locally/ for other configurations

# 3. Install torchlimix directly from GitHub
pip install git+https://github.com/bi-horn/torchlimix.git

# 4. Test the installation
torchlimix --help
```

## Running on Google Colab

For easy GPU access, use torchlimix in Google Colab. See the notebook `run_torchLIMIX.ipynb` in [notebooks](./notebooks) for instructions.

## Quick Start

### GWAS Analysis

**Run with simulated data** (uses the Horton HapMap panel with 10% MAF filtering):

```bash
torchlimix \
    --simulated \
    --eta 0.3 \
    --analysis gwas \
    --test_type any_vs_common \
    --output_directory ./results
```

> The default dataset is `thaliana_horton`, which is the only dataset with
> bundled genotype data. Keep this default when running simulations.

**Run with your own data:**

```bash
torchlimix \
    --pheno_path ~/phenotypes.csv \
    --geno_path ~/genotypes \
    --dset my_study \
    --analysis gwas \
    --test_type any_vs_common \
    --output_directory ./results
```

> **PLINK format:** If your genotype data is in PLINK format (bim/bed/fam),
> pass the path **without** the file extension — e.g. `~/genotypes` rather than
> `~/genotypes.bed`.
>
> **Reserved dataset names:** Do not use `thaliana_horton` or `thaliana_1001`
> as your `--dset` name — these are reserved for simulation analysis.

**Run with covariates included as fixed effects:**

```bash
torchlimix \
    --pheno_path ~/phenotypes.tsv \
    --geno_path ~/genotypes \
    --cov_path ~/covariates.csv \
    --include_covariates_in_model \
    --dset my_study \
    --analysis gwas \
    --test_type any_vs_common \
    --output_directory ./results
```

### Variance Decomposition

```bash
torchlimix \
    --pheno_path ~/phenotypes.csv \
    --geno_path ~/genotypes \
    --dset my_study \
    --analysis vardec \
    --output_directory ./results
```

### Prediction

Prediction supports two scenarios: evaluating accuracy on a held-out test split, or predicting phenotypes for new individuals from an external genotype file.

**Scenario A — Internal train/test split** (ground truth available):

```bash
torchlimix \
    --pheno_path ~/phenotypes.csv \
    --geno_path ~/genotypes \
    --dset my_study \
    --analysis prediction \
    --train_pct 0.9 \
    --output_directory ./results
```

The model trains on 90% of the samples and predicts the remaining 10%. Samples are assigned to splits by random permutation with a fixed seed (default 42), so results are reproducible. The test proportion is `1 - train_pct - val_pct`; setting `--val_pct 0.0` gives a simple train/test split. Prediction metrics (MSE, correlation, R²) are computed against the held-out ground truth.

**Scenario B — External genotype file** (no ground truth):

```bash
torchlimix \
    --pheno_path ~/phenotypes.csv \
    --geno_path ~/genotypes \
    --predict_geno_path ~/new_samples.npz \
    --dset my_study \
    --analysis prediction \
    --output_directory ./results
```

The model trains on 100% of the original dataset and predicts phenotypes for the new individuals. Only predictions and uncertainty estimates are saved; no accuracy metrics are computed. The external genotype file must contain the same SNPs (in the same order) as the training genotypes.

## Input File Formats

### Genotypes

Genotype files are used both for the training panel (`--geno_path`) and for external prediction samples (`--predict_geno_path`).

| Format | Extension(s) | Notes |
|--------|-------------|-------|
| PLINK binary | `.bed` (+ `.bim`, `.fam`) | Recommended for large panels. Provide the path without extension. |
| HDF5 | `.h5`, `.hdf5` | Datasets: `genotypes` (n × s), optionally `fid` and `iid`. |
| Delimited text | `.csv`, `.tsv`, `.txt` | First two columns are `fid`/`iid`, remaining columns are SNP dosages. Delimiter is auto-detected. |
| NumPy archive | `.npz` | **External prediction only.** Keys: `genotypes` (n × s), optionally `fid`/`iid`. |

PLINK binary example (provide path without extension):

```bash
--geno_path ~/data/genotypes  # expects genotypes.bed, genotypes.bim, genotypes.fam
```

If `fid`/`iid` arrays are omitted from HDF5 or NPZ files, sample IDs are auto-generated as 1, 2, 3, …

Missing values (NaN) in genotype files are imputed with the per-SNP column mean.

### Phenotypes

A tabular file (`.csv`, `.tsv`, `.txt`) with `fid` and `iid` columns followed by one or more trait columns:

```
fid    iid    trait1    trait2    trait3
714    714    1.23      2.45      0.87
312    312    1.56      2.12      1.02
```

Missing values are imputed with the column mean. If column names `fid`/`iid` are not found, the first two columns are assumed to be the sample identifiers.

### Covariates (optional)

Same tabular format as phenotypes. Covariate columns follow the `fid` and `iid` columns:

```
fid    iid    PC1      PC2      PC3
714    714    0.12    -0.34     0.05
312    312    0.23     0.15    -0.11
```

Covariates can be either regressed out of the phenotype before model fitting (`--regress_out_covariates`) or included as fixed effects in the model (`--include_covariates_in_model`). These two options are mutually exclusive.

### Batch Effects (optional)

Same tabular format. Categorical batch variables are regressed out of the phenotype when `--regress_out_batch_effects` is set.

### SNP Annotations (optional)

A tabular file with `chrom` and `pos` columns. Used for Manhattan plots and regional analyses. If not provided, annotations are constructed from the PLINK `.bim` file when available.

### Delimiter Detection

For all text formats (`.csv`, `.tsv`, `.txt`), the delimiter is auto-detected by scanning the first line for tab, comma, semicolon, and space characters.

## Command-Line Options

### Required Arguments

These are required when running on real data. For simulations, `--pheno_path` and `--geno_path` are not needed since data is generated internally.

| Argument | Description |
|----------|-------------|
| `--pheno_path` | Path to phenotype file |
| `--geno_path` | Path to genotype file (PLINK, HDF5, or delimited text) |

### Analysis

| Argument | Default | Description |
|----------|---------|-------------|
| `--analysis` | `gwas` | Analysis type: `gwas`, `vardec`, or `prediction` |
| `--test_type` | `any_vs_common` | Hypothesis test: `common`, `any`, `specific`, `any_vs_common`, `specific_vs_common` |
| `--pheno_idx` | `0` | Trait index for phenotype-specific tests (0-indexed) |
| `--rank` | number of phenotypes | Model rank (defaults to full rank) |
| `--transformation_method` | `int` | Phenotype transformation: `int` (van der Waerden), `z_score`, or `none` |
| `--device` | GPU if available | Use CUDA GPU (`1`) or CPU (`0`) |

### Covariate Options

| Argument | Description |
|----------|-------------|
| `--cov_path` | Path to covariates file |
| `--regress_out_covariates` | Regress covariates from phenotypes before analysis |
| `--include_covariates_in_model` | Include covariates in the fixed effects matrix M |

### Batch Correction

| Argument | Description |
|----------|-------------|
| `--batch_path` | Path to batch file |
| `--regress_out_batch_effects` | Regress batch effects from phenotypes |

### Prediction Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_pct` | `1.0` | Training set proportion (0.0–1.0) |
| `--val_pct` | `0.0` | Validation set proportion (0.0–1.0) |
| `--predict_geno_path` | — | Path to external genotype file for prediction (`.bed`, `.h5`, `.hdf5`, `.csv`, `.tsv`, `.txt`, `.npz`) |

### Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--dset` | `thaliana_horton` | Dataset name (used for output directory structure) |
| `--output_directory` | `./results` | Results output directory |
| `--verbose` | off | Enable verbose output |

## Output Structure

Results are written to the output directory with the following structure.

### GWAS Output

```
output_directory/
└── gwas_results/               # or rep{XXXX}/ for simulations
    ├── null_model.csv          # Null model fixed effects (intercept + covariates)
    ├── null_model.txt          # Human-readable null model summary
    ├── log_likelihoods.csv     # Per-SNP likelihoods, LRT statistics, and p-values
    ├── beta_results.csv        # Per-SNP effect sizes (beta1, beta2) and standard errors
    ├── covariances.csv         # Fitted C0 and C1 covariance matrices
    ├── sim_params.csv          # Simulation parameters (if simulated)
    ├── optimization_stats.jsonl # Model fitting diagnostics
    ├── data_preprocessing.json # Record of corrections applied
    ├── phenotypes.csv          # Phenotype data used in analysis
    └── phenotype_comparison.txt # Before/after correction summary (if corrections applied)
```

### Null Model (`null_model.csv`)

Stores the fixed effect estimates from the null model (without SNP effects). Written once since these values are constant across all SNPs. Contains one row per covariate with beta and standard error columns for each trait:

```
covariate,beta_trait_0,se_trait_0,beta_trait_1,se_trait_1
intercept,0.000045,0.067827,0.000140,0.069242
cov_1,0.071736,0.067827,-0.028478,0.069242
```

### Likelihood Results (`log_likelihoods.csv`)

One row per SNP with log-marginal likelihoods under each hypothesis, likelihood ratio test statistics, degrees of freedom, and chi-squared p-values:

```
snp_index,lml0,lml1,lml2,lrt10,df10,pv10,lrt20,df20,pv20,lrt21,df21,pv21,scale_H0
```

### Per-SNP Effect Sizes (`beta_results.csv`)

One row per SNP. Effect sizes from H1 and H2 scans are stored as numeric columns. Multi-trait effects are expanded (e.g., `beta1_0`, `beta1_1`, `beta1_2` for three traits):

```
snp_index,beta1,beta1_se,beta2_0,beta2_se_0,beta2_1,beta2_se_1,beta2_2,beta2_se_2
```

### Variance Decomposition Output

```
output_directory/
└── vardec_results/
    ├── covariances.csv
    ├── variance_decomposition.csv
    ├── optimization_stats.jsonl
    └── sim_params.csv          # Simulation parameters (if simulated)
```

### Prediction Output

**Scenario A — Internal split** (ground truth available):

```
output_directory/
└── predictions/
    ├── predicted_phenotypes.csv # Pred, Std, and True columns per trait
    ├── metrics.json            # MSE, MAE, correlation, R² per trait
    ├── summary.json            # Predictions, CI, metrics, and metadata
    ├── data_preprocessing.json # Record of corrections applied
    └── phenotypes.csv          # Training phenotype data
```

**Scenario B — External prediction** (no ground truth):

```
output_directory/
└── predictions/
    ├── predicted_phenotypes.csv # Pred and Std columns per trait
    ├── summary.json            # Predictions, CI, and metadata (no metrics)
    ├── data_preprocessing.json # Record of corrections applied
    └── phenotypes.csv          # Training phenotype data
```

The `predicted_phenotypes.csv` file uses the `fid`/`iid` sample identifiers from the external genotype file and contains one row per predicted individual:

```
fid,iid,Trait_0_Pred,Trait_0_Std,Trait_1_Pred,Trait_1_Std
90001,90001,0.3046,2.0205,0.2768,1.8799
90002,90002,0.3302,2.0197,0.3000,1.8791
```

### File Formats

Simulation runs (identified by `rep_idx`) default to Parquet output (`.parquet` with zstd compression) for compact storage across many replicates. Real data analyses default to CSV.

## Test Types

| Test | Null (H0) | Alternative | Use case |
|------|-----------|-------------|----------|
| `common` | No effect | Same effect across all traits | Shared genetic architecture |
| `any` | No effect | Independent effects per trait | Any genetic signal |
| `specific` | No effect | Common + phenotype-specific effect | Trait-specific signals |
| `any_vs_common` | Common effect | Heterogeneous effects | Detect effect heterogeneity |
| `specific_vs_common` | Common effect | Additional phenotype-specific effect | Single trait deviation |

## Attribution

TorchLIMIX is a derivative work based on:

- **LIMIX** (Apache 2.0) — C. Lippert, D. Horta, F. P. Casale, O. Stegle
- **GLIMIX-core** (MIT) — D. Horta

See [NOTICE](./NOTICE) for full attribution details.

## License

MIT License. See [LICENSE](./LICENSE) for details.

## Authors

- Bibiana M. Horn
- Christoph Lippert