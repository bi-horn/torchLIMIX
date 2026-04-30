# Installation

## Option 1: Installation via Conda (Recommended)

```bash
# Create the environment (pick the file for your platform)
conda env create -f environment_macos.yml    # macOS
conda env create -f environment.yml          # Linux / Windows (CUDA)

conda activate torchlimix-env

# Test
python3 -c "import torchlimix; print('Import OK')"
torchlimix --help
```

## Option 2: Installation via pip

```bash
conda create -n torchlimix-env python=3.11 --no-default-packages -y
conda activate torchlimix-env

# Install PyTorch for your platform
pip3 install torch --index-url https://download.pytorch.org/whl/cu121  # Linux/Windows (CUDA 12.1)
pip3 install torch                                                      # macOS

# Install torchlimix (run from project root)
pip3 install .

# Test
python3 -c "import torchlimix; print('Import OK')"
torchlimix --help
```

## Clean up

```bash
conda deactivate
conda remove -n torchlimix-env --all -y
```