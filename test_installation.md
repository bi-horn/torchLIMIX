# Installation via pip

## Create new environment
conda create -n torchlimix-env python=3.11 --no-default-packages -y
conda activate torchlimix-env

## Install torch via conda
pip3 install torch --index-url https://download.pytorch.org/whl/cu121

## Install your package (run from your project root)
pip3 install .

## Test
python3 -c "import torchlimix; print('Import OK')"


------------------------------------------------

# Installation via conda
# Create the environment and installs everything
# This single command installs Python, Conda's PyTorch, and triggers pip to install your package
conda env create -f environment.yml

# Activate the new environment
conda activate torchlimix-env

# Test the Python import
python3 -c "import torchlimix; print('Import OK')"

# Test the CLI (since we fixed the JSON config path earlier!)
torchlimix --help

# Clean up
conda deactivate
conda remove -n torchlimix-env --all -y