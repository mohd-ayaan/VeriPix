#!/bin/bash
# VeriPix Environment Setup Script

echo "=========================================="
echo "VeriPix Environment Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $python_version"

if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "Error: Python 3.8+ required"
    exit 1
fi

# Check CUDA availability
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep -oP 'release \K[0-9.]+')
    echo "CUDA version: $cuda_version"
else
    echo "Warning: CUDA not found. CPU-only mode will be used."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Create directory structure
echo "Creating project directories..."
mkdir -p data/raw/casia_v2/{authentic,tampered}
mkdir -p data/raw/comofod/{original,forged}
mkdir -p data/processed/{train,val,test}
mkdir -p data/masks/{train,val,test}
mkdir -p checkpoints/{classifier,localizer}
mkdir -p logs/{classifier,localizer}
mkdir -p results/{plots,metrics,predictions}

# Verify installation
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "=========================================="
echo "Setup complete! Activate environment with:"
echo "source venv/bin/activate"
echo "=========================================="
