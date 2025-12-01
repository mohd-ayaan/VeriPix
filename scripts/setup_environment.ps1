# VeriPix Environment Setup Script (PowerShell)
Write-Host "=========================================="
Write-Host "VeriPix Environment Setup"
Write-Host "=========================================="

# Check Python version
$python_version = (python --version) -replace '[^\d\.]', ''
Write-Host "Python version: $python_version"

if ([version]$python_version -lt [version]"3.8") {
    Write-Host "Error: Python 3.8+ required"
    exit 1
}

# CUDA check (looks for nvcc)
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) {
    $cuda_version = (& nvcc --version | Select-String "release" | ForEach-Object { $_.ToString() -replace '.*release ', '' -replace ',.*', '' })
    Write-Host "CUDA version: $cuda_version"
} else {
    Write-Host "Warning: CUDA not found. CPU-only mode will be used."
}

# Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust index-url if needed)
Write-Host "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
Write-Host "Installing other dependencies..."
pip install -r requirements.txt

# Create directories (OneDrive path as per your prompt)
$dirs = @(
    "data/raw/casia_v2/authentic",
    "data/raw/casia_v2/tampered",
    "data/raw/comofod/original",
    "data/raw/comofod/forged",
    "data/processed/train",
    "data/processed/val",
    "data/processed/test",
    "data/masks/train",
    "data/masks/val",
    "data/masks/test",
    "checkpoints/classifier",
    "checkpoints/localizer",
    "logs/classifier",
    "logs/localizer",
    "results/plots",
    "results/metrics",
    "results/predictions"
)
foreach ($dir in $dirs) { New-Item -ItemType Directory -Force -Path $dir }

# Verify installation
Write-Host "Verifying PyTorch installation..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host "=========================================="
Write-Host "Setup complete! Activate environment with:"
Write-Host ".\venv\Scripts\Activate.ps1"
Write-Host "=========================================="
