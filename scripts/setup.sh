#!/bin/bash
# Swimming Pool Detection System - Unix/Linux Setup Script
# Author: Swimming Pool Detection Team
# Date: 2026-01-02

set -e

echo "============================================================"
echo "  Swimming Pool Detection System - Setup"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}[Step $1/9]${NC} $2"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Step 1: Check Python installation
print_step "1" "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"

# Check Python version is 3.8+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher is required."
    exit 1
fi
echo ""

# Step 2: Create virtual environment
print_step "2" "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    print_success "Virtual environment created."
fi
echo ""

# Step 3: Activate virtual environment
print_step "3" "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated."
echo ""

# Step 4: Upgrade pip
print_step "4" "Upgrading pip..."
pip install --upgrade pip
echo ""

# Step 5: Install dependencies
print_step "5" "Installing dependencies..."
pip install -r requirements.txt
print_success "Dependencies installed."
echo ""

# Step 6: Create directory structure
print_step "6" "Creating directory structure..."
python scripts/create_directories.py
echo ""

# Step 7: Download datasets (optional)
print_step "7" "Dataset download..."
echo "To download datasets, you need to configure API credentials:"
echo "  - Kaggle: Place kaggle.json in ~/.kaggle/"
echo "  - Roboflow: Set ROBOFLOW_API_KEY environment variable"
echo ""
echo "Skipping automatic download. Run manually:"
echo "  python datasets/download_dataset.py"
echo ""

# Step 8: Set permissions
print_step "8" "Setting script permissions..."
chmod +x scripts/*.py 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true
print_success "Permissions set."
echo ""

# Step 9: Verify installation
print_step "9" "Verifying installation..."
python scripts/verify_setup.py || {
    print_warning "Some components may not be properly configured."
    echo "Please review the verification output above."
}
echo ""

echo "============================================================"
echo "  Setup Complete"
echo "============================================================"
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To start training:"
echo "  python training/train.py --config config/training_config.yaml"
echo ""
echo "To run inference:"
echo "  python inference/detect_pools.py --input path/to/image.jpg --model weights/best.pt"
echo ""
