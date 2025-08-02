#!/bin/bash
# Development environment setup script for Analog PDE Solver
set -euo pipefail

echo "🔧 Setting up Analog PDE Solver development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.9+ is available
print_status "Checking Python version..."
if ! python3 --version | grep -E "Python 3\.(9|10|11|12)" > /dev/null; then
    print_error "Python 3.9+ is required but not found"
    exit 1
fi
print_success "Python version check passed"

# Check for system dependencies
print_status "Checking system dependencies..."

# Check for SPICE
if ! command -v ngspice &> /dev/null; then
    print_warning "NgSpice not found. Install with:"
    print_warning "  Ubuntu/Debian: sudo apt-get install ngspice"
    print_warning "  macOS: brew install ngspice"
    print_warning "  CentOS/RHEL: sudo yum install ngspice"
else
    print_success "NgSpice found: $(which ngspice)"
fi

# Check for Verilog tools
if ! command -v iverilog &> /dev/null; then
    print_warning "Icarus Verilog not found. Install with:"
    print_warning "  Ubuntu/Debian: sudo apt-get install iverilog"
    print_warning "  macOS: brew install icarus-verilog"
else
    print_success "Icarus Verilog found: $(which iverilog)"
fi

if ! command -v verilator &> /dev/null; then
    print_warning "Verilator not found. Install with:"
    print_warning "  Ubuntu/Debian: sudo apt-get install verilator"
    print_warning "  macOS: brew install verilator"
else
    print_success "Verilator found: $(which verilator)"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and basic tools
print_status "Upgrading pip and basic tools..."
pip install --upgrade pip setuptools wheel

# Install the package in development mode
print_status "Installing package in development mode..."
pip install -e ".[dev,docs,hardware]"

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    print_status "Installing pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
    print_success "Pre-commit hooks installed"
else
    print_warning "pre-commit not found in PATH, skipping hook installation"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p temp/spice
mkdir -p docs/_build
mkdir -p benchmark_results
mkdir -p logs

# Set up environment file
if [ ! -f ".env" ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_success ".env file created. Please customize it for your setup."
else
    print_status ".env file already exists"
fi

# Run a quick test to verify installation
print_status "Running quick verification test..."
if python -c "import analog_pde_solver; print('Import successful')"; then
    print_success "Package import test passed"
else
    print_error "Package import test failed"
    exit 1
fi

# Display next steps
echo ""
print_success "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Customize .env file for your system"
echo "3. Run tests: make test"
echo "4. Start coding!"
echo ""

# Show available make targets
print_status "Available development commands:"
make help