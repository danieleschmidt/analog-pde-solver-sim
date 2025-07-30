#!/bin/bash
# Development environment setup script

set -e

echo "🔧 Setting up analog-pde-solver-sim development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
if [[ $(echo "$python_version >= 3.9" | bc -l) -ne 1 ]]; then
    echo "❌ Python 3.9+ required. Found: $python_version"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt
pip install -e ".[dev]"

# Check system dependencies
echo "🔍 Checking system dependencies..."

check_command() {
    if command -v "$1" &> /dev/null; then
        echo "✅ $1 found"
    else
        echo "⚠️ $1 not found - install with system package manager"
    fi
}

check_command ngspice
check_command iverilog
check_command verilator

# Run initial tests
echo "🧪 Running initial tests..."
python -c "import numpy, scipy, matplotlib, torch; print('✅ Core dependencies working')"

echo "🎉 Development environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To format code:"
echo "  black ."