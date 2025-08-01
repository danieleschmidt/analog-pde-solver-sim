# Analog PDE Solver Development Container
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        # SPICE simulation tools
        ngspice \
        ngspice-doc \
        # Verilog tools
        iverilog \
        verilator \
        gtkwave \
        # Build tools
        build-essential \
        cmake \
        pkg-config \
        # Scientific computing
        libopenblas-dev \
        liblapack-dev \
        libfftw3-dev \
        # Graphics and visualization
        libgl1-mesa-glx \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        # Documentation tools
        pandoc \
        texlive-latex-base \
        texlive-latex-extra \
        # Development tools
        git-lfs \
        curl \
        wget \
        vim \
        htop \
        tree \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN python -m pip install --upgrade pip setuptools wheel

# Create workspace directory
WORKDIR /workspaces/analog-pde-solver-sim

# Set environment variables
ENV SPICE_LIB_DIR=/usr/share/ngspice/scripts
ENV PYTHONPATH=/workspaces/analog-pde-solver-sim

# Configure git for the container
RUN git config --global --add safe.directory /workspaces/analog-pde-solver-sim

# Create non-root user for development
USER vscode

# Install development tools in user space
RUN pip install --user pre-commit black flake8 mypy bandit safety pip-audit

# Set up shell environment
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc \
    && echo 'export SPICE_LIB_DIR=/usr/share/ngspice/scripts' >> ~/.bashrc \
    && echo 'export PYTHONPATH=/workspaces/analog-pde-solver-sim' >> ~/.bashrc

# Add helpful aliases
RUN echo 'alias ll="ls -la"' >> ~/.bashrc \
    && echo 'alias la="ls -la"' >> ~/.bashrc \
    && echo 'alias pytest-cov="pytest --cov=analog_pde_solver --cov-report=html"' >> ~/.bashrc \
    && echo 'alias run-benchmarks="python scripts/run-benchmarks.py"' >> ~/.bashrc

# Welcome message
RUN echo 'echo "🧮 Analog PDE Solver Development Environment"' >> ~/.bashrc \
    && echo 'echo "📊 SPICE: $(ngspice --version | head -1)"' >> ~/.bashrc \
    && echo 'echo "🔧 Verilog: $(iverilog -V | head -1)"' >> ~/.bashrc \
    && echo 'echo "🐍 Python: $(python --version)"' >> ~/.bashrc \
    && echo 'echo ""' >> ~/.bashrc