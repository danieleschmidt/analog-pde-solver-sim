# Multi-stage production Dockerfile for Analog PDE Solver
FROM python:3.13.7-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set up working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development and hardware dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ngspice \
    iverilog \
    verilator \
    gtkwave \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY --chown=appuser:appuser . .

# Install package in development mode
RUN pip install -e ".[dev,docs,hardware]"

USER appuser
CMD ["bash"]

# Production stage
FROM base as production

# Only copy necessary files
COPY --chown=appuser:appuser analog_pde_solver/ ./analog_pde_solver/
COPY --chown=appuser:appuser pyproject.toml README.md LICENSE ./

# Install package
RUN pip install --no-cache-dir .

# Create directories for runtime
RUN mkdir -p /app/temp /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import analog_pde_solver; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "analog_pde_solver"]

# Hardware simulation stage
FROM development as hardware

# Additional hardware simulation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    octave \
    gnuplot \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Configure X11 forwarding for GUI tools
ENV DISPLAY=:99

# Install additional hardware packages
RUN pip install --no-cache-dir \
    cocotb>=1.6.0 \
    cocotb-bus>=0.2.0 \
    myhdl>=0.11

USER appuser
CMD ["bash"]