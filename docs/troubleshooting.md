# Troubleshooting Guide

Common issues and solutions for the analog PDE solver framework.

## Installation Issues

### Python Version Compatibility

**Problem**: `ERROR: Package requires Python >=3.9`

**Solution**: Upgrade Python or use a compatible version:
```bash
# Check current version
python3 --version

# Install Python 3.11 (Ubuntu/Debian)
sudo apt-get install python3.11 python3.11-venv

# Use specific version
python3.11 -m venv venv
```

### System Dependencies Missing

**Problem**: `ngspice: command not found` or similar

**Solution**: Install required system packages:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ngspice iverilog verilator build-essential
```

**macOS:**
```bash
brew install ngspice icarus-verilog verilator
```

**CentOS/RHEL:**
```bash
sudo yum install ngspice iverilog verilator gcc gcc-c++
```

### Virtual Environment Issues

**Problem**: `No module named 'analog_pde_solver'`

**Solution**: Ensure virtual environment is activated and package is installed:
```bash
source venv/bin/activate
pip install -e ".[dev]"
```

## Testing Issues

### Tests Failing

**Problem**: `ImportError` or missing test dependencies

**Solution**: Install development dependencies:
```bash
pip install -e ".[dev]"
# or
make dev-setup
```

### SPICE Tests Timeout

**Problem**: SPICE simulations hang or timeout

**Solution**: 
1. Check SPICE installation: `ngspice --version`
2. Increase timeout in configuration
3. Run with hardware tests disabled: `pytest -m "not hardware"`

### Slow Test Execution

**Problem**: Tests take too long to run

**Solution**: Run only fast tests:
```bash
make test-fast
# or
pytest -m "not slow and not hardware"
```

## Hardware Simulation Issues

### SPICE Simulation Errors

**Problem**: `SPICE simulation failed` or convergence errors

**Solutions**:
1. **Check model parameters**: Ensure conductance values are in valid range
2. **Adjust simulation settings**: Modify time step or solver tolerances
3. **Verify netlist**: Check generated SPICE netlist for syntax errors

```python
# Example: Adjust simulation parameters
spice_sim.set_options({
    'abstol': 1e-12,
    'reltol': 1e-6,
    'vntol': 1e-6
})
```

### Verilog Compilation Errors

**Problem**: RTL generation fails or synthesis errors

**Solutions**:
1. **Check Verilog syntax**: Use `iverilog -t null file.v`
2. **Verify tool versions**: Ensure compatible versions of synthesis tools
3. **Review generated code**: Check for unsupported constructs

### Memory Issues

**Problem**: `MemoryError` or system becomes unresponsive

**Solutions**:
1. **Reduce problem size**: Use smaller grid dimensions
2. **Enable sparse matrices**: Use sparse representations where possible
3. **Increase swap space**: Add more virtual memory

```python
# Example: Use smaller problem size
pde = PoissonEquation(domain_size=(64, 64))  # Instead of (256, 256)
```

## Performance Issues

### Slow Convergence

**Problem**: Analog solver converges slowly or not at all

**Solutions**:
1. **Adjust conductance range**: Ensure good dynamic range
2. **Tune noise parameters**: Balance noise vs accuracy
3. **Use preconditioning**: Apply matrix preconditioning techniques
4. **Check boundary conditions**: Ensure proper boundary setup

### High Power Consumption

**Problem**: Simulated power usage is higher than expected

**Solutions**:
1. **Optimize conductance mapping**: Minimize unnecessary high conductances
2. **Use dynamic precision**: Start with low precision, increase as needed
3. **Enable power gating**: Turn off unused crossbar sections

## Development Issues

### Import Errors

**Problem**: `ModuleNotFoundError` during development

**Solutions**:
1. **Install in development mode**: `pip install -e .`
2. **Check PYTHONPATH**: Ensure project root is in path
3. **Verify package structure**: Check `__init__.py` files exist

### Pre-commit Hook Failures

**Problem**: Git commits rejected by pre-commit hooks

**Solutions**:
1. **Run formatting**: `make format`
2. **Fix linting issues**: `make lint`
3. **Update hooks**: `pre-commit autoupdate`
4. **Skip hooks temporarily**: `git commit --no-verify` (not recommended)

### Type Checking Errors

**Problem**: mypy reports type errors

**Solutions**:
1. **Add type annotations**: Ensure all functions have proper types
2. **Use type ignores sparingly**: `# type: ignore` for external libraries
3. **Update mypy configuration**: Adjust strictness in `pyproject.toml`

## Platform-Specific Issues

### Windows (WSL)

**Common Issues**:
- Path separator conflicts
- Permission issues with shell scripts
- SPICE GUI applications not working

**Solutions**:
- Use WSL 2 for better compatibility
- Set execute permissions: `chmod +x scripts/*.sh`
- Use headless SPICE simulation only

### macOS

**Common Issues**:
- Homebrew package conflicts
- Xcode command line tools missing
- Permission issues with system directories

**Solutions**:
- Install Xcode command line tools: `xcode-select --install`
- Use `brew doctor` to diagnose Homebrew issues
- Avoid using system Python, use Homebrew Python instead

### Docker Issues

**Problem**: Docker container fails to start or build

**Solutions**:
1. **Update Docker**: Ensure recent Docker version
2. **Check permissions**: Ensure user can access Docker daemon
3. **Clear cache**: `docker system prune`
4. **Check disk space**: Ensure sufficient space for images

## Getting More Help

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export ANALOG_PDE_DEBUG=1
```

### Reporting Issues

When reporting issues, include:

1. **Environment details**:
   - OS and version
   - Python version
   - Package version
   - SPICE/Verilog tool versions

2. **Minimal reproduction case**:
   - Smallest code example that shows the problem
   - Input data if relevant
   - Expected vs actual behavior

3. **Error messages**:
   - Complete stack traces
   - Log files if available
   - SPICE/Verilog error outputs

### Useful Debugging Commands

```bash
# Check system dependencies
make check-deps

# Run single test with verbose output
pytest -xvs tests/unit/test_core.py::TestBasePDE::test_placeholder

# Generate detailed coverage report
pytest --cov=analog_pde_solver --cov-report=html

# Profile performance
python -m cProfile -o profile.stats your_script.py

# Check for memory leaks
valgrind --tool=memcheck python your_script.py
```

### Community Resources

- [GitHub Issues](https://github.com/yourusername/analog-pde-solver-sim/issues)
- [Discussions](https://github.com/yourusername/analog-pde-solver-sim/discussions)
- [Contributing Guide](../CONTRIBUTING.md)
- [Development Documentation](DEVELOPMENT.md)