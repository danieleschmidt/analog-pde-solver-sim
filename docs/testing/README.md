# Testing Documentation

This directory contains comprehensive testing documentation for the Analog PDE Solver project.

## Test Structure

The test suite is organized into several categories:

### Test Categories

- **Unit Tests** (`tests/unit/`): Fast, isolated tests for individual components
- **Integration Tests** (`tests/integration/`): Tests for component interactions
- **End-to-End Tests** (`tests/e2e/`): Full pipeline tests from PDE spec to solution
- **Performance Tests** (`tests/performance/`): Benchmarking and performance regression tests
- **Hardware Tests** (`tests/hardware/`): Tests requiring SPICE/Verilog tools

### Test Markers

Use pytest markers to categorize and run specific test subsets:

```bash
# Run only fast unit tests
pytest -m "unit and not slow"

# Run integration tests
pytest -m integration

# Run all tests except hardware-dependent ones
pytest -m "not hardware"

# Run performance benchmarks
pytest -m performance --tb=line

# Run slow tests (includes performance and e2e)
pytest -m slow
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=analog_pde_solver --cov-report=html

# Run specific test file
pytest tests/unit/test_core.py

# Run tests matching pattern
pytest -k "poisson"

# Run tests in parallel (faster)
pytest -n auto
```

### Development Workflow

```bash
# Quick development cycle (fast tests only)
make test-fast

# Full test suite
make test-all

# With coverage
make test-cov

# Performance benchmarks
pytest -m performance --benchmark-only
```

### Continuous Integration

The CI pipeline runs different test configurations:

1. **Fast Tests**: Unit tests on multiple Python versions
2. **Integration Tests**: Integration and e2e tests on Linux
3. **Hardware Tests**: Tests requiring SPICE/Verilog tools
4. **Performance Tests**: Benchmarking and regression detection

## Test Data

Test data is organized in `tests/data/`:

- `reference_solutions/`: Known analytical solutions for validation
- `benchmark_problems/`: Standard PDE benchmark problems
- `spice_models/`: SPICE device models for circuit simulation
- `sample_grids/`: Pre-generated grid configurations

### Adding Test Data

1. Keep files small (< 10MB each)
2. Use standard formats (NumPy .npy, HDF5 .h5)
3. Document data source and format
4. Update relevant fixtures in `conftest.py`

## Test Fixtures

Common test fixtures are defined in:

- `tests/conftest.py`: Basic fixtures (temp dirs, mock objects)
- `tests/fixtures/pde_fixtures.py`: PDE-specific test cases and configurations

### Custom Fixtures

When creating new fixtures:

1. Use appropriate scope (`function`, `class`, `module`, `session`)
2. Clean up resources properly
3. Document fixture purpose and usage
4. Consider parametrization for multiple test cases

## Performance Testing

Performance tests verify:

- **Execution Time**: Solutions complete within expected timeframes
- **Memory Usage**: Memory consumption stays within limits
- **Convergence Rate**: Iterative solvers converge efficiently
- **Scaling**: Performance scales reasonably with problem size

### Performance Thresholds

Current performance targets:

| Problem Size | Max Time | Max Memory | Target Use Case |
|--------------|----------|------------|-----------------|
| 64×64        | 1.0s     | 100MB      | Development/testing |
| 256×256      | 10.0s    | 1GB        | Research problems |
| 1024×1024    | 60.0s    | 8GB        | Production workloads |

## Hardware Testing

Hardware tests require external tools:

### SPICE Testing

Requires NgSpice installation:

```bash
# Ubuntu/Debian
sudo apt-get install ngspice

# macOS
brew install ngspice
```

### Verilog Testing

Requires Verilog simulation tools:

```bash
# Ubuntu/Debian
sudo apt-get install iverilog verilator

# macOS
brew install icarus-verilog verilator
```

### Running Hardware Tests

```bash
# Check tool availability
make check-deps

# Run hardware tests
pytest -m hardware

# Run SPICE-specific tests
pytest -m spice

# Run Verilog tests
pytest -m verilog
```

## Coverage Requirements

Minimum coverage thresholds:

- **Overall**: 85%
- **Core modules**: 90%
- **Critical paths**: 95%

Coverage exclusions:

- Abstract methods and interfaces
- Platform-specific code
- Error handling for impossible conditions
- Debug and development utilities

## Troubleshooting

### Common Issues

**Tests fail with SPICE errors**:
- Verify NgSpice installation: `which ngspice`
- Check SPICE_LIB_DIR environment variable
- Run `make spice-test` to verify setup

**Performance tests timeout**:
- Tests may be resource-limited
- Consider running with `--tb=no` for cleaner output
- Check system resources with `htop` or Activity Monitor

**Import errors in tests**:
- Ensure package is installed in development mode: `pip install -e .`
- Check PYTHONPATH includes project root
- Verify virtual environment is activated

### Debug Mode

Enable debug mode for detailed test output:

```bash
# Enable debug logging
pytest --log-cli-level=DEBUG

# Capture stdout/stderr
pytest -s

# Drop into debugger on failure
pytest --pdb

# Profile test execution
pytest --profile
```

## Contributing Tests

When contributing new tests:

1. **Follow naming conventions**: `test_*.py`, `test_*()` functions
2. **Use appropriate markers**: Mark tests with relevant categories
3. **Write docstrings**: Document test purpose and expected behavior
4. **Keep tests focused**: One concept per test function
5. **Use fixtures**: Reuse common setup through fixtures
6. **Assert meaningfully**: Provide descriptive assertion messages
7. **Handle resources**: Clean up files, connections, etc.

### Test Review Checklist

- [ ] Tests follow project conventions
- [ ] Appropriate test markers applied
- [ ] Performance-sensitive tests marked as `slow`
- [ ] Hardware dependencies marked appropriately
- [ ] Test data files are reasonable size
- [ ] Assertions have descriptive messages
- [ ] Resource cleanup is handled
- [ ] Documentation is updated if needed