# Test Data

This directory contains test data files for the analog PDE solver test suite.

## Contents

- `reference_solutions/`: Known good solutions for validation
- `benchmark_problems/`: Standard PDE benchmark problems
- `spice_models/`: SPICE device models for testing
- `sample_grids/`: Pre-generated grid configurations

## Usage

Test data files are loaded automatically by the test fixtures in `conftest.py`.
All data should be in standard formats (NumPy `.npy`, HDF5 `.h5`, etc.).

## Adding New Test Data

1. Place files in appropriate subdirectory
2. Update relevant test fixtures if needed
3. Keep file sizes reasonable (< 10MB per file)
4. Document data format and source in this README