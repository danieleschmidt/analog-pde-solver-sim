# GitHub Actions Workflow Templates

## Required Manual Setup

**⚠️ Important**: GitHub Actions workflows require repository-specific permissions and cannot be automatically created. Follow this guide to manually create the required workflow files in `.github/workflows/`.

## 1. Continuous Integration - `ci.yml`

Create `.github/workflows/ci.yml`:

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        
    steps:
    - uses: actions/checkout@v4
      
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y ngspice iverilog verilator
        
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,hardware]"
        
    - name: Run linting
      run: |
        black --check analog_pde_solver/ tests/
        flake8 analog_pde_solver/ tests/
        isort --check-only analog_pde_solver/ tests/
        
    - name: Run type checking
      run: mypy analog_pde_solver/
      
    - name: Run tests with coverage
      run: |
        pytest --cov=analog_pde_solver --cov-report=xml --cov-report=term
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml
        
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety pip-audit
        
    - name: Run security analysis
      run: |
        bandit -r analog_pde_solver/
        safety check
        pip-audit
```

## 2. Security Scanning - `security.yml`

Create `.github/workflows/security.yml`:

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  workflow_dispatch:

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pip-audit safety
        
    - name: Audit dependencies
      run: |
        pip-audit --desc --format=json --output=audit-results.json
        safety check --json --output=safety-results.json
        
    - name: Upload security results
      uses: actions/upload-artifact@v3
      with:
        name: security-scan-results
        path: |
          audit-results.json
          safety-results.json
          
  sast-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python
        
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      
  secret-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: TruffleHog OSS
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD
```

## 3. Documentation Build - `docs.yml`

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
    paths: [ 'docs/**', 'analog_pde_solver/**' ]
  pull_request:
    paths: [ 'docs/**', 'analog_pde_solver/**' ]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[docs]"
        
    - name: Build documentation
      run: |
        cd docs
        make html
        
    - name: Check documentation links
      run: |
        pip install linkchecker
        linkchecker docs/_build/html/
        
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

## 4. Hardware Testing - `hardware-test.yml`

Create `.github/workflows/hardware-test.yml`:

```yaml
name: Hardware Validation

on:
  workflow_dispatch:
  push:
    tags: [ 'v*' ]
  schedule:
    - cron: '0 6 * * 0'  # Weekly on Sunday at 6 AM

jobs:
  spice-simulation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y ngspice
        
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[hardware]"
        
    - name: Run SPICE tests
      run: pytest -v tests/hardware/ -m spice
      
    - name: Generate performance report
      run: |
        python scripts/benchmark.py --output benchmark-results.json
        
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: hardware-test-results
        path: |
          benchmark-results.json
          
  verilog-synthesis:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      
    - name: Install Verilog tools
      run: |
        sudo apt-get update
        sudo apt-get install -y iverilog verilator
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[hardware]"
        
    - name: Run Verilog tests
      run: pytest -v tests/hardware/ -m verilog
      
    - name: Synthesis validation
      run: |
        python scripts/rtl-validation.py
```

## 5. Release Automation - `release.yml`

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags: [ 'v*' ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Run tests
      run: pytest
      
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Build package
      run: python -m build
      
    - name: Check package
      run: twine check dist/*
      
    - name: Upload to Test PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}
      run: twine upload --repository testpypi dist/*
      
    - name: Upload to PyPI
      if: startsWith(github.ref, 'refs/tags/v')
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
      
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Required Repository Secrets

Add these secrets in your repository settings:

### Required Secrets
- `CODECOV_TOKEN`: From Codecov.io for coverage reporting
- `PYPI_API_TOKEN`: From PyPI for package publishing
- `TEST_PYPI_API_TOKEN`: From Test PyPI for testing releases

### Optional Secrets
- `GH_PAGES_TOKEN`: Personal access token for GitHub Pages (if using custom domain)

## Branch Protection Rules

Configure these rules in your repository settings:

1. **Protect main branch**:
   - Require pull request reviews (1 reviewer minimum)
   - Require status checks to pass
   - Require branches to be up to date before merging
   - Include administrators

2. **Required status checks**:
   - `test (3.9)`
   - `test (3.10)` 
   - `test (3.11)`
   - `security`
   - `build-docs`

## Next Steps

1. Create `.github/workflows/` directory in your repository
2. Add the workflow files above with appropriate content
3. Configure repository secrets
4. Set up branch protection rules
5. Enable GitHub Pages for documentation (if desired)
6. Configure Codecov integration

The workflows will automatically trigger based on their defined conditions and provide comprehensive CI/CD coverage for your Python project.