# GitHub Actions Workflows Setup Guide

This repository contains comprehensive workflow templates in `docs/github-workflows/` that need to be manually copied to `.github/workflows/` to become active.

## 🚨 IMPORTANT: Manual Setup Required

Due to security restrictions, the GitHub Actions workflows cannot be automatically created. You must manually copy the workflow files from the templates directory.

## Quick Setup

```bash
# Create the workflows directory
mkdir -p .github/workflows

# Copy all workflow templates
cp docs/github-workflows/*.yml .github/workflows/

# Verify the copy
ls -la .github/workflows/
```

## Available Workflows

### 1. Continuous Integration (`ci.yml`)
**Location**: `docs/github-workflows/ci.yml` → `.github/workflows/ci.yml`

**Features**:
- Multi-Python version testing (3.9, 3.10, 3.11)
- System dependencies installation (ngspice, iverilog, verilator)
- Code quality checks (black, flake8, isort, mypy)
- Test execution with coverage reporting
- Security scanning (bandit, safety, pip-audit)
- Autonomous value discovery on main branch pushes

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main` branch

### 2. Security Scanning (`security.yml`)
**Location**: `docs/github-workflows/security.yml` → `.github/workflows/security.yml`

**Features**:
- Weekly dependency vulnerability scanning
- CodeQL static analysis for Python
- Secret detection with TruffleHog
- Automated security issue creation on vulnerabilities
- Security metrics tracking

**Triggers**:
- Push to `main` branch
- Weekly schedule (Monday 2 AM)
- Manual dispatch

### 3. Performance Monitoring (`performance.yml`)
**Location**: `docs/github-workflows/performance.yml` → `.github/workflows/performance.yml`

**Features**:
- Automated performance benchmarking
- Memory usage profiling
- Energy consumption estimation
- Performance regression detection
- Benchmark result artifacts

### 4. Dependency Review (`dependency-review.yml`)
**Location**: `docs/github-workflows/dependency-review.yml` → `.github/workflows/dependency-review.yml`

**Features**:
- Automated dependency review on PRs
- License compliance checking
- Vulnerability assessment for new dependencies

### 5. CodeQL Analysis (`codeql.yml`)
**Location**: `docs/github-workflows/codeql.yml` → `.github/workflows/codeql.yml`

**Features**:
- Advanced static analysis for security vulnerabilities
- Automated security advisory creation
- Integration with GitHub Security tab

### 6. Autonomous Value Discovery (`autonomous-value-discovery.yml`)
**Location**: `docs/github-workflows/autonomous-value-discovery.yml` → `.github/workflows/autonomous-value-discovery.yml`

**Features**:
- Continuous value discovery and backlog updates
- Technical debt identification
- Performance monitoring integration
- Automated PR creation for high-value improvements

## Required Secrets and Configuration

After copying the workflows, configure these repository secrets:

### GitHub Repository Secrets
```bash
# In GitHub repository settings → Secrets and variables → Actions

CODECOV_TOKEN          # For test coverage reporting
```

### Repository Variables
```bash
# In GitHub repository settings → Secrets and variables → Actions → Variables

PERFORMANCE_THRESHOLD  # Performance regression threshold (default: 5)
SECURITY_THRESHOLD     # Security scanning threshold (default: medium)
```

## Environment Setup

The workflows require these system dependencies to be available:

### Ubuntu/Debian Dependencies
```bash
sudo apt-get update
sudo apt-get install -y \
    ngspice \
    iverilog \
    verilator \
    build-essential \
    cmake
```

### Python Dependencies
All Python dependencies are automatically installed via:
```bash
pip install -e ".[dev,hardware]"
```

## Workflow Permissions

Ensure your repository has the following permissions enabled:

### Repository Settings → Actions → General
- **Actions permissions**: Allow all actions and reusable workflows
- **Workflow permissions**: Read and write permissions
- **Allow GitHub Actions to create and approve pull requests**: ✅ Enabled

### Security Settings
- **Dependabot alerts**: ✅ Enabled
- **Dependency graph**: ✅ Enabled
- **Dependabot security updates**: ✅ Enabled

## Validation Checklist

After setting up the workflows, verify:

- [ ] All workflow files copied to `.github/workflows/`
- [ ] Repository secrets configured
- [ ] Actions permissions enabled
- [ ] First workflow run successful
- [ ] Security scanning enabled
- [ ] Performance benchmarking working
- [ ] Autonomous value discovery active

## Advanced Configuration

### Custom Workflow Modifications

You can customize the workflows by editing:

```yaml
# Modify Python versions
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]

# Adjust performance thresholds
env:
  PERFORMANCE_REGRESSION_THRESHOLD: 10  # 10% regression allowed

# Configure security scanning
bandit:
  severity: high
  confidence: high
```

### Integration with External Services

The workflows support integration with:

- **Codecov**: Test coverage reporting
- **Dependabot**: Automated dependency updates
- **CodeQL**: Advanced security analysis
- **GitHub Security**: Vulnerability management

## Troubleshooting

### Common Issues

1. **Workflow not triggering**:
   - Check file location: `.github/workflows/` (not `docs/github-workflows/`)
   - Verify YAML syntax with `yamllint`
   - Check repository permissions

2. **Dependency installation failures**:
   - Verify system dependencies in workflow
   - Check Python version compatibility
   - Review requirements.txt for conflicts

3. **Security scanning failures**:
   - Update security tools to latest versions
   - Review and fix identified vulnerabilities
   - Adjust security thresholds if needed

4. **Performance benchmark failures**:
   - Increase timeout limits for large problems
   - Verify system resources availability
   - Check memory limits

### Getting Help

- Review workflow logs in Actions tab
- Check repository Issues for known problems
- Consult the troubleshooting guide: `docs/troubleshooting.md`

## Continuous Improvement

The autonomous value discovery system will:

1. **Monitor workflow performance** and suggest optimizations
2. **Identify bottlenecks** in CI/CD pipeline
3. **Recommend new tools** and integrations
4. **Track metrics** and provide improvement suggestions

This creates a self-improving SDLC that continuously enhances developer productivity and code quality.

---

## Next Steps

1. **Copy workflows** from templates to active directory
2. **Configure secrets** and permissions
3. **Test first workflow run** by creating a small commit
4. **Monitor dashboard** for autonomous value discovery results
5. **Review and approve** any automated improvement PRs

The autonomous SDLC system will take over from here, continuously discovering and implementing the highest-value improvements to your development workflow.