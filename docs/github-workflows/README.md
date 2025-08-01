# GitHub Actions Workflow Templates

⚠️ **Manual Setup Required**: Due to GitHub security restrictions, workflow files cannot be automatically created. Please manually copy these templates to `.github/workflows/` in your repository.

## Quick Setup

1. Create `.github/workflows/` directory in your repository
2. Copy the workflow files from this directory to `.github/workflows/`
3. Configure the required repository secrets (see below)
4. Enable GitHub Actions in your repository settings

## Available Workflows

### Core CI/CD Workflows
- **`ci.yml`** - Continuous Integration with testing, linting, and coverage
- **`security.yml`** - Comprehensive security scanning (weekly + on-demand)
- **`performance.yml`** - Performance benchmarking with regression detection
- **`codeql.yml`** - Advanced static analysis with CodeQL
- **`dependency-review.yml`** - Dependency vulnerability scanning

### Autonomous Operations
- **`autonomous-value-discovery.yml`** - Hourly value discovery and execution

## Required Repository Secrets

Add these in your repository Settings → Secrets and variables → Actions:

### Essential Secrets
- `CODECOV_TOKEN` - From [Codecov.io](https://codecov.io) for coverage reporting
- `ANTHROPIC_API_KEY` - For autonomous Claude-powered execution (optional)

### Optional Secrets
- `PYPI_API_TOKEN` - For automated package publishing
- `TEST_PYPI_API_TOKEN` - For testing releases

## Repository Settings

### Branch Protection Rules (Settings → Branches)
1. Protect `main` branch
2. Require pull request reviews (minimum 1)
3. Require status checks:
   - `test (3.9)`
   - `test (3.10)`
   - `test (3.11)`
   - `security / dependency-scan`
   - `security / sast-scan`

### Security Settings (Settings → Security)
1. Enable **Dependency graph**
2. Enable **Dependabot alerts**
3. Configure **Dependabot security updates**
4. Enable **Secret scanning**
5. Enable **Code scanning** (CodeQL)

## Autonomous Execution

Once workflows are set up, the repository will automatically:

- **Discover value items** hourly using pattern matching and static analysis
- **Execute high-priority work** when triggered manually or on schedule
- **Update the backlog** with newly discovered items and priorities
- **Learn and adapt** scoring models based on execution outcomes

### Manual Trigger
```bash
# Trigger autonomous value discovery and execution
gh workflow run autonomous-value-discovery.yml -f force_execution=true
```

## Workflow Features

### Advanced Security
- CodeQL semantic analysis
- Dependency vulnerability scanning
- Secret detection with TruffleHog
- SARIF upload for GitHub Security tab
- Automated security issue creation

### Performance Monitoring
- Comprehensive benchmark suite
- Multi-profile testing (fast/default/accurate)
- Performance regression detection
- Baseline comparison and updates
- Memory and CPU profiling

### Quality Gates
- Python 3.9, 3.10, 3.11 matrix testing
- Code formatting (Black), linting (Flake8), type checking (MyPy)
- Test coverage reporting with thresholds
- Documentation build validation

### Autonomous Operations
- Pattern-based work item discovery
- WSJF + ICE + Technical Debt scoring
- Automated backlog management
- Continuous learning and adaptation
- Value metrics tracking

## Troubleshooting

### Common Issues

1. **Workflow not triggering**
   - Check branch protection rules
   - Verify required secrets are set
   - Check workflow syntax with GitHub Actions validator

2. **Permission errors**
   - Ensure GITHUB_TOKEN has sufficient permissions
   - Check repository settings for Actions permissions

3. **Test failures**
   - Review failed test logs in Actions tab
   - Check dependency compatibility
   - Verify system requirements are met

### Support Resources
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Repository Security Best Practices](../security/security-guidelines.md)
- [Performance Benchmarking Guide](../benchmarking/README.md)

---

**🚀 Once configured, your repository will operate as a fully autonomous, self-improving development ecosystem with continuous value discovery and execution capabilities.**