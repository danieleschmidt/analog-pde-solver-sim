# Workflow Examples and Templates

This document provides additional workflow examples and templates for common development scenarios.

## Table of Contents

1. [Release Workflows](#release-workflows)
2. [Deployment Workflows](#deployment-workflows)
3. [Notification Workflows](#notification-workflows)
4. [Custom Action Examples](#custom-action-examples)
5. [Workflow Orchestration](#workflow-orchestration)

## Release Workflows

### Semantic Release Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write
  issues: write

jobs:
  release:
    runs-on: ubuntu-latest
    if: "!contains(github.event.head_commit.message, 'skip-ci')"
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install release tools
        run: |
          pip install build twine wheel
          pip install semantic-release

      - name: Generate changelog and bump version
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          semantic-release changelog
          semantic-release version

      - name: Build package
        run: |
          python -m build

      - name: Upload to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          python -m twine upload dist/*

      - name: Create GitHub Release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          semantic-release github-release

      - name: Update Docker images
        run: |
          echo "Triggering Docker image rebuild..."
          curl -X POST \
            -H "Accept: application/vnd.github.v3+json" \
            -H "Authorization: token ${{ secrets.GITHUB_TOKEN }}" \
            https://api.github.com/repos/${{ github.repository }}/dispatches \
            -d '{"event_type":"docker-rebuild"}'
```

### Pre-release Workflow

Create `.github/workflows/pre-release.yml`:

```yaml
name: Pre-release

on:
  push:
    branches: [develop]
  workflow_dispatch:

jobs:
  pre-release:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install build

      - name: Run tests
        run: pytest --cov=analog_pde_solver

      - name: Build pre-release
        run: python -m build

      - name: Upload to Test PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}
        run: |
          pip install twine
          python -m twine upload --repository testpypi dist/*

      - name: Create pre-release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v${{ github.run_number }}-pre
          release_name: Pre-release v${{ github.run_number }}
          prerelease: true
          draft: false
```

## Deployment Workflows

### Docker Image Build and Push

Create `.github/workflows/docker-build.yml`:

```yaml
name: Docker Build and Push

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]
  repository_dispatch:
    types: [docker-rebuild]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker images
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
          target: production

      - name: Build development image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:dev
          target: development
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Build hardware image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:hardware
          target: hardware
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Cloud Deployment

Create `.github/workflows/deploy-cloud.yml`:

```yaml
name: Cloud Deployment

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  deploy-staging:
    if: github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'staging'
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment..."
          # Add your staging deployment commands here

  deploy-production:
    if: github.event_name == 'release' || (github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'production')
    runs-on: ubuntu-latest
    environment: production
    needs: [deploy-staging]
    
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production environment..."
          # Add your production deployment commands here

      - name: Health check
        run: |
          sleep 30  # Wait for deployment
          curl -f https://your-app.com/health || exit 1

      - name: Notify deployment
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: "Production deployment ${{ job.status }}"
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## Notification Workflows

### Slack Integration

Create `.github/workflows/notifications.yml`:

```yaml
name: Notifications

on:
  workflow_run:
    workflows: ["CI", "Security", "Performance"]
    types: [completed]
  issues:
    types: [opened, closed]
  pull_request:
    types: [opened, closed, merged]

jobs:
  notify-slack:
    runs-on: ubuntu-latest
    if: always()
    
    steps:
      - name: Workflow notification
        if: github.event_name == 'workflow_run'
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ github.event.workflow_run.conclusion }}
          fields: repo,message,commit,author,action,eventName,ref,workflow
          text: |
            Workflow `${{ github.event.workflow_run.name }}` ${{ github.event.workflow_run.conclusion }}
            Repository: ${{ github.repository }}
            Branch: ${{ github.event.workflow_run.head_branch }}
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

      - name: Issue notification
        if: github.event_name == 'issues'
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          custom_payload: |
            {
              "text": "Issue ${{ github.event.action }}: ${{ github.event.issue.title }}",
              "attachments": [
                {
                  "color": "${{ github.event.action == 'opened' && 'warning' || 'good' }}",
                  "fields": [
                    {
                      "title": "Repository",
                      "value": "${{ github.repository }}",
                      "short": true
                    },
                    {
                      "title": "Issue",
                      "value": "<${{ github.event.issue.html_url }}|#${{ github.event.issue.number }}>",
                      "short": true
                    }
                  ]
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

      - name: PR notification
        if: github.event_name == 'pull_request'
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          custom_payload: |
            {
              "text": "Pull Request ${{ github.event.action }}: ${{ github.event.pull_request.title }}",
              "attachments": [
                {
                  "color": "${{ github.event.action == 'opened' && 'warning' || github.event.action == 'merged' && 'good' || 'danger' }}",
                  "fields": [
                    {
                      "title": "Repository",
                      "value": "${{ github.repository }}",
                      "short": true
                    },
                    {
                      "title": "Pull Request",
                      "value": "<${{ github.event.pull_request.html_url }}|#${{ github.event.pull_request.number }}>",
                      "short": true
                    },
                    {
                      "title": "Author",
                      "value": "${{ github.event.pull_request.user.login }}",
                      "short": true
                    }
                  ]
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## Custom Action Examples

### Composite Action for Setup

Create `.github/actions/setup-analog-pde/action.yml`:

```yaml
name: 'Setup Analog PDE Solver'
description: 'Setup Python environment and dependencies for Analog PDE Solver'

inputs:
  python-version:
    description: 'Python version to use'
    required: false
    default: '3.11'
  install-hardware-deps:
    description: 'Install hardware simulation dependencies'
    required: false
    default: 'false'

outputs:
  python-version:
    description: 'Python version that was installed'
    value: ${{ steps.setup-python.outputs.python-version }}

runs:
  using: 'composite'
  steps:
    - name: Setup Python
      id: setup-python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ inputs.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt', '**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install system dependencies
      shell: bash
      run: |
        sudo apt-get update
        sudo apt-get install -y build-essential cmake
        if [[ "${{ inputs.install-hardware-deps }}" == "true" ]]; then
          sudo apt-get install -y ngspice iverilog verilator
        fi

    - name: Install Python dependencies
      shell: bash
      run: |
        python -m pip install --upgrade pip wheel setuptools
        if [[ "${{ inputs.install-hardware-deps }}" == "true" ]]; then
          pip install -e ".[dev,hardware]"
        else
          pip install -e ".[dev]"
        fi

    - name: Install pre-commit hooks
      shell: bash
      run: |
        pre-commit install --install-hooks
```

### Custom Python Action

Create `.github/actions/performance-check/action.yml`:

```yaml
name: 'Performance Check'
description: 'Run performance benchmarks and check for regressions'

inputs:
  baseline-file:
    description: 'Path to baseline performance file'
    required: false
    default: 'benchmark_results/baseline.json'
  threshold:
    description: 'Performance regression threshold (percentage)'
    required: false
    default: '5'
  
outputs:
  performance-score:
    description: 'Overall performance score'
    value: ${{ steps.benchmark.outputs.score }}
  regression-detected:
    description: 'Whether performance regression was detected'
    value: ${{ steps.check.outputs.regression }}

runs:
  using: 'composite'
  steps:
    - name: Run benchmarks
      id: benchmark
      shell: bash
      run: |
        python scripts/run-benchmarks.py --output benchmark_results/current.json
        echo "score=$(python scripts/calculate-score.py benchmark_results/current.json)" >> $GITHUB_OUTPUT

    - name: Check for regression
      id: check
      shell: bash
      run: |
        if [[ -f "${{ inputs.baseline-file }}" ]]; then
          regression=$(python scripts/check-regression.py \
            --baseline "${{ inputs.baseline-file }}" \
            --current benchmark_results/current.json \
            --threshold "${{ inputs.threshold }}")
          echo "regression=$regression" >> $GITHUB_OUTPUT
        else
          echo "regression=false" >> $GITHUB_OUTPUT
          echo "No baseline file found, skipping regression check"
        fi

    - name: Update baseline
      if: steps.check.outputs.regression == 'false'
      shell: bash
      run: |
        cp benchmark_results/current.json "${{ inputs.baseline-file }}"
        echo "Baseline updated with current performance metrics"

    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark_results/
        retention-days: 30
```

## Workflow Orchestration

### Matrix Strategy Example

```yaml
name: Advanced Testing Matrix

on: [push, pull_request]

jobs:
  test-matrix:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11']
        test-type: [unit, integration, hardware]
        exclude:
          # Hardware tests only on Linux
          - os: windows-latest
            test-type: hardware
          - os: macos-latest
            test-type: hardware
          # Skip Python 3.9 on Windows for integration tests
          - os: windows-latest
            python-version: '3.9'
            test-type: integration
        include:
          # Add special GPU testing configuration
          - os: ubuntu-latest
            python-version: '3.11'
            test-type: gpu
            gpu: true
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup environment
        uses: ./.github/actions/setup-analog-pde
        with:
          python-version: ${{ matrix.python-version }}
          install-hardware-deps: ${{ matrix.test-type == 'hardware' }}

      - name: Run tests
        run: |
          case "${{ matrix.test-type }}" in
            unit)
              pytest tests/unit/ -v
              ;;
            integration)
              pytest tests/integration/ -v --timeout=300
              ;;
            hardware)
              pytest tests/hardware/ -v --timeout=600 -m hardware
              ;;
            gpu)
              pytest tests/gpu/ -v -m gpu
              ;;
          esac

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results-${{ matrix.os }}-${{ matrix.python-version }}-${{ matrix.test-type }}
          path: test-results/
```

### Conditional Workflow Execution

```yaml
name: Conditional Workflows

on:
  push:
    paths:
      - 'analog_pde_solver/**'
      - 'tests/**'
      - 'docs/**'
      - '.github/workflows/**'

jobs:
  changes:
    runs-on: ubuntu-latest
    outputs:
      code: ${{ steps.changes.outputs.code }}
      docs: ${{ steps.changes.outputs.docs }}
      tests: ${{ steps.changes.outputs.tests }}
      workflows: ${{ steps.changes.outputs.workflows }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v2
        id: changes
        with:
          filters: |
            code:
              - 'analog_pde_solver/**'
            docs:
              - 'docs/**'
            tests:  
              - 'tests/**'
            workflows:
              - '.github/workflows/**'

  test-code:
    needs: changes
    if: needs.changes.outputs.code == 'true' || needs.changes.outputs.tests == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run code tests
        run: pytest tests/ -v

  build-docs:
    needs: changes
    if: needs.changes.outputs.docs == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build documentation
        run: |
          cd docs
          make html

  validate-workflows:
    needs: changes
    if: needs.changes.outputs.workflows == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate workflow syntax
        run: |
          for workflow in .github/workflows/*.yml; do
            echo "Validating $workflow"
            yamllint "$workflow"
          done
```

## Best Practices

### Security Best Practices

1. **Use specific action versions**: `uses: actions/checkout@v4` not `@main`
2. **Limit permissions**: Specify minimum required permissions
3. **Validate inputs**: Always validate workflow inputs and environment variables
4. **Use secrets properly**: Never log secrets, use masked variables
5. **Pin dependencies**: Use specific versions for reproducibility

### Performance Optimization

1. **Cache dependencies**: Use `actions/cache` for pip, npm, etc.
2. **Parallel execution**: Use matrix strategies and parallel jobs
3. **Fail fast**: Set `fail-fast: false` only when needed
4. **Optimize Docker builds**: Use multi-stage builds and BuildKit
5. **Conditional execution**: Use path filters to skip unnecessary jobs

### Maintenance Guidelines

1. **Regular updates**: Keep actions and dependencies updated
2. **Monitor usage**: Track Action minutes and costs
3. **Document workflows**: Maintain clear documentation
4. **Test changes**: Validate workflow changes in feature branches
5. **Monitor performance**: Track workflow execution times and success rates

These examples provide a comprehensive foundation for building robust, efficient, and maintainable GitHub Actions workflows for the Analog PDE Solver project.