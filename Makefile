.PHONY: install test lint format clean docs build

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest

# Default target
help:
	@echo "Available commands:"
	@echo "  install     Install package and dependencies"
	@echo "  install-dev Install with development dependencies"
	@echo "  test        Run test suite"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with black and isort"
	@echo "  typecheck   Run mypy type checking"
	@echo "  clean       Clean build artifacts"
	@echo "  docs        Build documentation"
	@echo "  build       Build distribution packages"

# Installation
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev,docs,viz]
	pre-commit install

# Testing
test:
	$(PYTEST)

test-cov:
	$(PYTEST) --cov=analog_pde_solver --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 analog_pde_solver tests
	black --check analog_pde_solver tests
	isort --check-only analog_pde_solver tests

format:
	black analog_pde_solver tests
	isort analog_pde_solver tests

typecheck:
	mypy analog_pde_solver

# Documentation
docs:
	cd docs && $(MAKE) html

docs-live:
	cd docs && sphinx-autobuild . _build/html

# Build
build:
	$(PYTHON) -m build

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development workflow
dev-setup: install-dev
	@echo "Development environment ready!"

check: lint typecheck test
	@echo "All checks passed!"