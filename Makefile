.PHONY: help install test lint format clean docs dev-setup

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install package and dependencies"
	@echo "  dev-setup   - Set up development environment"
	@echo "  test        - Run test suite"
	@echo "  test-fast   - Run fast tests only"
	@echo "  test-all    - Run all tests including slow/hardware"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black and isort"
	@echo "  typecheck   - Run type checking with mypy"
	@echo "  docs        - Build documentation"
	@echo "  clean       - Clean build artifacts"
	@echo "  security    - Run security checks"

install:
	pip install -e .

dev-setup:
	pip install -e ".[dev,docs,hardware]"
	pre-commit install

test:
	pytest -v

test-fast:
	pytest -v -m "not slow and not hardware"

test-all:
	pytest -v --runslow --runhardware

test-cov:
	pytest --cov=analog_pde_solver --cov-report=html --cov-report=term

lint:
	flake8 analog_pde_solver/ tests/
	black --check analog_pde_solver/ tests/
	isort --check-only analog_pde_solver/ tests/

format:
	black analog_pde_solver/ tests/
	isort analog_pde_solver/ tests/

typecheck:
	mypy analog_pde_solver/

docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

security:
	bandit -r analog_pde_solver/
	safety check

spice-test:
	@echo "Checking SPICE installation..."
	@which ngspice || echo "NgSpice not found - install with package manager"

verilog-test:
	@echo "Checking Verilog tools..."
	@which iverilog || echo "Icarus Verilog not found"
	@which verilator || echo "Verilator not found"

check-deps: spice-test verilog-test
	@echo "Dependency check complete"

# Docker targets
.PHONY: docker-build docker-dev docker-test docker-clean

docker-build:
	docker build -t analog-pde-solver:latest .
	docker build -t analog-pde-solver:dev --target development .
	docker build -t analog-pde-solver:hardware --target hardware .

docker-dev:
	docker-compose up -d dev

docker-test:
	docker-compose run --rm test

docker-test-fast:
	docker-compose run --rm test-fast

docker-perf:
	docker-compose run --rm test-perf

docker-docs:
	docker-compose up -d docs
	@echo "Documentation server running at http://localhost:8000"

docker-jupyter:
	docker-compose up -d jupyter
	@echo "Jupyter server running at http://localhost:8888"

docker-spice:
	docker-compose run --rm spice-sim

docker-hardware:
	docker-compose run --rm hardware

docker-lint:
	docker-compose run --rm lint

docker-clean:
	docker-compose down -v
	docker system prune -f
	docker volume prune -f

# Build automation
.PHONY: build-all build-wheel build-sdist

build-all: build-wheel build-sdist

build-wheel:
	python -m build --wheel

build-sdist:
	python -m build --sdist

build-check:
	python -m twine check dist/*