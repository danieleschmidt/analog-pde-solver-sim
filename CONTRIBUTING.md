# Contributing to Analog PDE Solver Sim

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`

## Development Process

1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Run tests: `pytest`
5. Run linting: `black . && flake8`
6. Commit your changes with descriptive messages
7. Push to your fork and submit a pull request

## Priority Areas for Contributions

- Additional PDE types (Maxwell, Schrödinger, etc.)
- Improved noise models for analog circuits
- Hardware validation with real devices
- Optimization algorithms for crossbar mapping
- Documentation and tutorials

## Code Style

- Follow PEP 8 Python style guidelines
- Use Black for code formatting
- Add type hints where appropriate
- Write docstrings for public functions
- Keep functions focused and testable

## Pull Request Guidelines

- Provide a clear description of changes
- Reference any related issues
- Include tests for new features
- Update documentation as needed
- Ensure CI checks pass

## Questions?

Feel free to open an issue for questions or discussion about potential contributions.