# Contributing to Gridworld Q-Learning TD Simulation

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/gridworld-qlearning-td.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Set up development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use type hints for function signatures
- Write docstrings for all public functions/classes
- Keep functions focused and modular

### Testing

- Add tests for new features in `tests/`
- Ensure all tests pass before submitting PR:
  ```bash
  PYTHONPATH=$PWD pytest -v
  ```
- Aim for high test coverage

### Documentation

- Update README.md if adding new features
- Add docstrings following Google/NumPy style
- Update relevant docs in `docs/` folder

## Submitting Changes

1. Commit your changes with clear messages:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Open a Pull Request with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable

## Code Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged

## Areas for Contribution

- New maze generation algorithms
- Additional RL algorithms (SARSA, DQN, etc.)
- Performance optimizations
- UI improvements
- Documentation enhancements
- Bug fixes

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about usage or development

Thank you for contributing! 🎉
