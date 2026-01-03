# Contributing to lox

Thank you for your interest in contributing to `lox`! We welcome contributions from everyone, whether you're fixing a bug, improving documentation, or suggesting new features.

## Ways to Contribute

- **Report Bugs:** If you find a bug, please open an issue with a clear description and a minimal reproducible example.
- **Suggest Features:** Have an idea for a new feature? Open an issue to discuss it.
- **Improve Documentation:** Help us make our documentation clearer and more comprehensive.
- **Contribute Code:** Submit pull requests for bug fixes or new features.

## Development Setup

We aim to keep the setup process as simple as possible.

### 1. Environment Setup
We recommend using a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,tests,docs,wandb,neptune]"
```

### 2. Running Tests
Ensure all tests pass before submitting a pull request.

```bash
pytest tests
```

### 3. Code Quality
We use `black` and `isort` for formatting. Please run them before committing.

```bash
black .
isort .
```

## Pull Request Process

1.  **Open an Issue:** For significant changes, please open an issue first to discuss your proposal.
2.  **Create a Branch:** Work on a feature branch (e.g., `feat/awesome-new-feature`).
3.  **Submit PR:** Open a PR against the `main` branch.
4.  **Review:** Participate in the review process and address any feedback.

## Roadmap

Here are some areas where we're looking for help:

- **Neptune Support:** Implement `NeptuneLogger` (currently placeholder in dependencies).
- **Additional Loggers:** Add support for TensorBoard, MLflow, or other common backends.
- **Documentation:** 
    - Add more real-world examples (e.g., training a neural network with `lox`).
    - Expand "The Sharp Bits" section with common pitfalls.
- **Testing:** Improve test coverage, especially for different JAX transformations.
- **Performance:** Benchmarking `lox` overhead and optimizing where possible.

Looking forward to your contributions!
