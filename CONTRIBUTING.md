# Contributing to reptimeline

Thanks for your interest in contributing! This guide covers the basics.

## Development setup

```bash
git clone https://github.com/arturoornelasb/reptimeline.git
cd reptimeline
pip install -e ".[dev]"
pre-commit install
```

## Running checks

```bash
# Tests
pytest tests/ -v --cov=reptimeline

# Lint
ruff check reptimeline/ tests/

# Type check
mypy reptimeline/

# Build docs
pdoc3 --html --output-dir docs/ reptimeline
```

## Code style

- **Ruff** enforces style. Run `ruff check --fix` to auto-fix.
- Line length: 100 characters.
- Import sorting is enforced (isort-compatible via ruff).
- Type hints are expected on public API functions.

## Making changes

1. Fork the repo and create a branch from `master`.
2. Make your changes. Add tests for new functionality.
3. Run `pytest` and `ruff check` to ensure everything passes.
4. Commit with a clear message (e.g., `feat: add X`, `fix: resolve Y`).
5. Open a pull request against `master`.

## What to contribute

- Bug fixes and test improvements are always welcome.
- New extractors (implement `RepresentationExtractor`).
- New visualization modules.
- Documentation improvements.
- Performance improvements (especially for `BitDiscovery` triadic search).

## What requires discussion first

Open an issue before working on:

- Changes to the public API (`ConceptSnapshot`, `Timeline`, `TimelineTracker`).
- New dependencies added to `[project.dependencies]`.
- Architectural changes to the extractor abstraction.

## License

By contributing, you agree that your contributions will be licensed under the project's
[BUSL-1.1](LICENSE) license. The license converts to AGPL-3.0 on 2030-03-21.
