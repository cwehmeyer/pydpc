# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documentation site built with MkDocs and Material, published to GitHub
  Pages, with an API reference generated from docstrings and a tutorial page
  rendered from the example notebook
- Docstrings for the public API (`Cluster` and its building blocks)
- Quick-start example in the README
- Codecov coverage reporting in CI
- CodeQL code scanning (GitHub default setup)

### Changed

- Modernized the decision-graph and example-notebook plot styling
- Bumped copyright years and removed package metadata (`__author__`,
  `__email__`, `__license__`, `__copyright__`) that duplicated `pyproject.toml`

### Fixed

- `Cluster.assign` now raises a clear `ValueError` instead of passing empty
  clusters into the C extension, which could previously segfault (GH-8)

## [0.2.0] - 2026-07-01

First beta release.

### Added

- Support for supplying a precomputed distance matrix to `Cluster`
- GitHub Actions release workflow publishing to PyPI

### Changed

- Migrated packaging to `pyproject.toml` with `setuptools-scm` for versioning
- Switched development tooling to `uv`, `ruff`, and `pre-commit`
- Replaced CircleCI with GitHub Actions for CI
- Pinned `numpy>=2`

### Fixed

- Raise a clear error instead of a degenerate result when the estimated
  kernel size is zero (e.g. an all-zero distance matrix)

### Removed

- Dropped support for Python versions older than 3.10

## [0.1.3] - 2016-02-11

### Added

- Coveralls coverage reporting

### Changed

- Polished the README and example notebook

## [0.1.2] - 2016-01-19

### Added

- Tests for index handling

### Changed

- Moved `draw_decision_graph` onto the `Cluster` class

## [0.1.1] - 2016-01-17

### Added

- Continuous integration
- Example notebook
- C implementations of the halo and membership computations (performance)

## [0.1.0] - 2016-01-16

Initial release.
