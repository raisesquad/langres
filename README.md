# langres

A two-layer entity resolution framework with optimization, blocking, and human-in-the-loop capabilities.

## Overview

**langres** is a modern Python library for entity resolution (ER) tasks, designed to be both easy to use and fully extensible. It provides a two-layer API:

- **langres.tasks** (High-Level): Pre-built components for common ER workflows (deduplication, entity linking)
- **langres.core** (Low-Level): Extensible base classes for building custom ER pipelines

## Key Features

- **Pre-built Flows**: Out-of-the-box matching logic for companies, products, and more
- **Intelligent Blocking**: High-recall candidate generation using ANN search
- **Auto-Optimization**: Hyperparameter tuning with Optuna and prompt optimization with DSPy
- **Human-in-the-Loop**: Built-in review queue for uncertain matches
- **Synthetic Data Generation**: LLM-powered training data creation
- **Master Data Creation**: Configurable canonicalization with survivorship rules

## Installation

```bash
pip install langres
```

## Quick Start

```python
from langres.tasks import DeduplicationTask
from langres.flows import CompanyFlow
from langres.blockers import DedupeBlocker

# Set up the task
flow = CompanyFlow()
blocker = DedupeBlocker()
task = DeduplicationTask(flow=flow, blocker=blocker)

# Optimize and run
task.compile(training_data, metric="bcubed_f1")
clusters = task.run(your_data)
```

## Documentation

See the [full documentation](docs/index.md) for detailed guides and API references.

## Examples

Check out the [examples/](examples/) directory for sample scripts and usage patterns.

## Requirements

- Python e3.12
- See `pyproject.toml` for dependencies

## License

TBD

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
