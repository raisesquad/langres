# langres Examples

This directory contains example scripts demonstrating how to use the langres library.

## Planned Examples

### High-Level API (langres.tasks)

- **basic_deduplication.py** - Simple deduplication of a single dataset
- **entity_linking.py** - Linking source data to a target dataset
- **with_human_review.py** - Using the ReviewQueue for human-in-the-loop
- **synthetic_data_generation.py** - Generating training data with SyntheticGenerator

### Low-Level API (langres.core)

- **custom_blocker.py** - Building a custom Blocker from scratch
- **custom_flow.py** - Implementing a custom Module/Flow
- **optimization_workflow.py** - Using the Optimizer for HPO and prompt tuning
- **advanced_clustering.py** - Using constraints and different clustering methods

## Running Examples

```bash
# Run any example with:
uv run python examples/basic_deduplication.py

# Or activate the virtual environment first:
source .venv/bin/activate
python examples/basic_deduplication.py
```

## Requirements

Examples may require additional dependencies. Check each script for specific requirements.
