# langres Documentation

A two-layer entity resolution framework with optimization, blocking, and human-in-the-loop capabilities.

## Overview

langres is designed with a two-layer API:

- **langres.tasks** (High-Level): Pre-built components for common entity resolution tasks
- **langres.core** (Low-Level): Extensible base classes for custom implementations

## Table of Contents

- [Getting Started](getting-started.md)
- [API Reference](api/)
- [Tutorials](tutorials/)
- [Examples](../examples/)

## Quick Start

```python
from langres.tasks import DeduplicationTask
from langres.flows import CompanyFlow
from langres.blockers import DedupeBlocker

# Coming soon...
```
