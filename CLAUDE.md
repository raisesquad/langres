# Claude Code Guidelines for langres

## Project Overview

**langres** is a Python entity resolution framework in early development. It aims to provide a composable, optimizable approach to entity resolution with a two-layer API (high-level tasks and low-level core components).

**Current Stage**: We are at the **initial POC (Proof of Concept) stage**. Before building the full framework, we are validating the core architecture through three progressively sophisticated approaches:
1. Classical string matching (rapidfuzz baseline)
2. Semantic vector search (embedding-based)
3. Hybrid blocking + LLM judge (target architecture)

**📋 See `docs/POC.md` for the complete POC plan**, including:
- The hypothesis we're validating
- Success criteria (BCubed F1 ≥ 0.85 for Approach 3)
- Core components to build (`Module`, `Blocker`, `Clusterer`)
- TDD development approach
- Go/No-Go decision criteria

**Current focus**: Building production-quality `langres.core` primitives with 100% test coverage. This is NOT throwaway prototype code—these components will become the foundation of the full library.

## Code Style & Standards

### Python Guidelines

- **Python Version**: Requires Python >=3.12
- **Code Formatting**: Use `ruff` for code formatting and linting
- **Type Hints**: Use comprehensive type hints throughout. Use built-in types (`list`, `dict`, `str`, etc.) instead of `typing.List`, `typing.Dict`, etc. (Python 3.12+ feature)
- **Validation**: Pydantic-first approach - all data models should use Pydantic
- **Package Manager**: This project uses `uv` for dependency management

### Naming Conventions

- **Classes**: PascalCase (e.g., `DeduplicationTask`, `CompanyFlow`)
- **Functions/Methods**: snake_case (e.g., `generate_candidates`, `compile`)
- **Private Methods**: Prefix with underscore (e.g., `_internal_method`)
- **Constants**: UPPER_SNAKE_CASE

### Project Structure

```
langres/
├── src/langres/
│   ├── core/           # Low-level API (Module, Blocker, Optimizer, Clusterer, Canonicalizer)
│   ├── tasks/          # High-level API (DeduplicationTask, EntityLinkingTask)
│   ├── flows/          # Pre-built matching logic (CompanyFlow, ProductFlow)
│   ├── blockers/       # Candidate generation (DedupeBlocker, LinkingBlocker)
│   └── data/           # Synthetic data generation
├── tests/              # Test suite
├── examples/           # Usage examples
└── docs/               # Documentation
```

## Architecture Principles

### The Two-Layer API

1. **High-Level (`langres.tasks`)**: Pre-built task runners for common use cases
   - Target: 80% of users
   - Examples: `DeduplicationTask`, `EntityLinkingTask`
   - Philosophy: Like scikit-learn's Pipeline

2. **Low-Level (`langres.core`)**: Composable primitives for custom pipelines
   - Target: 20% of users (advanced use cases)
   - Components: `Module`, `Blocker`, `Optimizer`, `Clusterer`, `Canonicalizer`
   - Philosophy: Like PyTorch's primitives

### Key Design Principles

- **Pydantic-First**: All data models use Pydantic for validation
- **Full Observability**: Every `PairwiseJudgement` carries provenance and reasoning
- **Composable**: Components should be reusable across different tasks
- **Optimizable**: Support both hyperparameter tuning (Optuna) and prompt optimization (DSPy)
- **Cost-Aware**: Consider API costs, computation costs, and optimization budgets

## Dependencies

### Core Stack

- **Pydantic**: Data validation and schema management
- **Optuna**: Hyperparameter optimization
- **DSPy**: Prompt optimization for LLM-based matchers
- **sentence-transformers**: Semantic embeddings
- **rapidfuzz**: String similarity metrics
- **networkx**: Graph clustering algorithms
- **PyTorch**: Deep learning and learnable components

### Development Tools

- **ruff**: Code formatting and linting
- **pytest**: Testing framework

## Implementation Guidelines

### When Adding New Components

1. **Blockers**: Must implement candidate generation and schema normalization
2. **Flows (Modules)**: Must yield `PairwiseJudgement` objects
3. **Tasks**: Should compose Blocker + Flow + optional Optimizer
4. **All Components**: Should support both `.run()` and `.compile()` methods where appropriate

### Testing

- Write tests for all new components in `tests/`
- Use descriptive test names: `test_deduplication_task_with_company_flow`
- Include both unit tests and integration tests

### Documentation

- Update relevant docs in `docs/` when changing architecture
- Add docstrings to all public methods
- Include usage examples for new components in `examples/`

## Common Patterns

### Task Implementation

```python
class SomeTask:
    def __init__(self, flow: Module, blocker: Blocker):
        self.flow = flow
        self.blocker = blocker

    def compile(self, gold_data, metric: str):
        """Optimize hyperparameters on gold data"""
        pass

    def run(self, data):
        """Execute the task on input data"""
        pass
```

### Flow (Module) Implementation

```python
class SomeFlow(Module):
    def forward(self, candidates):
        """Yield PairwiseJudgement for each candidate pair"""
        for pair in candidates:
            score = self._compute_similarity(pair)
            yield PairwiseJudgement(
                left_id=pair.left.id,
                right_id=pair.right.id,
                score=score,
                score_type="calibrated_prob"
            )
```

## Important Notes

- This is an **early-stage project** - expect significant changes
- Prioritize clean, testable code over premature optimization
- Document design decisions in code comments
- Focus on the core use cases: Deduplication and Entity Linking (V1 scope)

## Reference Documentation

When working on langres, consult these comprehensive docs for detailed context:

### `docs/POC.md` - Proof of Concept Plan ⭐ **START HERE**
**When to use**: Understanding the current development stage and priorities
- **Before starting any implementation work** - this defines our current focus
- When deciding what to build next (we're doing core primitives, not tasks layer)
- To understand the three approaches we're validating (classical, semantic, LLM hybrid)
- When clarifying success criteria (BCubed F1 targets, TDD requirements)
- To see what's in scope NOW vs. what's planned for later (no Optimizer, no tasks yet)
- When evaluating architectural decisions (does this fit the POC validation goals?)

### `docs/PROJECT_OVERVIEW.md` - Architecture and Philosophy
**When to use**: Understanding the "why" behind design decisions
- Before proposing major architectural changes
- When clarifying the relationship between components (Blocker, Flow, Optimizer, etc.)
- To understand the philosophy behind the two-layer API
- When explaining langres to users or in documentation

### `docs/TECHNICAL_OVERVIEW.md` - API Reference and Data Contracts
**When to use**: Implementation details and type signatures
- When implementing new components (Blocker, Flow, Task, etc.)
- To understand data contracts (what `PairwiseJudgement`, `Candidate`, etc. should look like)
- When defining method signatures for consistency
- To see complete API examples and expected inputs/outputs

### `docs/USE_CASES.md` - Use Case Taxonomy and Roadmap
**When to use**: Understanding scope and future direction
- When evaluating whether a feature request is in scope
- To understand which use cases are V1 vs. V1.1 vs. out-of-scope
- When someone asks "can langres do X?" (streaming, temporal, collective resolution, etc.)
- To see the formal taxonomy of supported entity resolution patterns
