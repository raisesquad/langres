# langres: A Composable & Optimizable Entity Resolution Framework

langres is a Python-native, "batteries-included" library for building, optimizing, and deploying entity resolution (ER) pipelines. It is designed to replace rigid, configuration-driven systems and black-box tools with a flexible, testable, and transparent framework that leverages modern data science tools.

It is a "glue" library that provides a clear, type-safe API (built on Pydantic) and a powerful optimization engine (built on Optuna and DSPy) to connect your data to your final, deduplicated "golden" records.

The langres philosophy is inspired by a two-layer API model:

- **langres.tasks** (High-Level / "Task-Based"): For 80% of use cases. This is the main entry point for users. It provides out-of-the-box runners for specific operations (e.g., DeduplicationTask, EntityLinkingTask) that abstract away the underlying components.
- **langres.core** (Low-Level / "Power-User"): For 20% of use cases. This is the "PyTorch" layer, providing the core, composable "Lego bricks" (like Module, Optimizer, Blocker) for data scientists to build bespoke pipelines.

## The Two-Layer API

langres is designed to be accessible to all skill levels without sacrificing power.

### 1. The langres.tasks Layer (Speed-to-Value)

This high-level API is the recommended entry point. You instantiate a Task (the operation) and provide it with the two key components: a Blocker (the task-specific data transformer/normalizer) and a Flow (the entity-specific comparator).

This composition, inspired by scikit-learn's Pipeline (Transformers + Estimator), is what makes the architecture flexible. The Blocker handles the "tricks" (like schema-mapping), allowing the Flow (the "brain") to be reused across tasks.

**Use Case:** A developer needs to deduplicate company records AND link them from a separate Salesforce source.

```python
from langres.tasks import DeduplicationTask, EntityLinkingTask
from langres.flows import CompanyFlow # A pre-built "brain" from our library
from langres.blockers import DedupeBlocker, LinkingBlocker # Pre-built "normalizers"
from langres.data import SyntheticGenerator

# --- Component 1: The "Brain" (Reusable) ---
# Load a pre-built "Flow" for company data.
# This brain is the same for both tasks.
flow = CompanyFlow()

# --- Use Case A: Deduplication ---

# 1. Instantiate the high-level Task, plugging in the flow and a simple blocker.
task_dedupe = DeduplicationTask(
    flow=flow,
    blocker=DedupeBlocker() # This blocker assumes a single schema
)

# 2. "Compile" the task
# This internally runs the langres.core.Optimizer to tune
# the 'flow' logic and find the best thresholds.
gold_data = SyntheticGenerator(Company, ...).generate(5000)
task_dedupe.compile(gold_data, metric="bcubed_f1")

# 3. "Run" the task
# This internally handles blocking, runs the compiled flow,
# and calls the clusterer.
clusters = task_dedupe.run(all_companies)


# --- Use Case B: Entity Linking ---

# 1. Define the schema mapping "trick"
sfdc_map = {"sfdc_name": "name", "sfdc_addr": "address"}
internal_map = {"name": "name", "address": "address"}

# 2. Instantiate the Task, REUSING the *exact same brain*
# but plugging in a "smarter" blocker that handles schema mapping.
task_link = EntityLinkingTask(
    flow=flow, # <-- Reusing the same brain
    blocker=LinkingBlocker(source_map=sfdc_map, target_map=internal_map)
)

# 3. Compile and Run
task_link.compile(linking_gold_data, metric="bcubed_f1")
matches = task_link.run(source_data=sfdc_records, target_data=all_companies)
```

### 2. The langres.core Layer (Full Control)

This is the "PyTorch" layer that gives you complete control. The langres.tasks are built from these primitives. You can build any custom logic by subclassing the core components.

**Use Case:** A data scientist needs to match Products using a novel, multi-field logic combining rapidfuzz, embeddings, and a custom-trained PyTorch model.

```python
from langres.core import Module, Optimizer, Clusterer, CascadeBlocker
import torch.nn as nn

# 1. Define a custom PyTorch model
class MyCombinerModel(nn.Module):
    # ... (custom torch logic) ...

# 2. Subclass Module (the "Flow") to define a bespoke forward pass
class MyProductFlow(Module):
    def __init__(self):
        self.combiner_model = MyCombinerModel()
        self.embed_sim = EmbedSim()
        self.string_threshold = 0.8 # A tunable hyperparameter

    def _calculate_features(self, pair):
        # ... (logic using rapidfuzz, numeric diffs, etc.) ...
        return features

    def forward(self, candidates):
        # ... (custom cascade and feature combination logic) ...
        features = self._calculate_features(candidates)
        score = self.combiner_model(features)
        return PairwiseJudgement(score=score, ...)

# 3. Manually wire up the core components
flow = MyProductFlow()
optimizer = Optimizer(metric="bcubed_f1")
blocker = CascadeBlocker(...) # Manually configure

# 4. Manually orchestrate the two-stage optimization
# Stage 1: Train the learnable weights
trained_flow = optimizer.finetune(flow, gold_data)

# Stage 2: Find the best HPs (e.g., 'string_threshold')
# and prompts using the *already-trained* model
compiled_flow = optimizer.compile(trained_flow, gold_data)

# 5. Manually run the pipeline
# The compiled_flow now has both trained weights and tuned HPs
judgements = compiled_flow.forward(blocker.stream(all_products))
clusters = Clusterer().cluster(judgements) # Use threshold found by optimizer
```

## Core Architecture: The Five Pillars (langres.core)

langres is built on five core components that form the low-level, end-to-end pipeline.

### Blocker (Candidate Generation & Normalization)

This is the "Transformer" in the sklearn pipeline analogy. It has two jobs:

- **Generate Pairs:** Efficiently find candidate pairs to compare.
- **Normalize Schema:** Transform raw data from one or more sources into the single, clean, internal Pydantic schema that the Flow expects.

Key methods:
- `stream(data)`: For Deduplication (Use Case 1).
- `stream_against(source, target)`: For Entity Linking (Use Case 2) and Fuzzy FK (Use Case 10).
- `stream(datasets: List[...])`: For Record Linkage (Use Case 3).

### Module (The "Brain" / Flow)

This is the "Estimator" in the sklearn pipeline. It is the central component where you define your matching logic in a `forward()` pass.

**Operates on a Normalized Schema:** The Flow is designed to run on a clean, internal schema. It is the Blocker's job to perform all data-source-specific mapping before the data ever reaches the Flow.

Comparison strategies:
- **Classical:** Use rapidfuzz for fast, syntactic checks.
- **Semantic:** Use EmbedSim (via sentence-transformers) for semantic meaning.
- **Heuristic:** Write custom Python for numeric or categorical logic (e.g., price % difference).
- **Learnable:** Embed a torch.nn.Module to learn the optimal combination of signals.

### Clusterer (Entity Formation)

Consumes the PairwiseJudgement stream and builds the final entity clusters.

- Powered by **networkx** for efficient graph-based clustering (connected_components).
- Supports **scipy.cluster.hierarchy** for more noise-resistant hierarchical clustering.
- Accepts `cannot_link` constraints to prevent known false positives (Use Case 9).

### Optimizer (The "Compiler")

This is the "killer feature." It replaces weeks of manual tuning with an automated, data-driven process. It is a pragmatic, multi-stage harness that separates model training from hyperparameter tuning.

**Optimizes for the Right Goal:** Maximizes cluster-level metrics (BCubed F1, V-Measure), not just simple pairwise accuracy.

- **`finetune(flow, ...)`:** The first stage. This method only trains the learnable weights (i.e., torch.nn.Module components) within your Flow, returning the flow with the trained model.
- **`compile(flow, ...)`:** The second stage. This method only handles Hyperparameter Optimization (HPO) and Prompt Optimization.
  - Uses **Optuna** to find the optimal numeric values for all parameters (e.g., string_sim_threshold, name_weight).
  - Uses **DSPy** to tune any LlmJudge modules.

### Canonicalizer (Master Record Creation)

The "last mile" module that turns clusters of IDs into a final, "golden" master dataset (Use Case 4).

Provides declarative survivorship logic (e.g., "most recent," "most frequent," "merge all non-null") to create the canonical entity.

## Production-Ready by Design

- **Pydantic-First:** All data contracts are Pydantic models for fail-fast validation, IDE auto-complete, and powering the SyntheticGenerator.
- **Full Observability:** The PairwiseJudgement object carries full provenance, score_type, and reasoning, providing a complete audit trail for every decision.
- **Cost & Safety:** Designed for production with cost-aware cascades, PII redaction hooks, and budget-aware optimization.

## What langres Is (And Is Not)

langres is designed to be a lightweight, flexible, and best-in-class component for entity resolution.

### langres is the ideal solution for:

**Batch, Attribute-Based Resolution:** Its core strength.

Use Cases:
- Deduplication
- Record Linkage
- Entity Linking / Fuzzy FK
- Master Data Creation
- Privacy-Preserving Linkage (by implementing PPRL logic in the Module)
- Training ER "Brains": Creating and optimizing a portable Module that can be imported and used by other systems (like a streaming app).

### langres is not designed to be:

- **A Streaming System (UC 5):** langres is a batch-oriented pipeline. It can train the model that a streaming system (like Flink or Kafka Streams) would use for online, low-latency scoring.
- **A Temporal Graph Database (UC 6):** Its clustering model does not support entity splits (e.g., A -> {B, C}), which is a feature of temporal (mergers & acquisitions) analysis.
- **A Collective Inference Engine (UC 7):** The Module is stateless and pairwise. It cannot natively perform joint inference on a graph's relational structure.
