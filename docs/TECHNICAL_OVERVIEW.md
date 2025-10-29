# langres: Technical Documentation & API Reference

Welcome to the langres documentation. This document provides a deep dive into the two-layer API, the core architectural pillars, and the data contracts that power the library.

## 1. The langres Two-Layer API

langres is designed with a two-layer API to be both easy to use for common tasks and fully extensible for complex, bespoke problems.

- **langres.tasks** (High-Level): The "out-of-the-box" API. You compose pre-built components to solve a specific task (e.g., DeduplicationTask). This is the recommended entry point.
- **langres.core** (Low-Level): The "power-user" API. This gives you direct access to the base classes (Module, Blocker, etc.) to build entirely new logic from scratch.

## 2. The Abstraction Layer: langres as a "Glue" Framework

A primary goal of langres is to act as a powerful "glue" framework, simplifying and abstracting best-in-class libraries into a single, cohesive workflow. For contributors and advanced users, it's important to understand what langres is managing under the hood.

### core.Optimizer (Optimization Harness)

The Optimizer is a sophisticated abstraction over several powerful optimization libraries:

- **Optuna:** Used as the engine for Hyperparameter Optimization (HPO). langres abstracts the entire study and trial process, allowing you to simply define a metric and let the optimizer find the best numeric thresholds in your Flow.
- **DSPy:** Used as the preferred engine for prompt optimization. When a Flow includes an LLM-based component, the Optimizer can automatically run a DSPy compilation loop to find the optimal prompt templates and few-shot examples.

### core.Blocker (Candidate Generation)

The Blocker abstracts the complex logic of high-recall candidate generation:

- **ANN Libraries:** In-memory blocking (e.g., blockers.EmbedBlocker) abstracts libraries like faiss-cpu or hnswlib, managing index creation and ANN search.
- **External Frameworks:** We plan to provide wrappers for powerful, dedicated blocking libraries like BlockingPy to offer advanced strategies (e.g., Q-gram, Sorted-Neighborhood) as pre-built Blocker types.

### core.Clusterer & core.Evaluator (Metrics & Graph)

These components abstract the standard libraries for graph math and metrics:

- **networkx & scipy.cluster.hierarchy:** The Clusterer uses these libraries to perform the actual graph clustering.
- **scikit-learn.metrics:** The Evaluator uses sklearn for all standard pairwise metrics (Precision, Recall, F1).
- **BCubed F1:** We use a vetted internal implementation (or er-metrics) for this critical cluster-level metric.
- **pytrec_eval:** This IR library is used internally by the Optimizer for the specialized sub-task of tuning Blocker recall.

### data.ReviewQueue (Human-in-the-Loop)

The HITL system is abstracted into two parts:

- **data.ReviewQueue:** This is the storage backend (e.g., a simple SQLite database) that Tasks write to.
- **langres.ui (Coming Soon):** langres will provide a pre-built, standalone Streamlit application that reads from this queue, provides a clean UI for human labelers, and writes decisions back. This separates the storage logic from the visual application.

### Observability & Tracing (TBD)

The design for full tracing (e.g., via OpenTelemetry) is still to be determined. The foundation for this is the PairwiseJudgement's provenance field, which is designed to capture all necessary metadata for a future tracing system.

## 3. High-Level API: langres.tasks

The tasks layer provides pre-wired orchestrators that handle the plumbing of a full ER pipeline (blocking, optimizing, clustering, etc.).

You instantiate a Task and provide it with several components:

- **Blocker:** The component responsible for finding candidate pairs and normalizing their data.
- **Flow:** The "brain" responsible for comparing the normalized pairs.
- **ReviewQueue (Optional):** A utility to handle Human-in-the-Loop (HITL) for uncertain matches.

### tasks.DeduplicationTask

**Definition:** A task orchestrator for deduplicating a single dataset (Use Case 1).

**What it does:**

- Uses the provided Blocker's `.stream(data)` method to generate candidate pairs from within the dataset.
- Uses the Optimizer (via `.compile()`) to tune the provided Flow for this task.
- Uses the Clusterer (via `.run()`) to generate the final entity clusters.
- If a `review_queue` and `review_threshold` are provided, it flags uncertain pairs (e.g., scores between 0.6 and 0.8) and sends them to the queue for manual review instead of auto-clustering them.

**Example:**

```python
from langres.tasks import DeduplicationTask
from langres.flows import CompanyFlow
from langres.blockers import DedupeBlocker
from langres.data import SyntheticGenerator, ReviewQueue

# 1. Load the pre-built brain (Flow) and a simple Blocker
flow = CompanyFlow()
blocker = DedupeBlocker() # Assumes a single, clean schema

# 2. (Optional) Set up a review queue for HITL
review_queue = ReviewQueue(db_path="./reviews.sqlite")

# 3. Instantiate the task
task = DeduplicationTask(
    flow=flow, 
    blocker=blocker,
    review_queue=review_queue,
    review_threshold=(0.6, 0.8) # Flag scores in this range for HITL
)

# 4. Compile (optimize) the task using synthetic data
gold_data = SyntheticGenerator(Company).generate(5000)
task.compile(gold_data, metric="bcubed_f1")

# 5. Run the task
# This will create clusters AND populate 'reviews.sqlite'
clusters = task.run(all_my_company_data)
```

### tasks.EntityLinkingTask

**Definition:** A task orchestrator for linking one source dataset to one target (authoritative) dataset (Use Case 2).

**What it does:**

- Uses the provided Blocker's `.stream_against(source, target)` method. This is where schema mapping logic lives.
- Composes and optimizes the pipeline just like the DeduplicationTask.
- The final output is a set of matches (links), not clusters.
- Also supports ReviewQueue for HITL on uncertain links.

**Example:**

```python
from langres.tasks import EntityLinkingTask
from langres.flows import CompanyFlow
from langres.blockers import LinkingBlocker # A smart, schema-mapping blocker

# 1. Load the same brain, but a different blocker
flow = CompanyFlow() # <-- The brain is reusable!

# 2. Configure the blocker with the schema mapping "trick"
sfdc_map = {"sfdc_name": "name", "sfdc_addr": "address"}
internal_map = {"name": "name", "address": "address"}
blocker = LinkingBlocker(source_map=sfdc_map, target_map=internal_map)

# 3. Instantiate the task
task = EntityLinkingTask(flow=flow, blocker=blocker)

# 4. Compile and run
task.compile(linking_gold_data, metric="pairwise_f1")
matches = task.run(source_data=sfdc_records, target_data=all_my_company_data)
```

## 4. High-Level API: langres.flows & langres.blockers

These are the pre-built "pluggable" components for the tasks layer.

### langres.flows

These are pre-written subclasses of `langres.core.Module`.

- **flows.CompanyFlow:** A pre-built "brain" for matching company entities. Its `forward()` pass already knows how to compare names (using rapidfuzz), addresses (using EmbedSim), and other common fields.
- **flows.ProductFlow:** A pre-built "brain" for matching products. Its `forward()` pass knows how to compare titles, descriptions, categories, and prices (using numeric logic).

### langres.blockers

These are pre-written subclasses of `langres.core.Blocker`.

- **blockers.DedupeBlocker:** A simple blocker for the deduplication task. It assumes a single schema and generates internal pairs.
- **blockers.LinkingBlocker:** A sophisticated blocker for linking tasks. Its `__init__` method accepts `source_map` and `target_map` dictionaries to perform schema normalization before data is passed to the Flow.

## 5. Core API: The Five Pillars (langres.core)

This is the low-level "PyTorch" layer. You use these base classes to build your own custom components from scratch.

### core.Blocker (Base Class)

**Definition:** The Blocker is the Data Loader & Transformer of the pipeline. It has two jobs:

- **Generate Pairs:** Efficiently find candidate pairs (e.g., via ANN search) to avoid N² comparisons.
- **Normalize Schema:** Act as the ETL layer. It transforms raw data from one or more sources into the single, clean, internal Pydantic schema that the Flow expects.

**What it's not:** It does not compare records. It only loads and normalizes them.

**Key Methods:**

- `stream(data: List[Any]) -> Iterator[ERCandidate]`
- `stream_against(source: List[Any], target: List[Any]) -> Iterator[ERCandidate]`
- `stream(datasets: List[List[Any]]) -> Iterator[ERCandidate]`

**Example (Custom Blocker):**

```python
from langres.core import Blocker, PydanticBaseModel

class MyInternalSchema(PydanticBaseModel):
    id: str
    name_field: str
    text_field: str

class MyCustomBlocker(Blocker):
    def __init__(self, name_map: str, text_map: str):
        # This blocker is configured with the user's column names
        self.name_map = name_map
        self.text_map = text_map
        # ... initialize ANN index ...

    def stream(self, data: List[dict]) -> Iterator[ERCandidate[MyInternalSchema]]:
        # 1. Find pairs using ANN logic (not shown)
        for raw_a, raw_b in self.find_pairs(data):

            # 2. Normalize schema
            norm_a = MyInternalSchema(
                id=raw_a["uuid"],
                name_field=raw_a[self.name_map],
                text_field=raw_a[self.text_map]
            )
            norm_b = MyInternalSchema(
                id=raw_b["uuid"],
                name_field=raw_b[self.name_map],
                text_field=raw_b[self.text_map]
            )

            # 3. Yield the clean, standardized pair
            yield ERCandidate(left=norm_a, right=norm_b)
```

### core.Module (Base Class - The "Flow")

**Definition:** The Module (or "Flow") is the "Brain" of the pipeline. It is the central Estimator that performs the pairwise comparison.

**What it's not:** It is not a data loader. It must operate on the clean, normalized schema provided by the Blocker. This separation of concerns is what makes it reusable.

**Key Methods:**

- `__init__(self)`: Define your components (e.g., `self.embed_sim = EmbedSim()`, `self.model = MyTorchModel()`).
- `forward(self, candidates: Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]`: Your custom comparison logic.

**Example (Custom Flow):**

```python
from langres.core import Module, PairwiseJudgement
import rapidfuzz.fuzz
import torch.nn as nn

class MyCombiner(nn.Module):
    # ... (PyTorch logic to combine 3 features) ...

class MyProductFlow(Module):
    def __init__(self):
        # Init all components, from classical to learnable
        self.embed_sim = EmbedSim(model="e5-small")
        self.combiner_model = MyCombiner() # PyTorch weights
        self.name_weight = 0.5 # A tunable hyperparameter

    def _calculate_features(self, pair: ERCandidate[MyInternalSchema]):
        name_sim = rapidfuzz.fuzz.WRatio(pair.left.name_field, pair.right.name_field)
        desc_sim = self.embed_sim(pair.left.text_field, pair.right.text_field)
        # ... any other custom logic ...
        return torch.tensor([name_sim, desc_sim])

    def forward(self, candidates: Iterator[ERCandidate[MyInternalSchema]]) -> Iterator[PairwiseJudgement]:
        self.combiner_model.eval()
        with torch.no_grad():
            for pair in candidates:
                # 1. Get features
                features = self._calculate_features(pair)

                # 2. Run learnable model
                combined_score = self.combiner_model(features).item()

                yield PairwiseJudgement(
                    left_id=pair.left.id,
                    right_id=pair.right.id,
                    score=combined_score,
                    score_type="calibrated_prob",
                    decision_step="combiner_model",
                    provenance={"model_version": "v1.2"}
                )
```

### core.Clusterer (Base Class)

**Definition:** Consumes the PairwiseJudgement stream and builds the final entity clusters.

**Key Methods:**

- `cluster(self, judgements: Iterator[PairwiseJudgement], constraints: List[CannotLinkPair] = None) -> List[Set[str]]`

**Features:**

- **method:** Use "connected_components" (fast, default, uses networkx) or "hierarchical" (more noise-robust, uses scipy.cluster.hierarchy).
- **constraints:** Pass a list of `(id_a, id_b)` tuples that are known non-matches (Use Case 9). The clusterer will respect these, even if the score is high.

**Example:**

```python
from langres.core import Clusterer

clusterer = Clusterer(method="hierarchical", threshold=0.75)

# Define known non-matches
constraints = [("id_123", "id_456")] 

clusters = clusterer.cluster(judgements_stream, constraints=constraints)
```

### core.Optimizer (Base Class)

**Definition:** The "Compiler" for your Flow. It's a multi-stage harness that automates training and tuning.

**What it does:** It runs your Flow over the gold_data many times to find the settings that maximize your chosen metric.

**Key Methods:**

**`compile(self, flow: Module, gold_data: List[GoldPair]) -> CompiledFlow`:**

- **Role:** Hyperparameter Optimization (HPO).
- **Under the Hood:** Uses Optuna to tune any numeric parameters in your flow (e.g., `self.name_weight`, `self.string_threshold`).
- **Also:** Uses DSPy to tune any prompts in your flow (if using LlmJudge).

**`finetune(self, flow: Module, gold_data: List[GoldPair]) -> Module`:**

- **Role:** Model Training.
- **Under the Hood:** Runs a PyTorch training loop to train the weights of any torch.nn.Module (like MyCombiner) found inside your flow.

**Example:**

```python
from langres.core import Optimizer

flow = MyProductFlow() # The untrained, untuned flow
optimizer = Optimizer(metric="bcubed_f1") # Optimize for cluster quality

# 1. Train the PyTorch weights
trained_flow = optimizer.finetune(flow, gold_data, epochs=10)

# 2. Tune the hyperparameters (weights, thresholds)
# The optimizer will find the best value for 'self.name_weight'
compiled_flow = optimizer.compile(trained_flow, gold_data)

# compiled_flow now contains the trained model AND the best HPs
```

### core.Canonicalizer (Base Class)

**Definition:** The "last mile" module for Master Data Creation (Use Case 4). It turns a list of cluster IDs into a single, merged, "golden" master dataset.

**Key Methods:**

- `canonicalize(self, clusters: List[Set[str]], all_data: Dict[str, Any]) -> List[PydanticBaseModel]`

**Features:** You configure it with a list of "survivorship rules" per field.

**Example:**

```python
from langres.core import Canonicalizer

# Define survivorship logic
rules = {
    "name": "most_frequent",
    "address": "most_recent", # Assumes 'all_data' has timestamps
    "phone_numbers": "merge_unique"
}

canonicalizer = Canonicalizer(rules=rules, output_schema=Company)

# 'all_data' is a simple dict lookup: {id -> raw_record}
master_dataset = canonicalizer.canonicalize(clusters, all_data)
```

## 6. Core API: langres.data

This module provides utilities for creating and managing the data that powers the Optimizer and Tasks.

### data.SyntheticGenerator

**Definition:** A utility that uses LLMs to create a gold_data set for training and compiling your Flow. It creates realistic variations (typos, synonyms, abbreviations) of your data.

**Example:**

```python
from langres.data import SyntheticGenerator

# 'Company' is your Pydantic schema
gen = SyntheticGenerator(schema=Company, hints={"name": "add typos"})

# Creates 5000 (candidate_pair, label) tuples
gold_data = gen.generate(n_pairs=5000)
```

### data.ReviewQueue

**Definition:** A utility for managing the Human-in-the-Loop (HITL) workflow. It provides a simple storage backend (like SQLite or a file) for uncertain pairs flagged by a Task.

**What it does:**

- Receives uncertain pairs from a Task (based on the `review_threshold`).
- Stores them persistently.
- Allows an external UI (e.g., a Streamlit app you build) to read these pairs, display them to a human, and save the labels. Note: A pre-built `langres.ui` app will be provided for this.
- Lets you export these verified labels to be added back into your `gold_data` set, creating an "active learning" loop.

**Example:**

```python
from langres.data import ReviewQueue

# 1. Init queue (in your main task script)
review_queue = ReviewQueue(db_path="./reviews.sqlite")

# ... (task is configured with this queue and run) ...

# 2. In a separate review app (e.g., app.py):
# You can run the pre-built: $ langres-ui --db-path ./reviews.sqlite
# Or build your own:
queue = ReviewQueue(db_path="./reviews.sqlite")

# Get one pair for a human to label
pair_to_review = queue.get_unlabeled_pair()

# ... (display pair_to_review in a UI) ...

# Save the human's decision
human_label = True # (from a button click)
queue.submit_label(pair_to_review.id, human_label)

# 3. Back in your main script, export labels to retrain
verified_pairs = queue.get_labeled_data()
gold_data.add(verified_pairs)

# Re-compile the task with the new, human-verified data
task.compile(gold_data, metric="bcubed_f1")
```

## 7. Core Data Contracts (Pydantic Models)

### ERCandidate[SchemaT]

The internal data wrapper passed into a Flow.

- `left: SchemaT`
- `right: SchemaT`
- `blocker_name: str`

### PairwiseJudgement

The rich data object passed out of a Flow. This is the auditable log of a decision.

- `left_id: str`
- `right_id: str`
- `score: float`: The combined score (0.0 to 1.0).
- `score_type: Literal["sim_cos", "prob_llm", "heuristic", "calibrated_prob", ...]`: What kind of score is this? Critical for calibration and clustering.
- `decision_step: str`: Which logic branch made this decision (e.g., "string_sim" or "llm_judge").
- `reasoning: Optional[str]`: The LLM's natural language explanation.
- `provenance: Dict[str, Any]`: A full audit trail (e.g., `{"model": "e5-small", "rapidfuzz_score": 0.85}`).