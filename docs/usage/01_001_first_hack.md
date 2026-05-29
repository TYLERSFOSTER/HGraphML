# HGraphML First Usage Hack

## Status

This is the first practical usage note for `HGraphML`.

The package is not yet a mature graph ML framework. It is a first executable
bridge between:

```text
state_collapser quotient towers
    + known graph message passing
    + PyTorch autograd
```

The first honest use case is a small trainability proof: build a known graph,
construct a quotient tower over it, run message passing on a coarse tier, lift
messages back to the fine graph, compute a readout, and backpropagate through
the learned lift and message model.

## Mental Model

`state_collapser` was originally built around discovered RL state/action graphs.
`HGraphML` uses the same tower idea in a different posture:

```text
state_collapser RL posture:
    an agent explores an environment
    -> the discovered state/action graph grows over time
    -> quotient towers organize decision structure

HGraphML first posture:
    a graph ML problem already has a known graph
    -> the whole graph is treated as already discovered
    -> quotient towers organize message-passing structure
```

The current package does not yet claim that this is faster. The present claim is
that the computation is real, differentiable, and package-shaped.

## Run The Toy Demo

From the repository root:

```bash
uv run --extra dev python -m hgraphml.examples.learned_lift_demo
```

The demo builds a repeated-motif graph, collapses it through a
`state_collapser` label schema, trains a learned lift for a few steps, and prints
the initial and final toy losses.

Expected shape of the output:

```text
initial_loss=...
final_loss=...
```

The exact numbers may change as the toy objective changes. The important first
milestone properties are:

- the demo runs,
- the loss is finite,
- gradients reach trainable parameters,
- the optimizer changes parameters.

## Build A Toy Graph

```python
from hgraphml.examples.toy_graphs import make_repeated_motif_graph

graph = make_repeated_motif_graph()
```

The toy graph has two repeated directed motifs and one bridge edge:

```text
motif 0: 0 -> 1 -> 2, 0 -> 2
motif 1: 3 -> 4 -> 5, 3 -> 5
bridge:  2 -> 3
```

Motif edges receive the label `"motif"`; the bridge receives the label
`"bridge"`. This makes it possible to ask `state_collapser` to collapse by a
label block.

## Make Node Features

```python
from hgraphml.examples.toy_graphs import make_toy_node_features

node_features = make_toy_node_features(seed=0, feature_dim=8)
```

Node features are ordinary rank-2 PyTorch tensors with shape:

```text
(node_count, feature_dim)
```

## Build A Tower Bundle

```python
from state_collapser.tower.partition import LabelBlockSchema

from hgraphml.adapters import build_tower_bundle

schema = LabelBlockSchema.from_labels(("motif",))
tower_bundle = build_tower_bundle(graph, contraction_schema=schema)
```

The adapter treats the graph as fully known. It registers all nodes and edges
with `state_collapser`, builds a partition tower, and returns an HGraphML-facing
`TowerBundle` containing:

- the original fine graph,
- the underlying `state_collapser` partition tower,
- node fibers by tier,
- edge fibers by tier,
- coarse `TensorGraph` objects by tier.

## Build An Encoding Registry

```python
from hgraphml.adapters import build_encoding_registry

encoding_registry = build_encoding_registry(tower_bundle)
```

This builds the shared upstream `state_collapser` encoding vocabulary for the
same partition tower that HGraphML uses for message passing. The registry can
encode base states, base edges, state cells, action cells, tiers, and related
tower metadata.

The registry metadata is JSON-safe:

```python
import json

json.dumps(encoding_registry.to_dict())
```

This is the current bridge toward tensorized HGraphML kernels. It is not an RL
training-batch path: this usage does not require `ActionSelectionInput`,
`TrainingTransition`, or `TorchDecisionBatch`.

## Create Message Model And Lift

```python
from hgraphml.lifts import LearnedFiberLift
from hgraphml.messages import EdgeMessageMLP

message_model = EdgeMessageMLP(feature_dim=8, message_dim=8, hidden_dim=32)
lift = LearnedFiberLift(message_dim=8, context_dim=16, hidden_dim=32)
```

`EdgeMessageMLP` consumes source-node and target-node features for each coarse
edge. `LearnedFiberLift` takes each coarse message and a fine-edge context, then
learns how to distribute messages back over the fine edges in each coarse-edge
fiber.

For the default fine-edge context, `context_dim` is:

```text
source feature dim + target feature dim
```

So with 8-dimensional node features and no explicit edge features, the context
dimension is `16`.

## Call `collapse_messages(...)`

```python
from hgraphml import collapse_messages

result = collapse_messages(
    graph=graph,
    node_features=node_features,
    message_model=message_model,
    lift=lift,
    tower_bundle=tower_bundle,
)
```

The orchestration flow is:

```text
fine node features
    -> mean-pool to coarse nodes by node fiber
    -> run coarse edge message model
    -> build fine edge contexts
    -> lift coarse messages over fine edge fibers
    -> incoming-sum readout on the fine graph
```

The result contains:

- `graph`, the original fine `TensorGraph`,
- `tower_bundle`, the tower/fiber data used,
- `coarse_tier`, the tier selected for coarse message passing,
- `coarse_messages`, edge messages on the selected coarse graph,
- `fine_messages`, lifted messages on the original fine graph,
- `node_readout`, incoming-message sums at fine nodes.

## Run One Train Step

```python
import torch

from hgraphml.training import make_teacher_node_targets, train_step

target = make_teacher_node_targets(graph, node_features, output_dim=8, seed=100)
optimizer = torch.optim.Adam(
    list(message_model.parameters()) + list(lift.parameters()),
    lr=0.01,
)

step = train_step(
    graph=graph,
    node_features=node_features,
    target=target,
    message_model=message_model,
    lift=lift,
    optimizer=optimizer,
    loss_fn=torch.nn.MSELoss(),
    tower_bundle=tower_bundle,
)
```

The train-step result records:

- the scalar loss,
- the full `HGraphMLResult`,
- gradient diagnostics,
- viability diagnostics.

Useful first checks:

```python
assert step.viability_diagnostics.loss_is_finite
assert step.viability_diagnostics.parameters_changed
assert step.gradient_diagnostics.nonzero_gradient_count > 0
```

## What This Does Not Yet Do

This first hack intentionally does not provide:

- speed benchmarks,
- a PyG/DGL adapter,
- exact belief propagation,
- dynamic graph updates,
- mature experiment tracking,
- checkpointing,
- GPU/device hardening beyond normal PyTorch tensor behavior.

The point is to make the core bridge real before optimizing it.
