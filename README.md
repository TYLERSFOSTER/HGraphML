# **H***ierarchical* • **Graph** • **M***achine* • **L***earning*</br> (HGraphML)

`HGraphML` is a lightweight Python package that realizes an insight that
[Abdullah N. Malik](https://abdullahnaeemmalik.github.io/) made while thinking
about the relationship between graph ML and the quotient-tower machinery behind
[`state_collapser`](https://github.com/TYLERSFOSTER/state_collapser). The basic
idea is that the same hierarchical graph-collapse tools used by
`state_collapser` for reinforcement-learning state/action graphs should also be
useful for graph ML systems where data flows across a known graph.

For `state_collapser`'s original RL application, the graph that data flows
across is the state/action graph of the underlying problem. The moving data can
be understood as the agent itself, as a Dirac delta of a single run, or as the
probabilistic flow induced by a policy over the state/action graph.

For `HGraphML`, the first target is graph message passing. A known graph is
handed to `state_collapser` as an already-discovered graph; `state_collapser`
builds a quotient tower; `HGraphML` runs message passing on a coarse tier; and a
lift maps coarse messages back to the fine graph. This is intended as the first
executable bridge toward hierarchical versions of graph ML methods such as
[belief propagation](https://en.wikipedia.org/wiki/Belief_propagation).

This repository does not yet claim speed-up. The current milestone is stricter
and smaller: prove that quotient-tower-backed graph message passing can sit
inside a trainable PyTorch computation without breaking autograd.

## Current Milestone

The package currently provides:

- a minimal `TensorGraph` surface for known directed graphs,
- a direct `state_collapser` adapter that builds quotient towers from known
  graphs,
- node-fiber and edge-fiber readouts from tower tiers,
- deterministic uniform and fiber-normalized lifts,
- a learned PyTorch lift,
- a small `collapse_messages(...)` orchestration call,
- a tiny train-step helper proving gradients and optimizer updates work.

The central call is:

```python
from hgraphml import collapse_messages

result = collapse_messages(
    graph=graph,
    node_features=node_features,
    message_model=message_model,
    lift=lift,
    contraction_schema=schema,
)
```

## Tiny Demo

```python
from hgraphml.examples.learned_lift_demo import run_learned_lift_demo

result = run_learned_lift_demo(steps=10, seed=0)
print(result.initial_loss, result.final_loss)
```

From a local checkout:

```bash
uv run --extra dev python -m hgraphml.examples.learned_lift_demo
```

## Documentation

- [First usage hack](docs/usage/01_001_first_hack.md)
- [First API surfaces](docs/api_notes/01_001_first_surfaces.md)
- [Implementation log](docs/design/01_004_hgraphml_first_import_implementation_log.md)
- [Design blueprint](docs/design/01_002_hgraphml_first_import_blueprint.md)
- [Implementation gameplan](docs/design/01_003_hgraphml_first_import_implementation_gameplan.md)

