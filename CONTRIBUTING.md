# Contributing

Thanks for contributing to `HGraphML`.

This repository is a pre-alpha Python package that imports the quotient-tower
machinery of [`state_collapser`](https://github.com/TYLERSFOSTER/state_collapser)
into graph ML message passing. It has a real first vertical slice, but it is
not yet a mature graph ML framework. Contributions should preserve that honest
posture: make the bridge stronger, testable, and easier to use without
pretending the package has already solved benchmarking, framework integration,
or exact belief propagation.

## Scope

Contributions are welcome in areas such as:

- bug fixes,
- tests,
- documentation,
- package/release workflow,
- graph message-passing examples,
- lift operators,
- `state_collapser` adapter hardening,
- diagnostics and benchmark tooling,
- PyTorch training-surface improvements.

Because this package depends conceptually and technically on `state_collapser`,
many contributions should begin by deciding which repository should own the
change.

## Which Repository Should Receive The Change?

Use `HGraphML` for work about graph ML dataflow over quotient towers:

- `TensorGraph` and graph-feature surfaces,
- graph message-passing models,
- deterministic or learned lifts,
- node/edge fiber readouts for graph ML,
- training demos for message-passing computations,
- graph ML examples such as belief propagation or factor graphs,
- PyG/DGL/NetworkX-style adapters,
- HGraphML docs and release workflow.

Use `state_collapser` for upstream tower/runtime work:

- quotient-tower construction semantics,
- partition/state/action-cell internals,
- contraction schemas,
- tower snapshots and compatibility readouts,
- RL state/action graph discovery,
- Gymnasium/RL adapters,
- core runtime performance of tower construction,
- general changes that should benefit both RL and graph ML users.

If an HGraphML contribution needs a new tower API, prefer adding or requesting
that surface upstream in `state_collapser` rather than reimplementing it locally.
`HGraphML` should not silently grow its own private copy of `state_collapser`.

## Current Project Roadmap

`HGraphML` has moved past the first trainability proof and public GitHub
research release baseline. The current bridge has two real pieces:

- trainable quotient-tower-backed graph message passing, and
- compatibility with `state_collapser` `v0.7.0`'s shared `EncodingRegistry`.

The next technical gap is turning that shared encoding vocabulary into useful
graph-ML tensorization without pretending that HGraphML already has benchmarked
tensorized kernels. Serious benchmarking remains a release-shaping requirement:
the package needs controlled evidence about when quotient-tower message passing
is cheaper, comparable, or worse than direct flat message passing.

### Critical TODO

- ***Serious benchmarking:*** Build the benchmark surface that can justify the
  package beyond trainability. HGraphML needs controlled graph families,
  repeatable benchmark commands, flat-vs-quotient comparisons, tower-build cost
  accounting, lift/readout cost accounting, train-step timing, memory
  measurements where practical, schema comparisons, and artifact output. No
  public speed-up claim should be made until this exists.
- ***Benchmark-oriented graph families:*** Add graph generators that stress the
  quotient idea in different ways: repeated motifs, nearly repeated motifs,
  sparse bridges, dense local neighborhoods, bad-collapse controls, and graphs
  where quotienting should not help. These should make it possible to see both
  wins and failures.
- ***Benchmark baselines:*** Implement direct flat message-passing baselines
  next to quotient-tower message passing. Benchmarks should separate at least:
  tower construction, coarse message pass, lift, fine readout, backward pass,
  and full train step.
- ***Benchmark artifacts:*** Add structured benchmark outputs, not just terminal
  prints. At minimum, runs should emit machine-readable records containing graph
  family, graph size, schema, tier choice, lift type, seed, timing, message
  shapes, and package/git metadata.
- ***Tensorization bridge:*** Use the shared `EncodingRegistry` surface to
  design HGraphML tensorization around upstream state, edge, state-cell,
  action-cell, and tier IDs. This should remain shared tower encoding
  compatibility, not a local reimplementation of `state_collapser`'s RL
  tensorization stack.
- ***Adapter compatibility:*** Keep the current `state_collapser` compatibility
  contract explicit. HGraphML currently targets the public `state_collapser`
  `v0.7.0` tag for `EncodingRegistry` support. If HGraphML needs a more stable
  tower/fiber/readout API, add that upstream rather than introducing a local
  fallback tower.
- ***Real graph ML example:*** Add a first non-toy example, preferably a small
  belief-propagation or factor-graph-style message-passing problem. This should
  be chosen partly because it can become a meaningful benchmark case, not merely
  another demonstration script.
- ***Lift semantics:*** Clarify exact-vs-approximate lift semantics for uniform
  pullback, fiber-normalized disintegration, and learned lifts. Tests should
  distinguish algebraic exactness checks from approximation/training checks.
- ***Batch/device surfaces:*** Define conventions for batching, tensor devices,
  dtype movement, reusable tower bundles across train/eval loops, and how
  encoded tower IDs align with HGraphML tensors.
- ***Framework adapters:*** Decide whether the first serious adapter should be
  PyTorch Geometric, DGL, NetworkX, or a small explicit factor-graph surface.
- ***README and quickstart hardening:*** Keep the root README aligned with the
  actual package surface. The quickstart should always run from a fresh checkout
  with the documented commands.

### Required Before Any Speed-Up Or Scaling Claim

Do not claim that `HGraphML` accelerates graph ML until the repository contains
a benchmark surface that can make that statement measurable and falsifiable.
The toy learned-lift demo proves trainability, not speed.

A credible speed-up or scaling claim requires at least:

- ***A flat baseline:*** Implement direct fine-graph message passing beside the
  quotient-tower path, using comparable PyTorch code and comparable tensor
  shapes. The baseline must be part of the same benchmark harness, not an
  informal external comparison.
- ***A quotient path:*** Measure the actual HGraphML path end to end: tower
  bundle acquisition, coarse message passing, lift, fine readout, loss,
  backward pass, and optimizer step when training is included.
- ***Timing decomposition:*** Report separate timings for tower construction or
  reuse, coarse pass, lift, readout, backward pass, and full train step. A
  quotient method can only be understood if its overheads are visible.
- ***Controlled graph families:*** Benchmark repeated motifs, nearly repeated
  motifs, sparse bridges, dense local neighborhoods, bad-collapse controls, and
  graphs where quotienting should not help. The package needs to show where the
  idea wins, where it ties, and where it loses.
- ***Schema comparisons:*** Compare contraction schemas and tier choices. A
  speed-up claim should name the schema assumptions under which it holds.
- ***Repeated runs and seeds:*** Run repeated trials with fixed seed reporting.
  Single-run timing anecdotes are not enough.
- ***Negative controls:*** Include graph families and schemas where quotienting
  should add overhead or fail to help. If the benchmark only contains friendly
  examples, the result is not credible.
- ***Memory measurements:*** Track memory where practical, especially when
  comparing cached tower/fiber structures against flat message passing.
- ***Artifact output:*** Emit machine-readable benchmark records containing
  graph family, size, edge count, schema, tier, lift, model dimensions, seed,
  timing breakdown, memory fields where available, package version, and git
  commit.
- ***Public documentation:*** Publish the benchmark command, artifact schema,
  and interpretation rules before using benchmark language in README, release
  notes, papers, or outreach.

Until this exists, safe wording is:

```text
HGraphML demonstrates trainable quotient-tower-backed graph message passing.
```

Unsafe wording is:

```text
HGraphML speeds up graph ML.
```

### Non-Critical TODO

- Add richer examples for multiple contraction schemas.
- Add visualization of coarse graphs, fibers, and lifted messages.
- Add more learned-lift variants.
- Add optional edge-feature examples.
- Add docs showing how encoded tower IDs map onto graph-message tensors.
- Add docs comparing `HGraphML` with direct flat message passing.
- Add a glossary for quotient, fiber, lift, coarse tier, and fine tier terms.
- Add an outreach-oriented conceptual explainer after the technical surface is
  stable.

## Development Setup

The project uses:

- a `src/` layout,
- `pyproject.toml`,
- Python `>=3.11,<3.13`,
- `uv` for the current local workflow,
- PyTorch,
- `state-collapser` as a declared package dependency.

Recommended local setup:

```bash
uv sync --extra dev --group dev
```

For the lightweight GitHub research release, `state-collapser` is pinned to a
public upstream tag in `pyproject.toml`. Before a PyPI release, prefer switching
that to a normal registry dependency after `state-collapser` is published there.
If you need to test against an unpublished local upstream change, keep that
override local and do not commit machine-specific paths.

## Local Validation

Run these checks before opening a pull request or merging a branch:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev mypy
```

Run the demo when your change touches message passing, lifting, training, or the
adapter:

```bash
uv run --extra dev python -m hgraphml.examples.learned_lift_demo
```

If your change touches only documentation, a full test pass is still preferred
before release-facing work, but a documentation-only PR may report that code
tests were not rerun.

## Repository Layout

Current important repository areas:

```text
src/hgraphml/
  adapters/
  diagnostics/
  examples/
  graph/
  lifts/
  messages/
  training/

tests/
  adapters/
  diagnostics/
  examples/
  graph/
  lifts/
  messages/
  training/

docs/
  api_notes/
  design/
  engineer_continuity/
  prime_directive/
  usage/
```

High-level roles:

- `adapters/`
  - bridges from known graph data to upstream `state_collapser` towers
- `diagnostics/`
  - gradient and trainability checks for first experiments
- `examples/`
  - toy graphs and runnable viability demos
- `graph/`
  - tensor graph and fiber data structures
- `lifts/`
  - rules for lifting coarse messages back to fine edges
- `messages/`
  - message containers, message models, pooling, and readout
- `training/`
  - tiny training helpers and toy objectives

## Public API Expectations

Only names exported from:

```text
src/hgraphml/__init__.py
```

should be treated as public-feeling package surface right now.

Current top-level exports:

```python
from hgraphml import HGraphMLResult, __version__, collapse_messages
```

Subpackage imports are usable, typed, and tested, but they remain pre-alpha.
Contributors should avoid treating every importable module as stable public API.

## Contribution Workflow

The normal contribution flow is:

1. Branch from `main`.
2. Make a focused change.
3. Add or update tests in the same change.
4. Update README, usage docs, or API notes if behavior or workflow changes.
5. Run local validation.
6. Open a pull request or propose the merge.

Keep changes scoped. Avoid mixing unrelated refactors, docs rewrites, packaging
work, and runtime behavior changes unless they are part of one approved design
move.

## Design Authority

This repository is design-sensitive because it is importing a mathematical
runtime idea from `state_collapser` into graph ML.

Before changing structural behavior, read the relevant design documents:

- [docs/design/01_001_lifting_strategies_for_hierarchical_graph_message_passing.md](./docs/design/01_001_lifting_strategies_for_hierarchical_graph_message_passing.md)
- [docs/design/01_002_hgraphml_first_import_blueprint.md](./docs/design/01_002_hgraphml_first_import_blueprint.md)
- [docs/design/01_003_hgraphml_first_import_implementation_gameplan.md](./docs/design/01_003_hgraphml_first_import_implementation_gameplan.md)
- [docs/design/01_004_hgraphml_first_import_implementation_log.md](./docs/design/01_004_hgraphml_first_import_implementation_log.md)
- [docs/design/tensorization/01_006_state_collapser_encoding_compatibility_blueprint.md](./docs/design/tensorization/01_006_state_collapser_encoding_compatibility_blueprint.md)
- [docs/design/tensorization/01_007_state_collapser_encoding_compatibility_implementation_gameplan.md](./docs/design/tensorization/01_007_state_collapser_encoding_compatibility_implementation_gameplan.md)
- [docs/design/tensorization/01_008_state_collapser_encoding_compatibility_implementation_log.md](./docs/design/tensorization/01_008_state_collapser_encoding_compatibility_implementation_log.md)

For upstream quotient/tower semantics, consult `state_collapser` design docs
and prefer upstream changes when the surface is not graph-ML-specific.

If code and design documents disagree, surface that disagreement explicitly.
Do not silently weaken quotient, fiber, or lift semantics to make an
implementation easier.

## Coding Expectations

Contributions should preserve the current package style:

- typed Python,
- small explicit data structures,
- tests for runtime behavior,
- minimal public surface,
- PyTorch-native tensor behavior,
- clear separation between adapter, graph, message, lift, and training code.

Contributors should not silently:

- reimplement `state_collapser` towers locally,
- reimplement `state_collapser` encoding registries or RL tensor batches
  locally,
- turn quotient towers into decorative metadata,
- detach learned-lift tensors in a way that breaks gradients,
- claim speed-up from the toy viability demo,
- replace a real training path with a scripted placeholder,
- add a heavy graph ML dependency without an adapter design.

## Testing Expectations

New runtime behavior should usually include tests in the matching package area:

```text
src/hgraphml/lifts/foo.py
tests/lifts/test_foo.py
```

Expected tests depend on the change, but common checks include:

- shape validation,
- invalid input validation,
- gradient flow,
- deterministic behavior under a seed,
- fiber coverage over all fine nodes or edges,
- adapter compatibility with `state_collapser`,
- train-step viability when training behavior changes.

## Documentation Expectations

Update documentation when a contribution changes:

- install or validation commands,
- public imports,
- README quickstart behavior,
- lift semantics,
- adapter behavior,
- training helper behavior,
- benchmark claims or benchmark workflow.

Use:

- [README.md](./README.md) for engineer-facing orientation and quick use,
- [docs/usage](./docs/usage) for workflow guides,
- [docs/api_notes](./docs/api_notes) for implemented surfaces,
- [docs/design](./docs/design) for design authority and implementation records.

## Release Expectations

Before a public release, verify:

- README quickstart runs from a clean checkout,
- tests, lint, and mypy pass,
- package metadata is current,
- release notes explain current limitations,
- no speed-up claim is made without benchmark artifacts,
- `state_collapser` compatibility is explicit,
- TODOs in this document still reflect the real roadmap.

## License

By contributing, you agree that your contribution is provided under the
repository's [MIT License](./LICENSE).
