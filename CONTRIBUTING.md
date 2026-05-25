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

`HGraphML` is currently moving from first viability proof toward public-release
readiness.

### Critical TODO

- ***Public release basics:*** Add or verify CI, package metadata, repository
  URLs, release tags, PyPI publishing configuration, and a changelog before
  claiming a public package release.
- ***README and quickstart hardening:*** Keep the root README aligned with the
  actual package surface. The quickstart should always run from a fresh checkout
  with the documented commands.
- ***Adapter compatibility:*** Document and test the supported
  `state_collapser` version range. If HGraphML needs a more stable tower/fiber
  readout API, add that upstream rather than introducing a local fallback tower.
- ***Real graph ML example:*** Add a first non-toy example, preferably a small
  belief-propagation or factor-graph-style message-passing problem.
- ***Lift semantics:*** Clarify exact-vs-approximate lift semantics for uniform
  pullback, fiber-normalized disintegration, and learned lifts. Tests should
  distinguish algebraic exactness checks from approximation/training checks.
- ***Benchmarking:*** Build a benchmark harness comparing flat message passing
  with quotient message passing on controlled graph families. No speed-up claim
  should be made until this exists.
- ***Batch/device surfaces:*** Define conventions for batching, tensor devices,
  dtype movement, and reusable tower bundles across train/eval loops.
- ***Framework adapters:*** Decide whether the first serious adapter should be
  PyTorch Geometric, DGL, NetworkX, or a small explicit factor-graph surface.
- ***Diagnostics and artifacts:*** Add structured benchmark/training artifacts
  so results can be inspected rather than hand-read from terminal output.

### Non-Critical TODO

- Add richer examples for multiple contraction schemas.
- Add visualization of coarse graphs, fibers, and lifted messages.
- Add more learned-lift variants.
- Add optional edge-feature examples.
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
- a sibling editable checkout of `state_collapser` during local development.

Recommended local setup:

```bash
uv sync --extra dev --group dev
```

The current `uv` source configuration expects:

```text
../state_collapser
```

to exist relative to the `HGraphML` repository. If you are not working in that
layout, install a compatible `state-collapser` package or adjust your local
development environment without committing machine-specific paths.

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
