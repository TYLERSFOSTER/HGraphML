# Changelog

All notable changes to `HGraphML` will be documented here.

This project follows semantic-versioning vocabulary, but early releases should
be read as research milestones rather than stability promises.

## Unreleased

Tensorization compatibility bridge.

- Updated the `state-collapser` dependency pin to the public `state_collapser`
  `v0.7.0` tag.
- Added `build_encoding_registry(...)`, a thin HGraphML adapter helper around
  upstream `EncodingRegistry.from_tower(...)`.
- Added compatibility tests covering registry construction, JSON-safe registry
  metadata, base state and edge encodability, state-cell and action-cell
  encodability, and HGraphML node/edge fiber coverage.
- Documented that this is shared tower encoding compatibility for future
  tensorization work, not full graph-message tensorization, RL transition
  tensorization, or benchmark-backed speed-up.

## 0.1.0 - Research Viability Release

Initial lightweight public research release.

This release establishes the first executable bridge from `state_collapser`
quotient towers into trainable graph message passing:

- Added the `TensorGraph` graph surface.
- Added a direct adapter from known graph data into `state_collapser` towers.
- Added node-fiber and edge-fiber readouts from tower tiers.
- Added deterministic uniform and fiber-normalized lifts.
- Added a learned PyTorch lift.
- Added message containers, pooling, edge-message MLPs, and readout helpers.
- Added `collapse_messages(...)` as the package-native orchestration call.
- Added a small supervised train-step helper and runnable learned-lift demo.
- Added tests, type checking, Ruff linting, build metadata, README, usage docs,
  API notes, design notes, and contributor guidance.

This release does not claim graph-ML speed-up. It demonstrates trainable
quotient-tower-backed message passing and sets up the benchmarking work needed
before performance claims are appropriate.
