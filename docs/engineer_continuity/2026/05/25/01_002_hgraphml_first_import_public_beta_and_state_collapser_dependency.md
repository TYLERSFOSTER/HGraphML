# Engineer Continuity Report
## 01_002_hgraphml_first_import_public_beta_and_state_collapser_dependency

## Date

2026-05-25

## Interval Covered

This report covers the `HGraphML` work completed after:

- [01_001_init.md]([HGraphML repository root]/docs/engineer_continuity/2026/05/21/01_001_init.md)

The prior report recorded the creation of the `HGraphML` repository in response
to Abdullah N. Malik's observation that the quotient-tower machinery behind
`state_collapser` should apply naturally to graph ML message-passing problems.

This interval covers the move from that initial repository note to a real
lightweight public-beta research package:

- foundational design discussion for lifting quotient-tier messages back to
  fine graph fibers,
- blueprint and Phase.Stage.Action implementation gameplan,
- first package implementation,
- direct import of `state_collapser` partition towers,
- trainable PyTorch message/lift/readout path,
- usage/API notes,
- public README and CONTRIBUTING hardening,
- CI and package build surface,
- public-release and local-path/security audits,
- `v0.1.0` tag,
- README banner image work,
- and the downstream dependency issue created by the later `state_collapser`
  public-history rewrite.

Release-readiness follow-up: immediately after this report was first written,
the dependency pin was updated from the now-deleted upstream `state_collapser`
`v0.5.0` tag to the clean public `v0.6.0` tag.

At the time of this report, `HGraphML` public `main` points at:

```text
7084c64 package README banner image
```

and the current tag is:

```text
v0.1.0
```

## Executive Status

At the beginning of the interval, `HGraphML` was only an idea and a repository
shell. It existed to explore Malik's observation:

```text
state_collapser quotient towers are not only RL state/action tools; they should
also scaffold graph ML computations whose dataflow already lives on a graph.
```

At the end of the interval, `HGraphML` has a real first vertical slice:

```text
known graph
    -> state_collapser partition tower
        -> coarse message passing
            -> deterministic or learned lift over node/edge fibers
                -> fine graph readout
                    -> PyTorch loss/backprop/optimizer step
```

The package currently proves trainability and integration viability. It does
not yet prove speed-up.

The intended public posture is:

```text
HGraphML demonstrates trainable quotient-tower-backed graph message passing.
```

The unsafe public posture remains:

```text
HGraphML speeds up graph ML.
```

The main technical gap is serious benchmarking, not another toy demo.

## Source Reconstruction Note

This report is reconstructed from:

- current `HGraphML` git history,
- current `README.md`,
- current `CONTRIBUTING.md`,
- `CHANGELOG.md`,
- `pyproject.toml`,
- `docs/design/01_001_lifting_strategies_for_hierarchical_graph_message_passing.md`,
- `docs/design/01_002_hgraphml_first_import_blueprint.md`,
- `docs/design/01_003_hgraphml_first_import_implementation_gameplan.md`,
- `docs/design/01_004_hgraphml_first_import_implementation_log.md`,
- `docs/design/public_release_audit/01_001_lightweight_research_public_release_readiness_audit.md`,
- `docs/design/public_release_audit/01_002_security_and_local_path_public_release_audit.md`,
- current source and test layout,
- and the live cross-repo discussion with `state_collapser`.

Relevant commits in this interval:

```text
0071637 Implement first HGraphML import surface
54a6164 pytest warning fix
989105a better public-facing docs
8d0b513 CONTRIBUTING.md TODOs aligned with PO goals
afc06f0 public release readiness and security audit
7084c64 package README banner image
```

The public tag is:

```text
v0.1.0 -> 7084c64
```

## Authorship And PO Attribution

The Project Owner supplied the central idea, framing, scope decisions, and
release judgment. Specifically, the PO:

- identified Malik's graph-ML observation as the reason to create `HGraphML`,
- connected `state_collapser`'s RL state/action graph flow to graph ML dataflow,
- requested two algebraic lifting strategies from tier `i+1` to tier `i`,
- observed that once lifting is specified, much of the package architecture is
  nearly automatic,
- directed the design notes to live in the `HGraphML` repo, not
  `state_collapser`,
- answered design questions inside the foundational design document,
- asked for an implementation blueprint and Phase.Stage.Action gameplan,
- authorized broad repo changes without repeated permission checks,
- reacted strongly when documentation work appeared to be dropped, which led to
  a fuller README/usage/API/log/documentation pass,
- requested `.gitignore` hardening before a broad `git add`,
- requested professional README and CONTRIBUTING docs with `state_collapser`
  cross-routing,
- clarified that the real `HGraphML` TODO is serious benchmarking,
- requested a hardcore public-release readiness audit,
- responded inside that audit,
- asked whether `state_collapser` not yet being public explained dependency
  release issues,
- chose to set `HGraphML` down temporarily while `state_collapser` was prepared
  as the upstream public dependency,
- and tagged the repo as `v0.1.0` for the first public beta/research release.

Codex supplied:

- foundational design synthesis,
- algebraic lifting strategy write-up,
- blueprint and gameplan documents,
- first implementation of the package,
- direct `state_collapser` adapter,
- PyTorch message/lift/training code,
- tests,
- documentation,
- public release audit,
- security/local-path audit,
- README/CONTRIBUTING hardening,
- CI/package metadata setup,
- release-version investigation,
- and this continuity reconstruction.

## Major Movement 1: The Foundational Design Was Written

The first substantive HGraphML document was:

- `docs/design/01_001_lifting_strategies_for_hierarchical_graph_message_passing.md`

This document records the core import:

```text
graph carrying local dataflow
    -> quotient/coarsening tower of that graph
        -> coarse computation on cheaper graph tiers
            -> lift/refine results back to finer tiers
```

The immediate motivating target was belief propagation, but the package was
defined more broadly:

```text
If a graph ML method can be expressed as local message passing or dataflow
across graph edges, then a state_collapser-style quotient tower should be able
to accelerate, organize, or regularize that computation by running part of the
computation on coarser graph tiers and lifting the resulting messages back to
finer tiers.
```

Two first lifting strategies were specified:

1. Uniform pullback lift.
2. Fiber-normalized disintegration lift.

The design also opened the door to learned lifts, which became essential in the
first implementation because the package needed to prove PyTorch trainability,
not just algebraic readout.

The design clarified that the hard part is not wrapping PyTorch. The hard part
is the algebraic question:

```text
Once a coarse quotient edge or cell has a message, how is that message assigned
back to the fine preimage edges or nodes?
```

That question is the graph-ML analogue of the RL question:

```text
Once a policy/value/control signal has been learned on tier i+1, how does it
lift to executable behavior at tier i?
```

## Major Movement 2: Blueprint And Gameplan Established The First Import

After design discussion and PO replies, Codex created:

- `docs/design/01_002_hgraphml_first_import_blueprint.md`,
- `docs/design/01_003_hgraphml_first_import_implementation_gameplan.md`.

The blueprint intentionally aimed at a first executable bridge rather than a
complete graph ML framework.

The gameplan's non-negotiable rule was:

```text
Do not silently reimplement state_collapser inside HGraphML.
```

This was important. HGraphML should prove that `state_collapser` can serve as
real upstream quotient-tower infrastructure. If the adapter could not import the
needed partition-tower surfaces, the correct response would have been to stop
and fix upstream, not to fork the tower logic locally.

The planned first package shape was:

- `TensorGraph` for known directed graphs,
- direct `state_collapser` adapter,
- node and edge fiber readouts,
- uniform/fiber-normalized/learned lift operators,
- message containers and an edge-message MLP,
- collapse orchestration,
- tiny train-step helper,
- toy repeated-motif example,
- validation proving gradients and optimizer updates.

## Major Movement 3: The First Package Implementation Landed

The implementation commit was:

```text
0071637 Implement first HGraphML import surface
```

It created the real package:

```text
src/hgraphml/
```

and the test suite:

```text
tests/
```

Core source areas:

```text
src/hgraphml/adapters/
src/hgraphml/diagnostics/
src/hgraphml/examples/
src/hgraphml/graph/
src/hgraphml/lifts/
src/hgraphml/messages/
src/hgraphml/training/
```

Core implemented surfaces:

- `TensorGraph`,
- `NodeFiber`,
- `EdgeFiber`,
- `TowerBundle`,
- `build_tower_bundle(...)`,
- `MessageBatch`,
- `EdgeMessageMLP`,
- `run_edge_message_model(...)`,
- `mean_pool_node_features(...)`,
- `incoming_sum_readout(...)`,
- `UniformPullbackLift`,
- `FiberNormalizedLift`,
- `LearnedFiberLift`,
- `collapse_messages(...)`,
- `HGraphMLResult`,
- `GradientDiagnostics`,
- `ViabilityDiagnostics`,
- `make_teacher_node_targets(...)`,
- `train_step(...)`,
- `TrainStepResult`,
- `make_repeated_motif_graph(...)`,
- `run_learned_lift_demo(...)`.

The top-level public-feeling import became intentionally small:

```python
from hgraphml import HGraphMLResult, __version__, collapse_messages
```

Everything else lives in explicit subpackages because the package is still
pre-alpha.

## Major Movement 4: The Direct state_collapser Adapter Worked

The implementation log is:

- `docs/design/01_004_hgraphml_first_import_implementation_log.md`

The adapter lives at:

```text
src/hgraphml/adapters/state_collapser.py
```

It uses upstream `state_collapser` surfaces:

```python
from state_collapser.core.action import PrimitiveAction
from state_collapser.core.edges import BaseEdge
from state_collapser.core.state import State
from state_collapser.tower.partition import ContractionSchema, PartitionTower
from state_collapser.tower.partition.tower import build_partition_tower_full
```

The graph is treated as fully explored:

- every HGraphML node is mapped to a `State`,
- every HGraphML edge is mapped to a `PrimitiveAction` and `BaseEdge`,
- edge labels are forwarded into action/edge labels,
- the supplied contraction schema is passed to `build_partition_tower_full`,
- no RL environment loop or incremental exploration runtime is used.

The final edge-fiber readout rule was corrected during implementation:

```text
At tier i, group original fine edges by the active tier-i state cells of their
source and target endpoints.
```

This matters because internal action-cell labels inside `state_collapser` are
not the right public HGraphML coarse-edge identifiers. HGraphML needs graph ML
coarse edges, which are induced by coarse source and target state cells.

## Major Movement 5: The Trainability Path Was Proved

The first implementation is not just a structural readout. It includes a
trainable PyTorch path:

```text
TensorGraph
    -> tower bundle
        -> coarse graph features
            -> EdgeMessageMLP
                -> coarse messages
                    -> LearnedFiberLift
                        -> fine messages
                            -> incoming_sum_readout
                                -> loss
                                    -> backward
                                        -> optimizer step
```

The demo:

```bash
uv run --extra dev python -m hgraphml.examples.learned_lift_demo
```

proves:

- the package can construct a graph,
- call `state_collapser`,
- run message passing,
- lift messages,
- compute a fine readout,
- backpropagate through trainable parameters,
- and update those parameters.

The package still does not prove speed-up. It proves the computation exists and
is differentiable.

## Major Movement 6: Documentation Was Completed After A Mid-Stream Friction

During implementation, the user became frustrated that planned README/usage/API
documentation appeared to be omitted or treated as optional. That correction
resulted in a fuller documentation pass rather than a code-only first import.

The repository now includes:

- `README.md`,
- `CONTRIBUTING.md`,
- `CHANGELOG.md`,
- `docs/api_notes/01_001_first_surfaces.md`,
- `docs/usage/01_001_first_hack.md`,
- implementation/design docs,
- public-release audit docs,
- and this continuity report.

The README now explains:

- Malik's insight,
- the relation to `state_collapser`,
- what the current milestone proves,
- what it does not prove,
- the central `collapse_messages(...)` call,
- a tiny demo,
- a real minimal use example,
- a train-step example,
- the current implemented surface,
- and the honest repository status.

The CONTRIBUTING guide now explains:

- which changes belong in `HGraphML`,
- which changes belong upstream in `state_collapser`,
- current critical TODOs,
- why serious benchmarking is the real next milestone,
- and why speed-up claims are currently unsafe.

## Major Movement 7: Public Release Readiness Was Audited

The primary public-release audit is:

- `docs/design/public_release_audit/01_001_lightweight_research_public_release_readiness_audit.md`

Its verdict:

```text
HGraphML is close to being ready for a lightweight public research release on
GitHub, provided the release is framed honestly as first trainable
quotient-tower graph-message-passing bridge, not as benchmark-proven graph ML
acceleration framework.
```

Validation recorded during the audit:

```text
uv run --extra dev pytest -> 36 passed
uv run --extra dev ruff check . -> passed
uv run --extra dev mypy -> passed
uv build -> succeeded with temporary cache/output adjustments
```

The audit's main conclusion:

```text
The package is ready for a lightweight research/public-beta posture, but not
for performance claims.
```

The audit identified the main future work:

- serious benchmarking,
- flat-vs-quotient baselines,
- tower-build/coarse-pass/lift/readout/backward cost accounting,
- graph family comparisons,
- schema comparisons,
- repeated seeds,
- negative controls,
- memory measurement where practical,
- artifact output,
- and clearer future dependency policy once `state_collapser` is public or on
  PyPI.

## Major Movement 8: Security And Local-Path Audit Was Performed

The focused security/local-path audit is:

- `docs/design/public_release_audit/01_002_security_and_local_path_public_release_audit.md`

The audit found no credential material, private keys, API tokens, or obvious
secrets.

The main issue was release hygiene:

```text
committed documentation and lock/source metadata should not encode local
machine layout, sibling-checkout assumptions, or absolute user paths.
```

Actions taken:

- removed host-specific absolute paths from public-release audit text,
- rewrote implementation-log local source notes into non-layout-specific
  provenance language,
- removed stale `uv.lock` because it encoded an editable dependency outside the
  repository root,
- changed package metadata away from local editable source assumptions,
- documented that `state_collapser` must be publicly resolvable for CI/release
  users.

The audit's release rule:

```text
No committed file should require, reveal, or imply a local directory layout
outside [HGraphML repository root].
```

## Major Movement 9: Release Metadata, CI, And Versioning Were Set

The repository now has:

- `pyproject.toml`,
- package version `0.1.0`,
- Python `>=3.11,<3.13`,
- `hatchling` build backend,
- typed package marker,
- CI workflow,
- README badges,
- changelog,
- MIT license,
- public-facing repository URLs.

CI is defined at:

```text
.github/workflows/ci.yml
```

and runs:

- dependency sync,
- tests,
- Ruff,
- mypy,
- package build,
- on Python `3.11` and `3.12`.

The package tag:

```text
v0.1.0
```

was selected because this is a public beta/research milestone, not a mature
framework release.

## Major Movement 10: README Banner Assets Were Added

The final visible commit in this interval is:

```text
7084c64 package README banner image
```

It added:

- `assets/images/quotient_graph_dark.jpg`,
- `assets/images/quotient_graph_light.jpg`,
- `assets/images/quotients.xml`,
- README image markup.

This gave the repository a more polished public-facing landing page while
keeping the text honest about research status and missing benchmarks.

## Current Repository Shape

Current top-level shape:

```text
.github/
.gitignore
CHANGELOG.md
CONTRIBUTING.md
LICENSE
README.md
assets/
docs/
pyproject.toml
src/
tests/
```

Current package shape:

```text
src/hgraphml/
  adapters/
  diagnostics/
  examples/
  graph/
  lifts/
  messages/
  training/
```

Current test shape:

```text
tests/
  adapters/
  diagnostics/
  examples/
  graph/
  lifts/
  messages/
  training/
```

Current public-feeling top-level API:

```python
from hgraphml import HGraphMLResult, __version__, collapse_messages
```

## Current Dependency Situation

At the moment this report was first written, `pyproject.toml` declared:

```text
state-collapser @ git+https://github.com/TYLERSFOSTER/state_collapser.git@v0.5.0
```

This was reasonable when the HGraphML release-hardening work was performed,
because `state_collapser` was still expected to preserve old tags or release
around `v0.5.0`.

However, after the later `state_collapser` public-history rewrite, old remote
tags `v0.1.0` through `v0.5.0` were deleted from the `state_collapser` GitHub
remote. The clean public upstream tag is now:

```text
state_collapser v0.6.0
```

This meant the HGraphML dependency pin was stale and likely broken for fresh
installs.

Required follow-up:

```text
Update pyproject.toml to point to state_collapser v0.6.0, regenerate dependency
metadata if the project chooses to track it, then rerun HGraphML validation.
```

Release-readiness follow-up: this dependency pin has now been corrected to
`state_collapser` `v0.6.0`. Treat HGraphML's public-beta package surface as
conceptually ready pending final validation against the public upstream tag.

## Current Public Claims

Safe claims:

```text
HGraphML is a lightweight research package showing that state_collapser quotient
towers can scaffold trainable graph message passing in PyTorch.
```

```text
HGraphML demonstrates trainable quotient-tower-backed graph message passing.
```

```text
The current milestone proves integration and differentiability, not speed-up.
```

Unsafe claims:

```text
HGraphML speeds up graph ML.
```

```text
HGraphML is a mature graph ML framework.
```

```text
HGraphML provides benchmark-proven acceleration.
```

## Known Open Issues And Next Actions

### P0: Update state_collapser Dependency To v0.6.0

This was the immediate cross-repo follow-up.

Current:

```text
state_collapser @ v0.5.0
```

Needed:

```text
state_collapser @ v0.6.0
```

Release-readiness follow-up: this pin has been updated. Rerun:

```bash
uv sync --extra dev --group dev
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev mypy
uv build
```

### P1: Serious Benchmarking

This is the main scientific/engineering gap.

Required benchmark work:

- direct flat message-passing baseline,
- quotient-tower path benchmark,
- tower construction and reuse cost accounting,
- coarse message-pass timing,
- lift timing,
- readout timing,
- backward-pass timing,
- full train-step timing,
- graph family generators,
- repeated seeds,
- negative controls,
- schema comparisons,
- memory measurement where practical,
- artifact output with graph/schema/version/git metadata.

### P1: Real Graph ML Example

The first non-toy example should probably be a small belief-propagation or
factor-graph-style problem because that was the motivating target and because it
can become a benchmark case.

### P1: Dependency Strategy Before PyPI

For a public GitHub research release, a public Git tag dependency can be
acceptable.

For a PyPI release, direct Git dependencies are not the right final posture.
The likely route is:

```text
publish state-collapser first
    -> depend on state-collapser from the registry
        -> then publish hgraphml
```

The PO has already decided that `state_collapser` PyPI should wait until
serious benchmarking. HGraphML PyPI should therefore also wait.

### P2: Batch/Device/Dtype Semantics

The current package proves single-graph viability. It still needs explicit
batching, device movement, dtype policy, and tower-bundle reuse conventions.

### P2: Framework Adapter Decision

The next real adapter should be chosen deliberately:

- PyTorch Geometric,
- DGL,
- NetworkX,
- or a small explicit factor-graph surface.

The benchmark plan should influence this choice.

## Handoff Summary For Next Engineer

If you pick up HGraphML after this report:

1. Understand it as the first downstream graph-ML application of
   `state_collapser`, not as a standalone tower implementation.
2. Do not reimplement `state_collapser` quotient logic locally.
3. First fix the `state_collapser` dependency pin to `v0.6.0`.
4. Rerun validation after that dependency update.
5. Preserve the honest release language: trainability proven, speed-up not
   proven.
6. Treat serious benchmarking as the main next milestone.
7. Use the existing README/CONTRIBUTING distinction to decide whether a change
   belongs in HGraphML or upstream in `state_collapser`.

## Closing Assessment

This interval succeeded at the correct first target:

```text
not a mature graph ML framework,
not benchmark-proven acceleration,
but a real trainable quotient-tower-backed graph message-passing package.
```

That is already a meaningful proof of Malik's observation. The next test is
whether the same architecture can produce measured wins, losses, and tradeoffs
under serious benchmark conditions.

## Final Release Closure Addendum

After the main body of this report was written, the HGraphML public beta release
sequence was closed out.

The repository now has a GitHub Release for `v0.1.0` titled
`HGraphML v0.1.0: Trainable Quotient-Tower Graph Message Passing`. This is the
correct version posture for the project: public beta / research-mode release,
not a mature graph-ML framework release and not a benchmark-backed acceleration
claim.

The `v0.1.0` tag identifies the first public research baseline. Public `main`
has subsequently advanced with release-support cleanup, including citation
metadata. This is acceptable: the tag remains the release anchor, while `main`
continues normal post-release hardening.

The dependency posture was corrected so HGraphML consumes public
`state_collapser` release infrastructure rather than relying on a local path.
The important compatibility fact is that HGraphML should continue to exercise
`state_collapser` through its public quotient-tower/readout surfaces, not by
forking or reimplementing upstream tower logic locally.

The release-badge concern was also clarified. The GitHub release badge resolves
against the public `v0.1.0` release; the remaining issue was presentation color,
not release absence. The README badge was normalized so the public page reads
cleanly alongside the `state_collapser` badge row.

A `CITATION.cff` file was added after the release so GitHub can surface a
`Cite this repository` affordance and so readers have a canonical citation path
for the HGraphML beta.

The remaining release gaps are deliberately not hidden:

- no PyPI release yet,
- no serious benchmark suite yet,
- no broad graph-ML framework adapter story yet,
- no speed-up claim beyond the architectural motivation,
- serious benchmarking remains the next major public-readiness milestone.

PO attribution: the PO made the final release-posture decisions here, including
treating HGraphML as a public beta, prioritizing serious benchmarking over PyPI,
requiring explicit downstream/upstream separation from `state_collapser`, and
requiring release-continuity notes after the final public release pass.
