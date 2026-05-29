# Lightweight Research Public Release Readiness Audit

## Status

Hardcore repository audit for `HGraphML` public-release readiness.

Superseded current-status note, 2026-05-29:

- The lightweight GitHub research release happened at `v0.1.0`.
- The live dependency pin has moved beyond this audit's `v0.5.0` references;
  HGraphML now targets the public `state_collapser` `v0.7.0` tag.
- A committed `uv.lock` is present again and points at the public `v0.7.0`
  dependency, not a local editable checkout.
- GitHub CI has passed on `main` after the `v0.7.0` compatibility update.
- The current post-release bridge includes `EncodingRegistry` compatibility for
  future tensorization work. The benchmark and speed-up cautions in this audit
  still apply.

Audit date:

```text
2026-05-25
```

Repository audited:

```text
[HGraphML repository root]
```

Current branch during audit:

```text
main
```

Current package version:

```text
hgraphml 0.1.0
```

## Executive Verdict

`HGraphML` is close to being ready for a lightweight public research release on
GitHub, provided the release is framed honestly as:

```text
first trainable quotient-tower graph-message-passing bridge
```

and not as:

```text
benchmark-proven graph ML acceleration framework
```

The repo has a real package, real tests, typed source, a direct
`state_collapser` adapter, deterministic and learned lifts, working training
viability, a README, CONTRIBUTING guide, usage docs, API notes, and a buildable
wheel/sdist.

However, the repo is not yet ready for a confidence-heavy public release that
claims performance or broad usability. The main missing piece is serious
benchmarking. The current package proves that the idea is executable and
trainable. It does not yet prove that it is faster, cheaper, more scalable, or
better than direct flat message passing.

### Release Readiness By Release Type

| Release target | Current readiness | Verdict |
|---|---:|---|
| Private/local development milestone | 95% | Ready |
| Lightweight public GitHub research release | 80% | Nearly ready after small release hardening |
| PyPI research package release | 60% | Possible, but dependency/release metadata should be fixed first |
| Public benchmark-supported speed-up release | 25% | Not ready; serious benchmarking is the core missing work |
| Mature graph ML framework release | 15% | Explicitly out of scope right now |

## Bottom-Line Recommendation

Cut a lightweight public research release only if the release language is
careful:

```text
HGraphML is a small research package showing that state_collapser quotient
towers can scaffold trainable graph message passing in PyTorch.
```

Do not claim:

```text
HGraphML speeds up graph ML.
```

The strongest immediate next move is not another toy feature. It is a serious
benchmarking track:

```text
controlled graph families
    + flat-vs-quotient baselines
    + tower-build / coarse-pass / lift / readout / backward cost accounting
    + artifact output
    + schema comparisons
```

Once that exists, HGraphML can move from "clever executable research bridge" to
"credible research package with evidence."

## Audit Method

The audit inspected:

- repository status and tracked files,
- root release-facing docs,
- package metadata,
- package source layout,
- direct `state_collapser` adapter,
- graph/message/lift/training surfaces,
- tests,
- validation commands,
- build artifacts,
- ignored/generated files,
- release infrastructure presence or absence,
- roadmap/TODO alignment.

Commands run during audit:

```bash
git status --short
git branch --show-current
git ls-files | sort
find . -maxdepth 3 -type f | sort
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev mypy
UV_CACHE_DIR=[temporary cache directory] uv build --out-dir [temporary build directory]
```

Build note: the first `uv build` attempt failed because the local sandbox could
not access the default user-level `uv` cache. A second attempt with a temporary
cache and explicit output directory succeeded after network/cache access was
allowed.

## Validation Results

### Tests

```text
uv run --extra dev pytest
36 passed
```

Coverage areas represented in the test suite:

- package import,
- tensor graph validation,
- fiber validation,
- `state_collapser` adapter,
- deterministic lifts,
- learned lift,
- message passing,
- readout,
- collapse orchestration,
- diagnostics,
- toy training objective,
- train step,
- learned-lift demo.

### Lint

```text
uv run --extra dev ruff check .
All checks passed
```

### Typing

```text
uv run --extra dev mypy
Success: no issues found in 27 source files
```

### Build

Successful build artifacts:

```text
[temporary build directory]/hgraphml-0.1.0.tar.gz
[temporary build directory]/hgraphml-0.1.0-py3-none-any.whl
```

Wheel contents are clean and package-focused:

```text
hgraphml/*
hgraphml-0.1.0.dist-info/METADATA
hgraphml-0.1.0.dist-info/WHEEL
hgraphml-0.1.0.dist-info/licenses/LICENSE
hgraphml-0.1.0.dist-info/RECORD
```

The wheel does not include tests, design docs, prime-directive docs, or local
cache files. This is good.

The sdist is broader. It includes:

- package source,
- tests,
- README,
- CONTRIBUTING,
- LICENSE,
- docs,
- prime-directive docs,
- engineer continuity docs.

For a research source distribution this is acceptable, though a polished package
release might choose to exclude prime-directive and engineer-continuity material
from the sdist.

## Repository Shape

Current tracked top-level structure:

```text
.gitignore
CONTRIBUTING.md
LICENSE
README.md
docs/
pyproject.toml
src/hgraphml/
tests/
```

Current source layout:

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

Current test layout mirrors the package reasonably well:

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

The package is small and inspectable. This is good for a lightweight research
release. It avoids the common pre-alpha failure mode of pretending to be a full
framework before the central idea is proven.

## Release-Facing Documentation Audit

### README

The README now does the right high-level job:

- explains what HGraphML is,
- identifies `state_collapser` as the quotient-tower provider,
- gives quick demo commands,
- gives a minimal real `collapse_messages(...)` use case,
- gives a train-step example,
- lists implemented surfaces,
- states that speed-up is not yet claimed,
- points to usage, API, and design docs.

This is directionally release-ready.

#### README Strengths

- Clear conceptual framing.
- Honest pre-alpha status.
- Explicit no-speed-up-claim language.
- Engineer can see how to run something real quickly.
- The code snippets reflect actual implemented imports.
- The docs map is useful.

#### README Weaknesses

The README still carries public-release risk in four places.

First, badges point to public infrastructure that may not exist yet:

```text
https://github.com/TYLERSFOSTER/HGraphML/actions/workflows/ci.yml
https://pypi.org/project/hgraphml/
https://github.com/TYLERSFOSTER/HGraphML/releases
```

There is no `.github` directory in the repo. If the GitHub Actions workflow does
not exist remotely, the CI badge will look broken. If the PyPI project and
release do not exist yet, the badge row may look aspirational rather than
professional.

Second, installation instructions are development-oriented:

```bash
uv sync --extra dev --group dev
```

That is fine for a research repo, but a public release should also say exactly
what a stranger should do when resolving the `state_collapser` dependency.

Third, the README explains that normal package installation can happen after
`state-collapser` and `hgraphml` are published in the needed form. That is
honest, but it means the repo is not yet fully stranger-installable as a simple
PyPI package.

Fourth, the README's demo is still toy-only. This is acceptable for lightweight
research release, but not for a release whose pitch is graph ML usefulness.

### CONTRIBUTING

The CONTRIBUTING guide is strong and unusually honest.

It correctly distinguishes:

- HGraphML-owned graph ML surfaces,
- upstream `state_collapser` quotient/tower surfaces,
- serious benchmarking as the main next technical gap.

The TODOs now reflect the actual repo state. They do not pretend the next task
is vague framework maturity. They identify benchmarking as the critical release
work.

#### CONTRIBUTING Strengths

- Clear repo-boundary rules between HGraphML and `state_collapser`.
- Clear no-local-fallback-tower principle.
- Good release expectations.
- Good testing expectations.
- Good design-authority links.
- Critical TODOs prioritize benchmarking.

#### CONTRIBUTING Weaknesses

- It may be too internally sophisticated for a first-time external contributor.
- It assumes contributors understand `state_collapser` and quotient towers.
- It has no short "good first contribution" path.

For a lightweight research release this is acceptable. For broader open-source
adoption, it should eventually add a short first-contribution section.

### Usage Docs

`docs/usage/01_001_first_hack.md` is useful and accurate for the current
milestone.

It explains:

- the mental model,
- how HGraphML differs from the RL posture of `state_collapser`,
- toy graph construction,
- tower bundle construction,
- message model and learned lift setup,
- `collapse_messages(...)`,
- `train_step(...)`,
- current non-goals.

This is adequate for a lightweight research release.

### API Notes

`docs/api_notes/01_001_first_surfaces.md` is very useful. It gives exact current
surfaces without overclaiming stability.

It correctly warns:

```text
The surfaces below should be read as milestone-one APIs.
```

That is the right public posture.

### Design Docs

The design docs are strong and provide unusually deep context. They are useful
for serious collaborators.

However, the design docs include conversation traces and internal development
language in places. This is not necessarily a blocker for research release, but
it makes the repo feel like an active lab notebook rather than a polished public
package.

That may be exactly the desired posture for a lightweight research release.

## Package Metadata Audit

Current `pyproject.toml` is functional and now includes basic public-release
metadata.

Strengths:

- `src/` package layout.
- Hatchling build backend.
- Python bounded to `>=3.11,<3.13`.
- Runtime dependencies explicit: `torch`, `state-collapser`.
- Dev extras for mypy/pytest/ruff.
- `py.typed` included in wheel.
- Wheel builds successfully.
- Project URLs, classifiers, and keywords have been added.
- Direct-reference metadata is explicitly enabled for the temporary Git-tag
  `state_collapser` dependency strategy.

Release weaknesses:

1. Dependency uncertainty around `state-collapser`.

   `HGraphML` depends on `state_collapser` surfaces that are still moving. For a
   lightweight GitHub research release, pinning to a public upstream tag is
   acceptable. For a PyPI release, `state-collapser` should be published as a
   registry package and the dependency should be switched back to a normal
   version constraint.

2. Split dev dependency declaration.

   `pytest`, `ruff`, and `mypy` are in `[project.optional-dependencies].dev`,
   while `numpy` is in `[dependency-groups].dev`. This works for current `uv`
   flows, but it is slightly confusing for public contributors. If NumPy is only
   needed to suppress PyTorch warnings, this should be documented or folded into
   one dev story.

3. No console scripts.

   Not mandatory, but a public research package might benefit from a small demo
   script entry point once benchmarks exist.

## Source Code Audit

### Overall Source Health

The source is small, typed, and clear. The package is not overbuilt.

This is a major strength.

Current source count is modest enough that a new engineer can read the package
in one sitting. That is ideal for lightweight research release.

### Public Surface

Top-level exports are intentionally small:

```python
from hgraphml import HGraphMLResult, __version__, collapse_messages
```

This is the right instinct. It avoids prematurely stabilizing subpackage APIs.

### Adapter Layer

The direct `state_collapser` adapter is the package's most important technical
piece.

Strengths:

- It uses real `state_collapser` classes.
- It does not locally reimplement a private tower engine.
- It treats the known graph as fully explored, which is the correct HGraphML
  adaptation.
- It recovers node fibers by active state cells.
- It groups edge fibers by active coarse source/target state-cell endpoints,
  avoiding unstable internal action-cell labels.

Risks:

- It imports internal-ish `state_collapser` surfaces directly.
- It assumes `build_partition_tower_full` and state-layer internals remain
  stable.
- It uses `dict[object, int]` for cell-to-coarse mapping, which is practical but
  not semantically rich.
- It constructs `State` and `BaseEdge` objects per call, which is fine now but
  will matter in benchmarks.
- It has no adapter capability/version check.

For a lightweight research release, this is acceptable. For a stable package,
it needs a formal compatibility contract with `state_collapser`.

### Graph Layer

`TensorGraph` is appropriately minimal.

Strengths:

- Validates edge-index rank and shape.
- Validates integer dtype.
- Validates label count.
- Validates index bounds.

Limitations:

- No explicit device policy.
- No batching.
- No graph-level metadata.
- No graph-family/generator metadata.
- Edge labels are strings only, which is fine for milestone one.

This is ready for research release, but not framework-grade.

### Lift Layer

Lift operators are a good first slice.

Implemented:

- `UniformPullbackLift`,
- `FiberNormalizedLift`,
- `LearnedFiberLift`.

Strengths:

- Clear semantics.
- Tested gradient behavior.
- Learned lift exposes recent within-fiber weights.
- Deterministic lifts serve as baselines.

Risks:

- Exactness semantics are not formalized in code.
- Learned lift is tiny and intentionally not a serious architecture.
- No batching/vectorization over fibers.
- Lift loops are Python-level and will likely dominate for large graphs unless
  optimized.

For the next stage, benchmarking should measure lift overhead explicitly.

### Message Layer

`EdgeMessageMLP`, pooling, and incoming readout are intentionally simple.

Strengths:

- Easy to understand.
- Validates shapes.
- Works with autograd.
- Does not drag in PyG/DGL prematurely.

Limitations:

- Not a full GNN layer.
- No multi-step message-passing abstraction.
- No residual/update functions.
- No node-state update beyond incoming sum readout.

This is good for lightweight release, as long as the README remains clear that
this is not yet a graph ML framework.

### Training Layer

The training layer is a tiny viability helper, not a training framework.

Strengths:

- Proves loss/backprop/optimizer integration.
- Collects useful diagnostics.
- Keeps train loop simple and inspectable.

Limitations:

- No checkpointing.
- No experiment manifest.
- No dataset abstraction.
- No train/eval mode handling.
- No batching.
- No artifact output.

This is acceptable for current release posture.

### Diagnostics Layer

Diagnostics are exactly the right size for milestone one.

Strengths:

- Gradient count.
- Nonzero gradient count.
- Gradient norm.
- Parameter-change check.
- Message shape capture.

Limitations:

- No structured serialization.
- No benchmark records.
- No memory/timing metrics.

The next diagnostics work should be benchmark-artifact-oriented.

## Test Audit

The current test suite is strong for the first vertical slice.

Test count:

```text
36 tests
```

Strengths:

- Good package-area mirroring.
- Adapter is tested.
- Lifts are tested.
- Learned lift gradient behavior is tested.
- `collapse_messages(...)` is tested.
- Train step is tested.
- Demo is tested.
- Invalid inputs are tested in core data structures.

Gaps:

- No benchmark tests.
- No fresh-install test from built wheel.
- No CI matrix test.
- No Python 3.12 validation observed during audit.
- No tests for multiple graph sizes.
- No tests for large graph behavior.
- No tests for bad-collapse controls.
- No tests for exact-vs-approximate lift semantics beyond basic lift behavior.
- No tests for install behavior when `state-collapser` is resolved through the
  same public dependency path used by fresh users and CI.

For lightweight research release, the tests are good. For PyPI confidence, add
CI and a fresh-install smoke test.

## Build And Distribution Audit

The package can build a wheel and sdist when build dependencies are available.

This is an important positive result.

However, public distribution is not yet fully solved.

### Wheel

Wheel status:

```text
good
```

The wheel includes package files and license only. That is appropriate.

### Sdist

Sdist status:

```text
acceptable for research release, noisy for polished package release
```

The sdist includes docs, tests, prime directive docs, and engineer continuity
docs. This may be fine for a research repo where provenance and design context
matter. If the aim is a clean public package distribution, consider excluding
internal process docs from sdist or relocating them.

### Installability Risk

The largest distribution risk is the `state-collapser` dependency.

Current metadata says:

```text
Requires-Dist: state-collapser @ git+https://github.com/TYLERSFOSTER/state_collapser.git@v0.5.0
```

This is acceptable for a lightweight GitHub research release if the upstream
repository and tag are public. For a public PyPI release one of the following
must be true:

1. `state-collapser>=0.5.0` is actually published and compatible.
2. HGraphML documents that public installation currently uses the public
   upstream Git tag.
3. HGraphML waits for `state_collapser` PyPI readiness before PyPI release.

For GitHub research release, option 2 is acceptable. For PyPI, option 1 is the
right answer.

## CI / Automation Audit

An in-repo GitHub Actions workflow has now been added at:

```text
.github/workflows/ci.yml
```

It covers:

- tests,
- lint,
- typing,
- build,
- Python 3.11,
- Python 3.12.

This resolves the README CI badge mismatch, provided the workflow is pushed to
GitHub and the upstream `state_collapser` dependency is publicly resolvable.

### Recommendation

Before public release, verify the workflow passes on GitHub with:

- Python 3.11,
- Python 3.12,
- `uv run --extra dev pytest`,
- `uv run --extra dev ruff check .`,
- `uv run --extra dev mypy`,
- `uv build`.

Because `state_collapser` is not assumed to be a sibling checkout, CI now
depends on the committed public dependency declaration. For PyPI release, switch
that declaration to a normal registry version constraint after `state-collapser`
is published.

## Benchmarking Audit

This is the most important part of the audit.

There is currently no serious benchmarking surface.

Current benchmark status:

```text
viability benchmark only
```

The learned-lift demo proves:

- the package runs,
- loss is finite,
- gradients flow,
- optimizer step changes parameters,
- toy loss can move.

It does not prove:

- speed-up,
- lower memory,
- better scaling,
- better convergence,
- graph-family robustness,
- useful behavior on belief propagation,
- production relevance.

### Benchmarking Is The Real Next HGraphML Work

The next release-shaping work should be a benchmark package or benchmark module.

The benchmark surface should measure at least:

- graph construction time,
- tower construction time,
- coarse graph construction/readout time,
- flat message-passing forward time,
- quotient message-passing forward time,
- lift time,
- fine readout time,
- backward time,
- full train-step time,
- memory where practical,
- graph sizes,
- edge counts,
- quotient tier sizes,
- fiber sizes,
- schema mode,
- lift type.

Benchmark graph families should include:

- repeated motifs where quotienting should help,
- nearly repeated motifs where quotienting may help approximately,
- random sparse graphs where quotienting may not help,
- dense local neighborhoods where tower construction may dominate,
- bad-collapse controls where quotienting should hurt or distort,
- factor-graph-like examples if belief propagation remains a target.

Benchmark outputs should be structured:

```text
artifacts/benchmarks/<run-id>/manifest.json
artifacts/benchmarks/<run-id>/results.jsonl
artifacts/benchmarks/<run-id>/summary.md
```

The benchmark package should not hide negative results. For this project,
knowing when quotient-tower message passing is not worth it is as important as
showing wins.

## Public Release Risk Register

### P0: No serious benchmarking yet

Impact:

```text
Cannot responsibly claim speed-up or scaling advantage.
```

Recommendation:

```text
Make release language viability-only, or build benchmark harness first.
```

### P0: `state-collapser` public dependency may not be installable

Impact:

```text
PyPI users may not be able to install HGraphML cleanly.
```

Recommendation:

```text
Verify state-collapser>=0.5.0 on PyPI or pin to a public upstream release tag.
```

### P1: CI workflow must be observed on GitHub

Impact:

```text
Public repo can look broken immediately if the badge points at a failing workflow.
```

Recommendation:

```text
Push .github/workflows/ci.yml and verify the workflow passes on GitHub.
```

### P1: Direct Git dependency is not a final PyPI dependency story

Impact:

```text
Public install depends on whether the upstream Git tag is fetchable.
```

Recommendation:

```text
Use the Git tag for the lightweight GitHub release; publish state-collapser and
switch to a registry dependency before PyPI.
```

### P1: README install path is still local-development-first

Impact:

```text
Strangers may not know whether pip install is expected to work.
```

Recommendation:

```text
Explain current Git-tag dependency mode and future PyPI dependency mode.
```

### P1: Changelog must be kept current

Impact:

```text
Release history becomes unclear if CHANGELOG.md is not updated before tagging.
```

Recommendation:

```text
CHANGELOG.md has been added; keep the 0.1.0 section aligned with final release contents.
```

### P2: Sdist intentionally contains internal process docs

Impact:

```text
Research provenance is rich, and the PO wants it included.
```

Recommendation:

```text
Keep process/provenance docs, but audit them for secrets and local machine layout.
```

### P2: No security/code-of-conduct/citation files

Impact:

```text
Not a blocker for lightweight research release, but public repo completeness is lower.
```

Recommendation:

```text
Add SECURITY.md only if issue-reporting policy matters now; add CITATION.cff if academic sharing matters.
```

### P2: Demo module warning has been cleaned

Observed earlier during local runs:

```text
RuntimeWarning: 'hgraphml.examples.learned_lift_demo' found in sys.modules after import of package 'hgraphml.examples'
```

Impact:

```text
Demo succeeded, but public demo output looked less clean.
```

Likely cause:

```text
hgraphml.examples.__init__ imports learned_lift_demo, then python -m executes it.
```

Recommendation:

```text
Resolved by lazy-importing runnable demo helpers from examples/__init__.py.
```

## Readiness Checklist

### Already Good Enough For Lightweight Research Release

- Package imports.
- Tests pass locally.
- Ruff passes locally.
- Mypy passes locally.
- Wheel builds.
- README exists and explains current state.
- CONTRIBUTING exists and has realistic TODOs.
- Usage docs exist.
- API notes exist.
- Implementation log exists.
- License exists.
- Direct `state_collapser` adapter exists.
- No local fake tower implementation exists.
- No speed-up claim is made in the core release docs.

### Should Fix Before Lightweight Public GitHub Release

- Add CI or remove/hide CI badge.
  - **PO Reply:** Add CI.
  - **Codex Reply:**
  Added `.github/workflows/ci.yml`. The workflow runs on pushes and pull
  requests to `main`, with a Python `3.11` / `3.12` matrix, and executes tests,
  Ruff, mypy, and package build. This is the correct release-facing move because
  the README already exposes a CI badge. The remaining practical check is that
  the workflow must be pushed and observed on GitHub, where it will also verify
  whether the public `state_collapser` dependency is resolvable for strangers.
- Add `CHANGELOG.md` with `0.1.0` entry.
  - **PO Reply:** Explain to me why `v0.1.0` and not `v1.0.0`.
  - **Codex Reply:**
  `v0.1.0` is the right number because this is a first public research
  viability release, not a stable public API release. The package has a real
  vertical slice, but it does not yet have benchmark evidence, mature graph-ML
  adapters, stable long-term API guarantees, or a production install story
  independent of upstream `state_collapser` release posture. Calling this
  `v1.0.0` would communicate API stability and broad readiness that the audit
  explicitly says we should not claim yet. I added `CHANGELOG.md` with a
  `0.1.0` section that frames this as a research viability release.
- Clarify install story for strangers resolving `state_collapser`.
  - **PO Reply:** This doesn't make sense to me; `state_collapser` is a dependency.
  - **Codex Reply:**
  Correct. My earlier phrasing was sloppy: the issue is not that
  `state_collapser` is some hidden sibling checkout requirement. It is a real
  dependency. The public-release question is how that dependency resolves for a
  fresh user or CI runner. I changed the package metadata away from a committed
  local path assumption and pinned the dependency to the public
  `state_collapser` `v0.5.0` Git tag for this lightweight GitHub research
  release. For PyPI, the better final state is to publish `state-collapser` and
  switch this back to a normal registry constraint such as
  `state-collapser>=0.5.0`.
- Verify README demo output is clean or document known warning.
  - **PO Reply:** Do this and report back.
  - **Codex Reply:**
  Done. The warning came from importing the runnable demo module through
  `hgraphml.examples.__init__` and then executing that same module with
  `python -m`. I changed `src/hgraphml/examples/__init__.py` so runnable demo
  helpers are exposed lazily. That preserves convenient imports while avoiding
  the prior `runpy` warning path. The validation rerun is recorded below in the
  final response for this work.
- Decide whether public release should include prime-directive and continuity docs in sdist.
  - **PO Reply:** They should. However, please do one more public release uaudit focussed on security. One thing I know I need fixed is prbably references to the layout of my local machine anywhere. There should not be any reference to the structure of local machien outside of `HGraphML` repo root.
  - **Codex Reply:**
  Agreed. I am treating design, prime-directive, and continuity material as part
  of the research-source provenance rather than something to hide from the
  source distribution. I also added a separate security/local-path public
  release audit in this same folder and cleaned this audit so it no longer
  exposes host-specific absolute paths. The one important dependency-related
  follow-up is that committed lock/source metadata must not encode a local
  machine layout outside the repo root.

### Should Fix Before PyPI Release

- Verify `state-collapser>=0.5.0` is published and compatible.
- Switch the temporary Git-tag dependency back to a normal registry dependency.
- Add package install smoke test from built wheel.
- Consider whether `numpy` belongs in optional dev extras, dependency group, or
  not at all.
- Confirm `uv.lock` policy for package distribution.

### Required Before Any Speed-Up / Scaling Claim

- Serious benchmark harness.
- Flat message-passing baseline.
- Quotient message-passing benchmark path.
- Controlled graph families.
- Artifact output.
- Repeated runs / seed handling.
- Negative controls.
- Timing decomposition.
- Memory measurement where feasible.
- Public benchmark documentation.
  - **PO Reply:** Put an extremely clear discussion of all the above "Required Before Any Speed-Up / Scaling Claim" in `CONTRIBUTING.md`.
  - **Codex Reply:**
  Done. `CONTRIBUTING.md` now has a dedicated "Required Before Any Speed-Up Or
  Scaling Claim" section. It explicitly requires flat baselines, quotient-path
  timing, controlled graph families, schema comparisons, repeated seeds,
  negative controls, memory measurements, artifact output, and public benchmark
  documentation before README, release notes, papers, or outreach use speed-up
  language.

## Proposed Public Release Wording

Safe wording:

```text
HGraphML is a pre-alpha research package showing that state_collapser quotient
towers can be used as scaffolding for trainable graph message passing in
PyTorch. The current release is a viability release, not a speed-up claim.
```

Unsafe wording:

```text
HGraphML accelerates graph ML.
```

Better future wording after benchmarks:

```text
On controlled repeated-motif graph families, HGraphML's quotient-tower path
reduces measured message-passing cost relative to a flat baseline under the
following schemas and lift assumptions...
```

  - **PO Reply:** Yes I completely support this.
  - **Codex Reply:**
  Good. I preserved this distinction as the governing release language:
  `HGraphML` may claim trainable quotient-tower-backed graph message passing,
  but it should not claim acceleration until the benchmark track exists.

## Recommended Immediate Implementation Plan

If the goal is a lightweight public research release soon, do this in order:


  - **PO Reply:** Please modify the below according to discussion turns above.
  - **Codex Reply:**
  Revised plan below. The main changes are: CI is now an action item already
  implemented, dependency language is corrected to treat `state_collapser` as a
  real dependency rather than a sibling checkout, security/local-path auditing is
  promoted to release hardening, and the release remains `v0.1.0` because this
  is a research viability release rather than a stable framework release.

1. Keep the release framed as `v0.1.0`, a research viability release.
2. Add and verify CI for tests, Ruff, mypy, and build on Python 3.11 and 3.12.
3. Add `CHANGELOG.md` with a clear `0.1.0` viability-release section.
4. Add project URLs, classifiers, and keywords to `pyproject.toml`.
5. Clarify README and CONTRIBUTING install language around `state_collapser` as
   a real package dependency.
6. For the lightweight GitHub release, pin `state_collapser` to a public
   upstream tag or otherwise ensure fresh users and CI can resolve it.
7. Before PyPI release, publish `state-collapser` or otherwise switch HGraphML
   to a normal registry dependency.
8. Remove machine-specific paths and local-layout assumptions from committed
   docs and lock/source metadata.
9. Clean the demo warning.
10. Keep prime-directive, design, and continuity material in the research sdist.
11. Tag `v0.1.0` only after CI, local validation, dependency resolution, and the
   security/local-path audit are acceptable.

If the goal is a stronger release that can attract serious graph ML attention,
do this first:

1. Build benchmark graph-family generators.
2. Build flat baseline path.
3. Build benchmark runner and artifact schema.
4. Run benchmark matrix.
5. Write benchmark report.
6. Then release.

## Final Assessment

HGraphML is in a promising and unusually clean state for a brand-new research
package. The first import of `state_collapser` into graph message passing is
real. The code is small, typed, tested, and buildable. The docs are honest about
what exists.

The project should not wait for maturity before being shown publicly, as long as
the release is framed as a lightweight research viability release.

But the next serious step is unmistakable:

```text
benchmarking, benchmarking, benchmarking
```

That is the line between "interesting executable idea" and "research package
other graph ML engineers should take seriously."
