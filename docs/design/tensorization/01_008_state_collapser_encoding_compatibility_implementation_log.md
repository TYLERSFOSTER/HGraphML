# State-Collapser Encoding Compatibility Implementation Log

Date: 2026-05-29

Branch: `codex/state-collapser-encoding-compat`

Source plan:

- `docs/design/tensorization/01_007_state_collapser_encoding_compatibility_implementation_gameplan.md`

## Starting State

Starting branch/log:

```text
b9ae0d6 (HEAD -> codex/state-collapser-encoding-compat, origin/main, origin/HEAD, main) tensorization instructions from state_collapser
55165ad Add release closure continuity addendum
63657d3 Add citation metadata
4fbb471 (tag: v0.1.0) release prep
51614a1 release prep
```

Starting status after creating the implementation branch:

```text
## codex/state-collapser-encoding-compat
?? docs/design/tensorization/01_006_state_collapser_encoding_compatibility_blueprint.md
?? docs/design/tensorization/01_007_state_collapser_encoding_compatibility_implementation_gameplan.md
```

## Dependency Update

Updated `pyproject.toml` from:

```text
state-collapser @ git+https://github.com/TYLERSFOSTER/state_collapser.git@v0.6.0
```

to:

```text
state-collapser @ git+https://github.com/TYLERSFOSTER/state_collapser.git@v0.7.0
```

The first plain lock attempt failed because the sandbox could not open the
default user uv cache:

```text
error: failed to open file /Users/foster/.cache/uv/sdists-v9/.git: Operation not permitted
```

The lock was rerun with a workspace-safe cache path:

```bash
UV_CACHE_DIR=/private/tmp/hgraphml-uv-cache uv lock
```

Result:

```text
Resolved 45 packages in 1.96s
Updated state-collapser v0.6.0 -> v0.7.0
```

The resulting `uv.lock` points to:

```text
state_collapser.git?rev=v0.7.0#1b8eb84a43501d79ce44255aa1cc3b1f17d3248e
```

Import verification:

```bash
uv run --extra dev python -c "from state_collapser.training import EncodingRegistry; print(EncodingRegistry.__name__)"
```

Result:

```text
EncodingRegistry
```

## Implementation

Added `build_encoding_registry(...)` in
`src/hgraphml/adapters/state_collapser.py`:

```python
def build_encoding_registry(bundle: TowerBundle) -> EncodingRegistry:
    """Build the shared state_collapser encoding registry for a tower bundle."""

    return EncodingRegistry.from_tower(bundle.partition_tower)
```

Exported the helper from `hgraphml.adapters`.

No `TowerBundle` fields were added or changed.

No `build_tower_bundle(...)` behavior was changed.

## Tests

Added:

- `tests/adapters/test_state_collapser_encoding_registry_compatibility.py`

The new tests cover:

- construction through `build_encoding_registry(...)`;
- parity with direct `EncodingRegistry.from_tower(...)`;
- JSON-safe `registry.to_dict()`;
- base state and edge encodability;
- state-cell and action-cell encodability;
- HGraphML node and edge fiber coverage/range compatibility.

The test imports `EncodingRegistry` from `state_collapser.training` and does
not use RL transition or torch batch surfaces.

## Documentation

Updated:

- `README.md`
- `docs/api_notes/01_001_first_surfaces.md`

Documentation now records:

- dependency pin to public `state_collapser` `v0.7.0`;
- HGraphML compatibility with the shared upstream `EncodingRegistry`;
- that this is shared tower encoding compatibility, not RL tensorization;
- that HGraphML does not require `ActionSelectionInput`,
  `TrainingTransition`, or `TorchDecisionBatch` for this bridge.

## Validation

Focused adapter tests:

```bash
uv run --extra dev pytest tests/adapters/test_state_collapser_adapter.py
```

Result:

```text
3 passed in 1.08s
```

New compatibility tests:

```bash
uv run --extra dev pytest tests/adapters/test_state_collapser_encoding_registry_compatibility.py
```

Result:

```text
5 passed in 0.65s
```

Full test suite:

```bash
uv run --extra dev pytest
```

Result after final rerun:

```text
41 passed in 1.69s
```

Ruff:

```bash
uv run --extra dev ruff check .
```

Result:

```text
All checks passed!
```

Mypy:

```bash
uv run --extra dev mypy
```

Result:

```text
Success: no issues found in 27 source files
```

## Audits

Dependency audit:

```bash
rg -n "state_collapser.git@v0.6.0|rev=v0.6.0|state-collapser.*v0.6.0" pyproject.toml uv.lock README.md
```

Result: no matches.

Code forbidden-surface audit:

```bash
rg -n "ActionSelectionInput|TrainingTransition|LinearizedActionSelectionInput|LinearizedTrainingTransition|TorchDecisionBatch|TorchTransitionBatch|ActionDecision" src tests
```

Result: no matches.

Speed-claim audit checked the touched public docs and tensorization plan docs.
The only matches were existing or planned non-claim language such as "does not
yet claim speed-up", "not speed", and "Do not add speed-up language."

## Final Status Snapshot

Status before this log file was added:

```text
## codex/state-collapser-encoding-compat
 M README.md
 M docs/api_notes/01_001_first_surfaces.md
 M pyproject.toml
 M src/hgraphml/adapters/__init__.py
 M src/hgraphml/adapters/state_collapser.py
 M uv.lock
?? docs/design/tensorization/01_006_state_collapser_encoding_compatibility_blueprint.md
?? docs/design/tensorization/01_007_state_collapser_encoding_compatibility_implementation_gameplan.md
?? tests/adapters/test_state_collapser_encoding_registry_compatibility.py
```

Expected status after this log file:

```text
## codex/state-collapser-encoding-compat
 M README.md
 M docs/api_notes/01_001_first_surfaces.md
 M pyproject.toml
 M src/hgraphml/adapters/__init__.py
 M src/hgraphml/adapters/state_collapser.py
 M uv.lock
?? docs/design/tensorization/01_006_state_collapser_encoding_compatibility_blueprint.md
?? docs/design/tensorization/01_007_state_collapser_encoding_compatibility_implementation_gameplan.md
?? docs/design/tensorization/01_008_state_collapser_encoding_compatibility_implementation_log.md
?? tests/adapters/test_state_collapser_encoding_registry_compatibility.py
```
