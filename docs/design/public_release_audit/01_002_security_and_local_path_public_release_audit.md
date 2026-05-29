# Security And Local-Path Public Release Audit

## Status

Focused security audit for lightweight public release readiness.

Superseded current-status note, 2026-05-29:

- The lightweight GitHub research release happened at `v0.1.0`.
- The dependency-visibility blocker described below has been resolved for the
  current GitHub-research-release posture: HGraphML now pins public
  `state_collapser` `v0.7.0`, and GitHub CI has passed against that dependency.
- `uv.lock` has been reintroduced and now records the public `v0.7.0` Git tag,
  not a local editable path.
- The PyPI guidance below still applies: a future PyPI release should prefer a
  normal registry dependency after `state-collapser` is published there.
- The local-path rule remains current: committed files should not require,
  reveal, or imply a local directory layout outside the HGraphML repository
  root.

Audit date:

```text
2026-05-25
```

Repository audited:

```text
[HGraphML repository root]
```

This audit was created in direct response to the PO's public-release concern:
the repository should not expose the structure of the PO's local machine outside
the `HGraphML` repository root.

## Executive Verdict

No credential material, private keys, API tokens, or obvious secrets were found
in the repository scan.

The main security/public-release issue was not a secret. It was release hygiene:
committed documentation and lock/source metadata should not encode local machine
layout, sibling-checkout assumptions, or absolute user paths. Those details make
a public project look local-only and can accidentally leak workstation
structure.

The concrete cleanup performed in this pass was:

- removed host-specific absolute paths from the public-release audit,
- replaced the historical implementation-log reference to a sibling
  `state_collapser` path with non-layout-specific provenance language,
- removed the stale `uv.lock` file because it encoded an editable dependency
  outside the repository root,
- changed package metadata away from a local editable source assumption,
- documented that `state_collapser` must be publicly resolvable for CI and
  release users.

The remaining release blocker is dependency visibility:

```text
HGraphML currently depends on a tagged upstream state_collapser release.
That upstream dependency must be publicly fetchable, or state-collapser must be
published to a package registry, before HGraphML is truly stranger-installable.
```

## Security Scan Scope

The audit searched for:

- absolute local user paths,
- temporary-build paths,
- references to parent-directory checkouts,
- private key markers,
- token/API-key/password language,
- local dependency source assumptions,
- stale lockfile entries pointing outside the repo root.

Representative scan command:

```bash
rg -n '[absolute user-home path pattern]|[machine temp path pattern]|[parent-directory dependency pattern]|BEGIN RSA|BEGIN OPENSSH|api[_-]?key|secret|password|token' .
```

The words `token` and `secret` appear in general process/theory prose in the
prime-directive material, not as credential values.

## Findings

### P0: Public dependency must be resolvable without local machine context

`HGraphML` imports real upstream `state_collapser` code. That is correct. The
security/release problem is only how that dependency is resolved.

For a lightweight GitHub release, a public tagged Git dependency is acceptable
if the upstream repository and tag are visible to fresh users and CI.

For a PyPI release, the stronger answer is:

```text
publish state-collapser first, then depend on state-collapser>=0.5.0
```

Attempting to refresh the lockfile against the tagged Git dependency from this
machine failed because GitHub requested credentials. That means at least one of
the following is true:

- the upstream repository/tag is not publicly fetchable,
- local network/auth configuration prevents unauthenticated fetch,
- the dependency URL must be changed before public release.

This is a release blocker until verified on GitHub CI or resolved by publishing
`state-collapser`.

### P0: Remove committed lock metadata that points outside the repo root

The previous `uv.lock` encoded a local editable dependency outside the repo
root. That is not appropriate for public release because it preserves a local
development topology.

Action taken:

```text
removed uv.lock
```

This is acceptable for a pre-alpha library repository. Once dependencies are
publicly resolvable, a new lockfile may be generated if the project chooses to
commit locks for application-style reproducibility. If a lockfile is
reintroduced, it must not contain parent-directory editable paths or absolute
machine paths.

### P1: Local path references in release documents

The first public-release audit included literal host paths and temporary build
paths because it recorded commands exactly as run during the audit. That is
useful privately but not ideal for public release.

Action taken:

```text
rewrote host-specific paths as [HGraphML repository root], [temporary cache directory],
and [temporary build directory]
```

This keeps the engineering meaning without exposing workstation layout.

### P1: Historical implementation-log local source note

The first-import implementation log recorded the initial local-development
source as a sibling `state_collapser` checkout.

Action taken:

```text
rewrote the note as provenance without preserving the path
```

This keeps the history: an unpublished upstream checkout was used during
development. It removes the release-facing implication that contributors need
the same local directory layout.

### P1: CI dependency path

CI has now been added. It intentionally uses the committed dependency metadata
rather than a sibling checkout.

That is the right public-release direction, but it means CI becomes the real
test of dependency visibility. If CI fails at dependency installation, the fix
should be upstream dependency publication or public Git/tag visibility, not a
return to local paths.

### P2: Prime-directive and continuity docs in sdist

The PO explicitly wants prime-directive and continuity material included. This
audit does not treat that as a security problem by itself.

The relevant security rule is narrower:

```text
include process/provenance docs, but do not include secrets, credentials, or
machine-specific local layout.
```

No credential material was found in those docs during this scan.

## Current Security Posture

### Safe For Lightweight Research GitHub Release After Verification

- No obvious secrets found.
- License is present.
- Docs may remain in source distribution.
- CI exists.
- Release language remains viability-first.
- Local absolute paths in the audited release docs were removed.
- Stale local-layout lockfile was removed.

### Must Verify Before Public Announcement

- GitHub CI can fetch or install `state_collapser`.
- The `state_collapser` `v0.5.0` dependency target is public, or the dependency
  is changed to a registry package.
- No newly generated `uv.lock` or build artifact reintroduces local editable
  paths.
- README install commands work from a clean checkout in the same mode a public
  user will use.

### Must Fix Before PyPI Release

- Publish `state-collapser` or otherwise make it a normal registry dependency.
- Regenerate lock/build metadata only after dependency resolution is public.
- Confirm package metadata does not contain direct private Git references.
- Run a fresh install smoke test from the built wheel in a clean environment.

## Recommended Release Rule

For this repository, the release security rule should be:

```text
No committed file should require, reveal, or imply a local directory layout
outside [HGraphML repository root].
```

Acceptable examples:

```text
[HGraphML repository root]
[temporary build directory]
state-collapser dependency declared in pyproject.toml
```

Unacceptable examples:

```text
absolute user-home paths
parent-directory editable dependency paths
machine-specific cache paths
private token-bearing URLs
```

## Final Assessment

The repository does not appear to have a secret-leak problem. It had a
public-release locality problem.

That problem is now mostly cleaned up in docs and metadata. The remaining
release-critical question is whether `state_collapser` is publicly resolvable.
If it is not, HGraphML should not be announced as stranger-installable yet. The
right fix is upstream dependency publication or public tagged-release
visibility, not reintroducing a sibling-checkout assumption.
