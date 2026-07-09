# Changelog

All notable changes to this project are documented in this file. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); pre-1.0, so the minor
version carries breaking changes.

## [Unreleased]

## [0.3.0] - 2026-07-09

### Added
- `Logger.spool`/`Logger.tap` (and therefore every concrete logger, including `MultiLogger`) now
  accept `argnames`/`tags` (and `unify` for `spool`), matching the standalone `spool`/`tap`
  functions instead of silently dropping that capability.

### Changed
- **Breaking:** `argnames`/`tags` now treat a bare string as a single name/tag, matching
  `jax.jit`'s handling of `static_argnames`, instead of iterating over it character-by-character.
  Previously `argnames="carry"` could silently also match unrelated keys containing `"carry"` as
  a substring, e.g. `"c"`.
- `MultiLogger` no longer keeps its own separate copies of `tap`/`log` — it only overrides
  `callback` (fanning out to each sub-logger) and inherits everything else from `Logger`, so it
  can no longer drift out of sync with future `Logger` features the way it just did.

### Fixed
- `MultiLogger.tap(..., tags=...)` no longer raises `TypeError` (its own `tap` override predated
  the `tags` parameter added to `Logger.tap`).

## [0.2.0] - 2026-07-09

### Added
- `lox.keep`: the whitelist complement of `lox.strip` — removes every logged value except the
  ones matching `argnames`/`tags`; with no filter it is a no-op.
- `spool(..., unify=True)`: unifies divergent `lax.cond` branch log keys instead of raising,
  filling missing keys with NaN/0/False.

### Changed
- **Breaking:** `argnames`/`tags` filtering is now consistent across `tap`, `spool`, `strip`, and
  `keep`: `None` means no restriction on that axis; a list — even an empty one — restricts to
  exactly what's given (previously `argnames=[]`/`tags=[]` could silently behave like `None`).
  When both are given, a log entry must now match both (logical AND); previously the two filters
  could silently shadow or override each other depending on the function.
- **Breaking:** Removed step/episode tracking (`stepdict`) from `logdict` and `lox.log` — logs no
  longer carry associated step metadata.
- Formatting/linting: added `ruff` as a dev dependency.

### Fixed
- `strip(f)` with no `argnames`/`tags` now correctly removes every log (previously a latent bug
  left it a silent no-op); `strip` also now correctly handles functions called with keyword
  arguments.
