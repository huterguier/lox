# Changelog

All notable changes to this project are documented in this file. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); pre-1.0, so the minor
version carries breaking changes.

## [Unreleased]

### Added
- `ConsoleLogger.close()` stops the live display and restores the terminal, and is registered with
  `atexit` so the cursor is unhidden even if the process exits without calling it.
- Test coverage for `ConsoleLogger`, which previously had none.

### Fixed
- `ConsoleLogger` no longer stops updating forever when two runs log different keys. It used to
  stack all runs into one array to compute `mean ± std`, which requires identical keys across runs;
  the resulting error was swallowed by a blanket `except`, permanently freezing the table. Each key
  is now aggregated over whichever runs logged it, and the row notes when a key is missing from
  some of them.
- `ConsoleLogger` no longer indexes into the leading axis of a logged value. That axis flattens
  scan iterations, `vmap` lanes and separate `lox.log` call sites together, so no element of it
  identifies "the latest" value — the table showed element `0` of that flattening, which for a
  spooled `scan` meant the first step's value was displayed and never updated. Rows now report
  `mean ± std` over all elements, plus the number of values.
- `ConsoleLogger` renders nested log dicts as `outer/inner` rows instead of raising `TypeError`.
- `ConsoleLogger.init` reuses a single `rich` `Live` display across runs rather than starting a new
  one per run without stopping the old. Extra displays were dead on `rich` 15 and raised
  `LiveError` on older versions, making a second `init` a hard failure there.
- `ConsoleLogger.callback` no longer raises `KeyError` for a state it did not create.

## [0.3.1] - 2026-07-28

### Added
- Support for JAX 0.11, which replaced `scan`'s `num_consts`/`num_carry` parameters with the
  `ft_in`/`ft_out` FlatTrees. `spool` now extends `ft_out` alongside the log outputs it appends to
  a scan; without it every spooled `scan` failed on 0.11. JAX 0.11 requires Python 3.12, so the CI
  matrix pairs it with 3.12 only.
- Tests covering `spool` of a `scan` under `grad`, `vmap`, `cond`, nesting, and a non-trivial `ys`
  pytree — combinations that were previously untested on every JAX version.

### Fixed
- `lox.save(..., key=...)` with a batched key wrote the full, unsharded data into *every* per-key
  folder instead of that key's shard: the inner `save_data` took its payload as `v` but iterated
  the enclosing `data`, so its argument was ignored.
- `lox.save`/`lox.load` could silently mix unrelated runs. Folder names came from concatenating
  the key's raw words as text, so distinct keys collided — raw `[1, 23]` and `[12, 3]` both
  produced `123`. Words are now packed into separate 32-bit slots, which is a bijection. Keys made
  with `jax.random.key(n)` still map to `n`, so existing saved data is unaffected.
- `lox.load` on a path that is not a directory now raises `FileNotFoundError` naming the path,
  instead of an opaque error from the first file open.
- `lox.save` creates its target directory instead of failing when it does not exist.
- Paths are built with `os.path.join` rather than string concatenation.
- `ConsoleLogger` no longer swallows `KeyboardInterrupt`/`SystemExit` while rendering: the guard
  around log stacking caught bare exceptions, so Ctrl-C during a live display could be discarded.

### Changed
- `argnames` in `lox.load` accepts a bare string as a single name, matching the handling that
  `spool`/`tap`/`strip`/`keep` gained in 0.3.0.
- `MultiLogger.callback` raises instead of silently dropping loggers when it is handed a state
  whose logger count does not match its own.
- Tooling: `ruff` replaces `black` and `isort` for linting and formatting, and CI now enforces
  `ruff check`/`ruff format` on every branch.

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
