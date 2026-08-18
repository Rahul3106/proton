# Proton Data Science Agent (proton)

An autonomous Data Science Agent you talk to through a terminal.

```bash
proton start
```

```text
proton > load customer.csv
proton > analyze it
proton > build a model to predict churn
```

## Status

**Phase 1 - Package Foundation.** The package installs and `proton --help`
works. No agent behavior yet - see the project spec for the phase-by-phase
build order (Phase 2 adds `proton start` and an interactive session).

## Development

```bash
uv sync
uv run proton --help
```
