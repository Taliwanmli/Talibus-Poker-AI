# Roadmap

This roadmap is intentionally conservative. Talibus is a research prototype,
not a production poker bot or live-play assistant.

## v0.1 Research Snapshot

- Current Rust/Python architecture documented.
- Public result pack documented and linked.
- Setup and smoke-check commands available.
- Limitations and responsible-use boundaries published.
- No large private artifacts, raw logs, or model checkpoints committed.

## v0.2 Reproducibility Improvements

- Add a smaller demo configuration for local smoke runs.
- Clarify minimal commands for traversal, training, export, and evaluation.
- Add a result-pack validation script for expected files, hashes, and metadata.
- Document expected runtime and disk usage for smoke workflows.

## v0.3 Evaluation Improvements

- Add confidence intervals where result files provide enough data.
- Expand scripted baseline policy coverage.
- Explore duplicate-match or other variance-reduction experiments where
  appropriate.
- Separate regression metrics from showcase metrics more clearly.

## v0.4 Research Extensions

- Run better abstraction experiments.
- Compare search budgets under consistent evaluation settings.
- Improve model and checkpoint tracking metadata.
- Add clearer experiment manifests for long runs.
- Investigate additional imperfect-information game abstractions without
  overclaiming general strength.
