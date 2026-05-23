# Pre-Release Checklist

Checklist for a public `v0.1 Research Snapshot`.

This checklist is for maintainers preparing a GitHub release. Unchecked items
indicate manual review tasks before publishing a tagged release, not missing
core project documentation.

## Documentation

- [ ] README polished for first-time visitors.
- [ ] Architecture diagram visible on GitHub.
- [ ] Setup smoke checks verified or caveats documented.
- [ ] Result pack documented.
- [ ] Limitations linked from README.
- [ ] Responsible-use document linked from README.
- [ ] Portfolio notes linked from README.
- [ ] Roadmap linked from README.

## Repository Hygiene

- [ ] No large private artifacts committed.
- [ ] No secrets committed.
- [ ] No raw logs accidentally committed.
- [ ] No private checkpoints committed.
- [ ] No untracked generated files included by mistake.
- [ ] `.gitignore` still excludes large generated training/evaluation outputs.

## Release Metadata

- [ ] Release notes drafted.
- [ ] GitHub topics added manually.
- [ ] Result-pack commit hash and artifact hashes reviewed.
- [ ] Public limitations reviewed.
- [ ] Responsible-use language reviewed.

## Suggested GitHub Topics

Add these manually in the GitHub repository settings:

`poker-ai`, `deep-cfr`, `counterfactual-regret-minimization`,
`imperfect-information-games`, `game-theory`, `reinforcement-learning`, `rust`,
`pytorch`, `onnx`, `simulation`, `no-limit-holdem`, `research-prototype`.
