# v0.1 Research Snapshot

## Summary

Initial public research snapshot of Talibus, a 6-max No-Limit Texas Hold'em AI
systems research prototype. The project combines Rust simulation/runtime
infrastructure, a Deep-CFR-style training pipeline, PyTorch models, ONNX
deployment, scripted opponent evaluation, and depth-limited search experiments.

This release is for research, education, and portfolio review only. It is not a
production poker bot, real-money poker tool, live-play assistant, overlay, RTA
tool, casino automation tool, or platform-rule bypass tool.

## Included

- Rust/Python architecture documentation.
- Setup notes and smoke-check commands.
- Controlled simulator evaluation documentation.
- Compact public result-pack explanation.
- Limitations and responsible-use guidance.
- Portfolio notes for discussing the project professionally.
- Roadmap and lightweight contributor guidance.

## Not Included

- Full generated training buffers.
- Raw training or evaluation logs.
- PyTorch checkpoints.
- ONNX model binaries.
- Large local artifacts required for full long-run reproduction.

These files are intentionally excluded because they are generated artifacts and
can be large. Full long-run reproduction requires generated/local artifacts,
configured dependencies, and substantial compute.

## Responsible Use

Talibus is intended for imperfect-information game research, education,
simulation, and software engineering portfolio review. It should not be used for
real-money play, live decision support, platform automation, or bypassing
platform rules.

The project does not claim solved poker, superhuman play, real-world
profitability, human-level strength, solver-level strength, or proven
multiplayer Deep CFR convergence.

## Result Interpretation

Published result numbers are controlled simulator measurements against scripted
baseline opponents. They are useful for regression/evaluation inside this
codebase only and are not evidence of real-money performance, human-level play,
solver-level play, or general poker strength.

## Suggested GitHub Release Text

Initial public research snapshot of Talibus, a 6-max No-Limit Texas Hold'em AI
systems research prototype. Includes Rust/Python architecture documentation,
setup notes, controlled simulator evaluation documentation, a compact public
result-pack explanation, limitations, responsible-use guidance, roadmap, and
contributor notes. This release is for research, education, and portfolio review
only; it is not a production poker bot, real-money poker tool, live-play
assistant, or poker-strength claim.
