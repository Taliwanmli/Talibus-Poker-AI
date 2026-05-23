# Portfolio Notes

This document explains how Talibus should be understood as a portfolio and
research project.

## What This Project Demonstrates

Talibus demonstrates system-building around an imperfect-information game. The
project connects a Rust simulation/runtime stack, a Deep-CFR-style training
pipeline, PyTorch model training, ONNX deployment, scripted opponent
evaluation, and depth-limited search experiments.

The most important portfolio signal is not a claim of poker strength. It is
the ability to design, integrate, test, document, and evaluate a multi-language
AI systems prototype with clear limits.

## AI-Assisted Development Workflow

This project was developed through an AI-assisted engineering workflow. The
author directed the research framing, system architecture, evaluation design,
documentation, and iterative debugging while using coding agents to accelerate
implementation.

When discussing the project, avoid claiming that every line was manually
written. Emphasize system ownership, design decisions, verification,
evaluation, debugging, and responsible documentation. The strongest framing is
that AI tools were used as engineering accelerators under human direction and
review.

## Technical Areas Covered

- Rust simulation and runtime engineering.
- Poker game-state modeling, legal actions, pots, betting, and showdown.
- Imperfect-information state representation.
- Fixed action abstraction for no-limit betting.
- Deep-CFR-style traversal and sample generation.
- Python/PyTorch training utilities.
- ONNX export and Rust-side inference.
- Scripted baseline opponent design.
- Evaluation harnesses, result packs, and reproducibility notes.
- Public documentation, limitations, and responsible-use framing.

## What This Project Does Not Claim

Talibus does not claim to solve poker. It does not claim superhuman play,
real-money profitability, human-level strength, solver-level strength, or
proven multiplayer Deep CFR convergence.

Talibus is not a production poker bot, live-play assistant, overlay, casino
automation system, or tool for bypassing poker-site rules.

## How To Discuss This Project In Interviews

Lead with the system design: Rust engine, imperfect-information wrapper,
training sample generation, PyTorch model training, ONNX deployment, runtime
inference, search experiments, and controlled evaluation.

Discuss what was hard: state representation, legal-action handling, abstraction,
long-run orchestration, generated artifact management, evaluation variance, and
claim framing.

Use result numbers carefully. Describe them as controlled simulator measurements
against scripted baselines, useful for regression inside this codebase. Do not
present them as proof of real-world poker strength.

Be direct about the AI-assisted workflow. The relevant ownership story is
setting the goals, making architecture and evaluation decisions, reviewing and
debugging implementation, and documenting the project responsibly.
