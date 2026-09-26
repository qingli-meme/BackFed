# Temporal backdoor injection experiments

This repository snapshot contains the temporal injection experiments developed for the qingli-meme research project. The benchmark framework is based on BackFed at commit `c851ce90373ef2659447ea25296c7b442e5dc5c7`; the original framework and paper are identified in [README.md](README.md). This snapshot has a new, independent Git history for the research work. No original commits are included in this branch.

## Included work

- Temporal oracle, cost, frontier, exact cohort matching, and actual model geometry probes.
- Pre-scaling directional susceptibility and joint spectral conflict probes.
- Compact result tables in [`research_results/`](research_results/); experiment specifications in [`research_tasks/`](research_tasks/).

The final seven-round directional probe matched the immediate excess ASR target in all seven rounds. The subsequent joint spectral conflict probe did not improve prediction beyond benign drift and the earlier directional measure by the required margin; its verdict was **STOP**. See `research_results/joint_spectral_conflict/diagnostics.json` and the stability configurations nearby.

Large model checkpoints, raw training runs, datasets, and virtual environments are intentionally excluded from Git. They can be regenerated with the scripts in `experiments/` and the task specifications under `research_tasks/`.
