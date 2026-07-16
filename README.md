# Hospital Resource Allocation Under Uncertainty Using Reinforcement Learning

**CISC 856 — Reinforcement Learning, Queen's University**
Group 8: Mirna Imbabi · Salma ElSherif · Norhan Mahmoud · Nehal Hamada

A reinforcement learning approach to dynamic hospital resource allocation under stochastic patient demand. We built a custom Gymnasium environment simulating ICU beds, ward beds, ventilators, and staff shifts under Poisson patient arrivals, and benchmarked PPO, DQN, and A2C (Stable-Baselines3) against a rule-based greedy baseline across **86 experimental conditions** (77 RL runs across twelve groups, plus 9 greedy baselines).

## Key Findings

- **DQN is the strongest overall performer**, beating the greedy baseline by +9.5% at λ=3.0, +23.3% at λ=1.5, and +52.8% at λ=4.5, while holding only a narrow +2.8% edge under saturation (λ=6.0). The DQN > PPO > A2C reward ranking holds at every arrival rate tested.
- **A2C's underperformance is structural**, not a tuning artefact: it collapses to a degenerate "always admit to ICU" policy that is invariant across four hyperparameter variants and 8-parallel-environment training.
- **Reward is a poor proxy for mortality outcomes.** At a moderate mortality penalty (δ=2), none of the three RL agents beats the greedy policy's death rate. Raising the penalty to δ=10 roughly halves deaths for both PPO and DQN and pushes both below greedy, while episode reward shifts by only 0.4–2.4%.
- **An overflow penalty backfires**: stacking a severity-weighted overflow penalty on top of the existing queue penalty reduces reward at every condition tested, by pushing agents toward *more* admission and *less* redirection — the opposite of its intended effect.
- **Reward-weight calibration (Group L)** — not structural reward redesign — is the most effective lever for improving absolute reward: lighter critical-queue weights raise both PPO and DQN reward and is the only variant under which DQN consistently beats greedy across all weight grids tested.
- Four environment/pipeline bugs were identified and corrected during development, most critically a missing ward-discharge mechanism that froze the environment after ~30 steps and invalidated all initial results (see [Implementation Issues](#implementation-issues) below).

Full methodology, results tables, and discussion are in [`CISC-856-Group-8-Final-Project-Report.pdf`](./CISC-856-Group-8-Final-Project-Report.pdf).

## Repository Contents

| File | Description |
|---|---|
| `group-8-hospital_allocation-final-project.ipynb` | Complete, self-contained notebook: environment, training, evaluation, and all analysis/figures |
| `CISC-856-Group-8-Final-Project-Report.pdf` | Full written report (IEEE format) |
| `output/models/` | Saved model checkpoints (generated on run; not tracked in repo) |
| `output/figures/` | Generated evaluation plots (generated on run; not tracked in repo) |
| `output/tb_logs/` | TensorBoard training logs (generated on run; not tracked in repo) |

## Environment: `HospitalEnv`

A custom Gymnasium environment modelling ED-to-hospital triage as an MDP.

**State space** (9-dimensional, 10-dimensional in mortality-aware Groups G/L):
`[icu_beds_free, ward_beds_free, ventilators_free, staff_shifts, queue_critical, queue_moderate, queue_stable, occupied_ward, hour_of_day]`

**Action space** (4 discrete actions):

| Action | Description |
|---|---|
| 0 | Admit to ICU (requires ventilator with probability 0.60) |
| 1 | Admit to ward (moderate/stable severity only) |
| 2 | Redirect patient to an external facility |
| 3 | Call in one additional staff shift |

**Dynamics:**
- Patient arrivals follow a Poisson process (λ ∈ {1.5, 3.0, 4.5, 6.0}), split into critical/moderate/stable severity via Poisson thinning (20% / 50% / 30%).
- Ward and ICU length-of-stay follow geometric (discrete-memoryless) service times, calibrated against published clinical literature.
- Ventilator returns on ICU step-down are drawn via a hypergeometric distribution to correctly reflect unknown ventilation status among discharged patients.

Every free parameter in the environment (arrival rate, severity split, ventilator-use probability, service rates, staffing ratio, episode length) is grounded in a cited clinical or operations-research source — see Table IV of the report.

## Experimental Design

| Group | Focus | Conditions |
|---|---|---|
| A | Baseline algorithm comparison (λ=3.0) | 3 |
| B | Arrival-rate sensitivity | 9 |
| C | Reward-shaping ablation | 9 |
| D | Resource-capacity stress test | 6 |
| E | PPO network architecture | 2 |
| F | PPO hyperparameter sensitivity | 4 |
| Aƒ/Bƒ | Seed-robustness replication | 12 |
| G | Time-aware reward + mortality penalty | 3 |
| H | Overflow-penalty ablation | 5 |
| I | A2C with parallel environments | 1 |
| J | High capacity at λ=6.0 | 3 |
| K | DQN architecture/hyperparameter sensitivity | 6 |
| L | Reward redesign (idle-ward, overflow-blend, mortality, weight grid) | 14 |
| Greedy | Baselines across all of the above | 9 |

All RL agents are trained for 2×10⁵ timesteps (seed 42; replication runs use seed 123) and evaluated over 100 deterministic episodes (seed 99).

## Implementation Issues

Four environment/pipeline bugs were identified and fixed during development; full root-cause analysis is in Section VI of the report.

1. **Missing ward-patient discharge** — no per-step ward discharge mechanism existed, causing ward capacity to saturate after ~30 steps and freezing admissions for the remainder of every episode. Fixed with a geometric discharge process (p=0.05/step).
2. **Ventilator over-return on ICU step-down** — ventilators were unconditionally returned for every stepping-down patient rather than only the ~60% who used one. Fixed by tracking ventilated patients separately and drawing returns via a hypergeometric distribution.
3. **Training-pipeline save/load collision** — the model-loading helper checked only for file existence, not algorithm/experiment identity, occasionally reusing a mismatched checkpoint. Fixed via a `meta.json` sidecar that verifies experiment ID and algorithm before reuse.
4. **Replication seed identity** — replication runs originally reused the same training seed as the original runs. Fixed by applying a dedicated `REPLICATION_SEED=123` to all replication experiments.

## Getting Started

### Requirements

- Python 3.10+
- [`gymnasium`](https://gymnasium.farama.org/)
- [`stable-baselines3`](https://stable-baselines3.readthedocs.io/)
- `numpy`, `pandas`, `matplotlib`

```bash
pip install gymnasium stable-baselines3 numpy pandas matplotlib jupyter
```

### Running

```bash
jupyter notebook group-8-hospital_allocation-final-project.ipynb
```

The notebook is organized to run top-to-bottom: environment definition and validation, the greedy baseline, the experiment registry, Groups A–F, seed-robustness replication, cross-group analysis, and finally the Group G–L follow-up experiments. Trained models, TensorBoard logs, and figures are written to `output/`; existing checkpoints are loaded automatically rather than retrained where available.

## Limitations

- Single-facility model with no receiving-facility capacity or network-level triage constraints.
- Patients dropped at queue overflow leave the simulation entirely rather than propagating congestion.
- ICU staffing is not explicitly constrained (only ward beds are staff-gated).
- Arrival rate λ is held constant per condition rather than modelled as a nonhomogeneous Poisson process.
- A2C's policy collapse was not resolved within this study; entropy regularisation and/or invalid-action masking are recommended directions.
- One data-integrity flag (`B1f`/`B3f`) remains open — see Section VI/VII of the report for details.

## AI Assistance Disclosure

Claude (Anthropic) was used during debugging and analysis phases, consistent with the course's AI-usage disclosure policy — specifically for code debugging, save/load pipeline design, numerical cross-checking, reference verification, and improving the scientific writing of the report and notebook documentation. All experimental design decisions, code, results, and conclusions are the authors' own work; every AI-generated suggestion was independently verified before use.

## Citation

If referencing this work, please cite:

```
M. Imbabi, S. ElSherif, N. Mahmoud, and N. Hamada, "Hospital Resource Allocation
Under Uncertainty Using Reinforcement Learning," CISC 856 Final Project,
Queen's University, 2026.
```
