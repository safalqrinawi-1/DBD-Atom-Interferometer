# DBD RL Environment – Scenario 1 (Detuning Optimization)

Reinforcement learning environment for optimizing the time‑dependent detuning Δ(t) in Double Bragg Diffraction.

## Overview

- **Units**: Dimensionless with ω_rec = 1 and ħ = 1.
- **Physics**: 5×5 Double‑Bragg Hamiltonian in the basis [|0⟩, |±2ℏk⟩_sym/anti, |±4ℏk⟩_sym].
  - Ω(t) is Gaussian with peak Ω_R and width τ_g.
  - Couplings: 0↔(+2)_sym has √2·g, (+2)_sym↔(+4)_sym = g, (−2)_anti↔(−4)_sym = g.
  - g = Ω(t)·[cos((4ω_rec + Δ)t) + ε_pol]; optional ε_pol offset, default 0.
  - Doppler mixing term between (±2) sym/anti: H[1,2] = 4p·ω_rec (p=0 here).
- **State**: `[time_norm, pop_0, pop_+2, pop_-2, pop_+4, pop_-4]` (bare momentum populations, derived from the symmetric basis).
- **Action**: One detuning value per step in [-1,1], scaled by `max_detuning · ω_rec`.
- **Episode**: 30 steps over a pulse of duration 1.0 (dimensionless time).
- **Reward**: Dense; per step increase of symmetric population plus a final‑step bonus equal to the final symmetric population.

## Files
- `dbd_env.py` – Gymnasium environment
- `dbd_physics.py` – Dimensionless Hamiltonian and integrator
- `train_simple.py` – RLlib PPO training (simple, single-file)
- `inference_rllib.py` – RLlib checkpoint inference helper
- `test_dbd.py` – Optional quick visualization

## Install

```bash
pip install numpy scipy gymnasium torch ray[rllib] pygame
```

## Train (RLlib PPO)

```bash
python train_simple.py
```

## Inference

```bash
python inference_rllib.py
```

## Configuration snippet

```python
env = DBDEnv(
    max_detuning=20.0,
    steps_per_episode=30,
    pulse_duration=1.0,
)
```

## Notes

- Symmetric population reported by the environment corresponds to |(+2)_sym|^2, while observations expose bare momentum populations for compatibility with RL.
