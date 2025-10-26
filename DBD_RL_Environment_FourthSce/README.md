# DBD RL Environment – Scenario 4 (ε_pol + Doppler, Beam‑Splitting)

Scenario targeting beam‑splitting fidelity under polarization errors and Doppler broadening, with RLlib PPO training.

## Overview

- **Units**: Dimensionless with ω_rec = 1 and ħ = 1.
- **Physics**: Dimensionless 5×5 Double‑Bragg Hamiltonian with symmetric/antisymmetric ±2 basis, ε_pol offset, and Doppler mixing H[1,2] = 4p·ω_rec.
- **Ensemble**: Uniform momentum samples over `p ∈ [-0.3, 0.3]`, averaged.
- **State**: `[time_norm, pop_0, pop_+2, pop_-2, pop_+4, pop_-4]` (ensemble‑averaged bare populations).
- **Action**: Detuning in [-1,1] scaled by `max_detuning · ω_rec`.
- **Episode**: 30 steps; pulse duration 1.0.
- **Reward**: Step improvement in beam‑splitting score plus final‑step bonus.

## Files

- `dbd_env.py` – Gymnasium environment (ensemble + ε_pol)
- `dbd_physics.py` – Dimensionless Hamiltonian with ε_pol and Doppler mixing
- `train_simple.py` – RLlib PPO training (simple)
- `inference_rllib.py` – RLlib checkpoint inference helper

## Install & Run

```bash
pip install numpy scipy gymnasium torch ray[rllib] pygame
python train_simple.py
python inference_rllib.py --checkpoint "path/to/checkpoint" --episodes 10
```

## Beam‑Splitting Score

`score = 0.5·(1 − |P(+2) − P(−2)|) + 0.5·(P(+2) + P(−2))` on ensemble‑averaged populations.

## Config snippet

```python
env = DBDEnv(
    max_detuning=20.0,
    steps_per_episode=30,
    pulse_duration=1.0,
    momentum_range=(-0.3, 0.3),
    num_momentum_samples=7,
)
```
