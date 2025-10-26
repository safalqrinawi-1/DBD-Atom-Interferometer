# DBD RL Environment – Scenario 5 (σ_p Gaussian + ε_pol, Beam‑Splitting)

Beam‑splitting optimization with Gaussian momentum width and polarization errors; RLlib PPO training.

## Overview

- **Units**: Dimensionless with ω_rec = 1 and ħ = 1.
- **Physics**: Dimensionless 5×5 Double‑Bragg Hamiltonian in symmetric ±2 basis with ε_pol and Doppler mixing H[1,2] = 4p·ω_rec.
- **Ensemble**: Momentum samples drawn from `N(0, σ_p)` with σ_p = 0.05 (ℏk units).
- **State**: `[time_norm, pop_0, pop_+2, pop_-2, pop_+4, pop_-4]` (ensemble‑averaged bare populations).
- **Action**: Detuning in [-1,1] scaled by `max_detuning · ω_rec`.
- **Episode**: 30 steps; pulse duration 1.0.
- **Reward**: Step improvement in beam‑splitting score plus final‑step bonus.

## Files

- `dbd_env.py` – Gymnasium environment (Gaussian momentum width + ε_pol)
- `dbd_physics.py` – Dimensionless Hamiltonian and ensemble evolution
- `train_simple.py` – RLlib PPO training
- `inference_rllib.py` – RLlib checkpoint inference helper

## Install & Run

```bash
pip install numpy scipy gymnasium torch ray[rllib] pygame
python train_simple.py
python inference_rllib.py --checkpoint "path/to/checkpoint" --episodes 10
```

## Config snippet

```python
env = DBDEnv(
    max_detuning=20.0,
    steps_per_episode=30,
    pulse_duration=1.0,
    momentum_width=0.05,
    num_momentum_samples=9,
)
```
