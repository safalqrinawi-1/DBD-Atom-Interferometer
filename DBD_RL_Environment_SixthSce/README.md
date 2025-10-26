# DBD RL Environment – Scenario 6 (Real‑Time Adaptation, Beam‑Splitting)

Beam‑splitting optimization with real‑time parameter drift/adaptation and Gaussian momentum ensemble; RLlib PPO training.

## Overview

- **Units**: Dimensionless with ω_rec = 1 and ħ = 1.
- **Physics**: Dimensionless 5×5 Double‑Bragg Hamiltonian in symmetric ±2 basis, with ε_pol and Doppler mixing H[1,2] = 4p·ω_rec.
- **Adaptation**: Polarization error and momentum width drift during episodes; physics updates online.
- **State**: `[time_norm, pop_0, pop_+2, pop_-2, pop_+4, pop_-4, pol_error, mom_width]`.
- **Action**: Detuning in [-1,1] scaled by `max_detuning · ω_rec`.
- **Episode**: 30 steps; pulse duration 1.0.
- **Reward**: Step improvement in beam‑splitting score plus final‑step bonus.

## Files

- `dbd_env.py` – Gymnasium environment (real‑time adaptation)
- `dbd_physics.py` – Dimensionless Hamiltonian with parameter drift and ensemble
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
)
```
