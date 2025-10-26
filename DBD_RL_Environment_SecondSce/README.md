# DBD RL Environment – Scenario 2 (Polarization Error)

Reinforcement learning environment to optimize Δ(t) with polarization error in Double Bragg Diffraction.

## Overview

- **Units**: Dimensionless with ω_rec = 1 and ħ = 1.
- **Physics**: Same 5×5 Double‑Bragg Hamiltonian as Scenario 1 with ε_pol ≠ 0.
- **State**: `[time_norm, pop_0, pop_+2, pop_-2, pop_+4, pop_-4]`.
- **Action**: Detuning in [-1,1] scaled by `max_detuning · ω_rec`.
- **Episode**: 30 steps over pulse duration 1.0.
- **Reward**: Dense; increase in symmetric population plus final‑step bonus.

## Polarization Error

- Parameter `polarization_error` maps to ε_pol in the Hamiltonian via `cos((4ω_rec+Δ)t)+ε_pol`.
- Range typically [0, 0.1]. Default 0.0.

## Files
- `dbd_env.py` – Gymnasium environment (dimensionless)
- `dbd_physics.py` – Dimensionless Hamiltonian with ε_pol
- `train_simple.py` – RLlib PPO training
- `inference_rllib.py` – Inference helper
- `test_dbd.py` – Optional visualization

## Install

```bash
pip install numpy scipy gymnasium torch ray[rllib] pygame
```

## Train

```bash
python train_simple.py
```

## Inference

```bash
python inference_rllib.py
```

## Config snippet

```python
env = DBDEnv(
    max_detuning=20.0,
    steps_per_episode=30,
    pulse_duration=1.0,
    polarization_error=0.05,
)
```
