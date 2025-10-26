# DBD RL Environment – Scenario 3 (Doppler Ensemble, Beam‑Splitting)

RL environment optimizing Δ(t) over a momentum ensemble with Doppler mixing; target metric is 50/50 beam splitting in |±2ℏk⟩.

## Overview

- **Units**: Dimensionless with ω_rec = 1 and ħ = 1.
- **Physics**: Dimensionless 5×5 Double‑Bragg Hamiltonian with symmetric/antisymmetric ±2 basis and Doppler mixing H[1,2] = 4p·ω_rec.
- **Ensemble**: Uniform momentum samples over `p ∈ [-0.3, 0.3]` (ℏk units), averaged in the observation and metrics.
- **State**: `[time_norm, pop_0, pop_+2, pop_-2, pop_+4, pop_-4]` (ensemble‑averaged bare populations).
- **Action**: Detuning in [-1,1] scaled by `max_detuning · ω_rec`.
- **Episode**: 30 steps, pulse duration 1.0.
- **Reward**: Step improvement in beam‑splitting score plus final‑step bonus.

## Beam‑Splitting Score

Let `p_plus2` and `p_minus2` denote ensemble‑averaged populations in |+2ℏk⟩ and |-2ℏk⟩. Define

`balance = 1 − |p_plus2 − p_minus2| / (p_plus2 + p_minus2 + 1e-8)` and `score = balance × min(1, p_plus2 + p_minus2)`.

The reward uses the score improvement each step; the final step adds the current score.

## Files
- `dbd_env.py` – Gymnasium environment (ensemble)
- `dbd_physics.py` – Dimensionless Hamiltonian with Doppler mixing across samples
- `train_simple.py` – RLlib PPO training
- `inference_rllib.py` – Inference helper
- `test_dbd.py` – Optional visualization

## Install & Run

```bash
pip install numpy scipy gymnasium torch ray[rllib] pygame
python train_simple.py
python inference_rllib.py
```

## Config snippet

```python
env = DBDEnv(
    max_detuning=20.0,
    steps_per_episode=30,
    pulse_duration=1.0,
    momentum_range=(-0.3, 0.3),
    num_momentum_samples=5,
)
```
