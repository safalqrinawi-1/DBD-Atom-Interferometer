#!/usr/bin/env python3
import os
import sys
import ray
from ray.rllib.algorithms.ppo import PPO
from ray import tune

# --- Env registration (unchanged) ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dbd_env import DBDEnv

def create_dbd_env(config=None):
    return DBDEnv(
        sample_conditions=True,
        momentum_range=(-0.3, 0.3),
        eps_pol_range=(-0.05, 0.05)  # keep as-is unless you want 0..0.3
    )

tune.register_env("DBDEnv-v0", create_dbd_env)

# --- Start Ray ---
ray.init(ignore_reinit_error=True)

checkpoint = r"C:/Users/jbs/ray_results/DBD_D1_Cost_Training/PPO_DBDEnv-v0_27598_00000_0_lr=0.0003_2025-10-14_11-52-46/checkpoint_000000"
print("Loading checkpoint...")

# Load once so we can grab the original (frozen) config
loaded = PPO.from_checkpoint(checkpoint)

# Make a mutable copy, change LR, then rebuild and restore
cfg = loaded.get_config().copy(copy_frozen=False)
cfg.training(lr=5e-5)

# Option A (robust across Ray versions): build new Algorithm and restore
algo = cfg.build()
algo.restore(checkpoint)

# Option B (if supported by your Ray version): 
# algo = PPO.from_checkpoint(checkpoint, config=cfg)

print("Resuming training with LR = 5e-5")
print("Target: episode_return_mean = 1.75")

for i in range(300):
    result = algo.train()
    # New API stack often provides the scalar at top-level; fall back if nested
    ret = result.get('episode_return_mean',
                     result.get('env_runners', {}).get('episode_return_mean', float('nan')))

    if i % 5 == 0:
        print(f"Iter {i:3d}: Return = {ret:.4f}")

    if i % 20 == 0:
        checkpoint_path = algo.save()
        print(f"  Saved: {checkpoint_path}")

    if ret >= 1.75:
        print(f"\n✓ SUCCESS! Reached {ret:.4f}")
        break

final_checkpoint = algo.save()
print(f"\nFinal checkpoint: {final_checkpoint}")
