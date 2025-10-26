#!/usr/bin/env python3
"""
RLlib Training Script for DBD Environment with D1 Cost Function
"""

import os
import sys
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.tuner import Tuner

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_dbd_env(config=None):
    """Create DBD environment for Ray workers."""
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from dbd_env import DBDEnv
    
    # Environment now handles sampling internally - just create it
    return DBDEnv(
        sample_conditions=True,        # Enable ensemble sampling
        momentum_range=(-0.3, 0.3),    # Momentum distribution
        eps_pol_range=(-0.05, 0.05)    # Polarization error distribution
    )

# Register environment
tune.register_env("DBDEnv-v0", create_dbd_env)

def train():
    """Train PPO agent using Ray Tune."""
    ray.init()

    # PPO configuration
    config = (
        PPOConfig()
        .environment("DBDEnv-v0")
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=24,
            rollout_fragment_length='auto',
        )
        .training(
            lr=tune.grid_search([0.0003, 0.00003, 0.000003]),
            train_batch_size_per_learner=2880,
        )
        .framework("torch")
    )

    # Tuner configuration
    tuner = Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"env_runners/episode_return_mean": 1.6},  # Target ~90% efficiency
            verbose=2,
            name="DBD_D1_Cost_Training",
        ),
    )

    # Start training
    results = tuner.fit()

    print("Training completed!")
    
    # Get best result with proper metric
    try:
        best_result = results.get_best_result(metric="env_runners/episode_return_mean", mode="max")
        print(f"Best result: {best_result}")
    except Exception as e:
        print(f"Could not get best result: {e}")
        print("All results:")
        for result in results:
            print(result)

    return results

if __name__ == "__main__":
    results = train()