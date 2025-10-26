#!/usr/bin/env python3
"""
DBD RL Training Script

Train a PPO agent to optimize time-dependent detuning for Double Bragg Diffraction.
The agent learns to maximize symmetric state population (beam-splitter efficiency).
"""

import os
import sys
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from datetime import datetime

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def create_dbd_env(config=None):
    """Create DBD environment for Ray workers."""
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from dbd_env import DBDEnv
    return DBDEnv()


# Register environment
tune.register_env("DBDEnv-v0", create_dbd_env)


def test_environment_before_training():
    """Test environment before starting training."""
    print("\n" + "="*70)
    print("🧪 PRE-TRAINING ENVIRONMENT TEST")
    print("="*70)
    
    try:
        from dbd_env import DBDEnv
        import numpy as np
        
        # Test 1: Create environment
        print("\n[1/4] Creating environment...")
        env = create_dbd_env()
        print("✅ Environment created successfully")
        
        # Test 2: Reset
        print("\n[2/4] Testing reset...")
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Initial populations: {info['populations']}")
        print(f"  Initial symmetric pop: {info['symmetric_population']:.6f}")
        
        # Critical check
        if info['populations'][0] < 0.99:
            raise RuntimeError(
                f"❌ PHYSICS BUG DETECTED!\n"
                f"Initial population should be in |p⟩ (index 0)\n"
                f"Got: {info['populations']}\n"
                f"Expected: [1.0, 0.0, 0.0, 0.0, 0.0]\n"
                f"Fix dbd_physics.py before training!"
            )
        print("✅ Initial state is correct")
        
        # Test 3: Step
        print("\n[3/4] Testing step with zero detuning...")
        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Reward: {reward:.6f}")
        print(f"  Symmetric pop: {info['symmetric_population']:.6f}")
        print(f"  Terminated: {terminated}")
        print("✅ Step works correctly")
        
        # Test 4: Full episode
        print("\n[4/4] Testing full episode (30 steps)...")
        env.reset()
        episode_return = 0.0
        for i in range(30):
            action = np.array([0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
        
        print(f"  Episode return: {episode_return:.6f}")
        print(f"  Final symmetric pop: {info['symmetric_population']:.6f}")
        print(f"  Final populations: {info['populations']}")
        
        if 0.60 < info['symmetric_population'] < 0.95:
            print("✅ Episode completed successfully")
        else:
            print(f"⚠️  Warning: Unexpected final symmetric pop: {info['symmetric_population']:.4f}")
        
        print("\n" + "="*70)
        print("✅ ALL PRE-TRAINING TESTS PASSED!")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ENVIRONMENT TEST FAILED: {e}")
        print("\nFix the error before training!")
        print("="*70 + "\n")
        import traceback
        traceback.print_exc()
        return False


def train_dbd_agent(
    target_reward: float = 1.70,
    max_iterations: int = 200,
    num_workers: int = 2,
    num_envs_per_worker: int = 4,
    learning_rate: float = 3e-4,
    checkpoint_freq: int = 10
):
    """
    Train PPO agent for DBD optimization.
    
    Args:
        target_reward: Stop when average reward reaches this value
        max_iterations: Maximum training iterations
        num_workers: Number of parallel workers
        num_envs_per_worker: Environments per worker
        learning_rate: Learning rate for PPO
        checkpoint_freq: Save checkpoint every N iterations
    
    Returns:
        Training results or None if failed
    """
    
    # Test environment first
    if not test_environment_before_training():
        print("\n❌ Cannot start training - environment test failed!")
        return None
    
    # Initialize Ray
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True)
    
    # Configure PPO
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .environment("DBDEnv-v0")
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=num_envs_per_worker,
            rollout_fragment_length=200,
        )
        .training(
            lr=learning_rate,
            train_batch_size_per_learner=num_workers * num_envs_per_worker * 200,
            gamma=0.99,  # Discount factor
            lambda_=0.95,  # GAE parameter
            num_sgd_iter=10,  # Number of SGD iterations per training batch
            minibatch_size=128,  # ← FIXED: was 'sgd_minibatch_size'
            clip_param=0.2,  # PPO clipping parameter
        )
        .framework("torch")
        .debugging(log_level="INFO")
        .resources(num_gpus=0)  # Set to 1 if you have GPU
    )
    
    # Create unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"DBD_PPO_{timestamp}"
    
    # Create tuner
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            name=run_name,
            stop={
                "env_runners/episode_return_mean": target_reward,
                "training_iteration": max_iterations
            },
            verbose=2,
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=checkpoint_freq,
                num_to_keep=5,  # Keep last 5 checkpoints
            ),
        ),
    )
    
    # Print training info
    print("\n" + "="*70)
    print(f"🚀 STARTING DBD RL TRAINING: {run_name}")
    print("="*70)
    print(f"Target reward: {target_reward}")
    print(f"Max iterations: {max_iterations}")
    print(f"Workers: {num_workers}")
    print(f"Envs per worker: {num_envs_per_worker}")
    print(f"Total parallel envs: {num_workers * num_envs_per_worker}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoint frequency: every {checkpoint_freq} iterations")
    print("="*70)
    print("\nTraining will stop when:")
    print(f"  1. Average reward reaches {target_reward}, OR")
    print(f"  2. {max_iterations} iterations complete")
    print("="*70 + "\n")
    
    try:
        # Start training
        print("Training started... (this may take 10-60 minutes)\n")
        results = tuner.fit()
        
        # Training completed
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        
        # Get best result
        best_result = results.get_best_result(
            metric="env_runners/episode_return_mean",
            mode="max"
        )
        best_checkpoint = best_result.checkpoint
        best_reward = best_result.metrics.get("env_runners/episode_return_mean", 0.0)
        iterations = best_result.metrics.get("training_iteration", 0)
        
        print(f"\n📊 Training Summary:")
        print(f"  Total iterations: {iterations}")
        print(f"  Best reward: {best_reward:.4f}")
        print(f"  Best checkpoint: {best_checkpoint.path}")
        
        print(f"\n💾 Checkpoint Location:")
        print(f"  {best_checkpoint.path}")
        
        print(f"\n🎯 Next Steps:")
        print(f"  1. Update DEFAULT_CHECKPOINT in inference_rllib.py:")
        print(f'     DEFAULT_CHECKPOINT = r"{best_checkpoint.path}"')
        print(f"  2. Run inference: python inference_rllib.py")
        
        print("="*70 + "\n")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DBD RL Agent")
    parser.add_argument(
        "--target-reward",
        type=float,
        default=1.70,
        help="Stop when average reward reaches this value (default: 1.70)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Maximum training iterations (default: 200)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )
    parser.add_argument(
        "--envs-per-worker",
        type=int,
        default=4,
        help="Environments per worker (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Checkpoint frequency in iterations (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Start training
    results = train_dbd_agent(
        target_reward=args.target_reward,
        max_iterations=args.max_iterations,
        num_workers=args.workers,
        num_envs_per_worker=args.envs_per_worker,
        learning_rate=args.lr,
        checkpoint_freq=args.checkpoint_freq
    )
    
    if results is not None:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n❌ Training failed. Check errors above.")
        sys.exit(1)# Configure