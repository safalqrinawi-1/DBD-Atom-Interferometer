#!/usr/bin/env python3
"""
DBD RL Inference - Real-Time Adaptation

Run inference to test how well the agent adapts to:
- Randomized polarization errors [0, 20%]
- Randomized momentum widths [0.03, 0.07]ℏkL
- Parameter drift during episodes
"""

import argparse
import sys
from pathlib import Path
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dbd_env import DBDEnv

try:
    from ray.rllib.algorithms.ppo import PPO
    from ray import tune
    RLLIB_AVAILABLE = True
except ImportError:
    print("Ray RLlib not available. Install: pip install ray[rllib]")
    RLLIB_AVAILABLE = False


def find_latest_checkpoint(base_dir: str = None) -> Path:
    """Find latest checkpoint automatically."""
    if base_dir is None:
        possible_dirs = [
            Path.home() / "ray_results",
            Path("C:/Users/jbs/ray_results"),
            Path.cwd() / "ray_results"
        ]
        
        base_path = None
        for d in possible_dirs:
            if d.exists():
                base_path = d
                break
        
        if base_path is None:
            raise FileNotFoundError(
                "Results directory not found. Run training first: python train_simple.py"
            )
    else:
        base_path = Path(base_dir)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Results directory not found: {base_path}")
    
    # Find all PPO training runs
    run_dirs = []
    for pattern in ["PPO_*", "DBD_*"]:
        run_dirs.extend(list(base_path.glob(f"**/{pattern}")))
    
    if not run_dirs:
        raise FileNotFoundError(f"No training runs in: {base_path}")
    
    # Get most recent run
    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    print(f"🔍 Found run: {latest_run.name}")
    
    # Find checkpoints
    checkpoints = list(latest_run.glob("checkpoint_*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in: {latest_run}")
    
    # Get highest checkpoint
    checkpoint_numbers = []
    for cp in checkpoints:
        try:
            num_str = cp.name.replace("checkpoint_", "").replace("_", "").lstrip("0")
            num = int(num_str) if num_str else 0
            checkpoint_numbers.append((num, cp))
        except ValueError:
            continue
    
    if not checkpoint_numbers:
        raise FileNotFoundError(f"No valid checkpoints in: {latest_run}")
    
    latest_checkpoint = max(checkpoint_numbers, key=lambda x: x[0])[1]
    return latest_checkpoint


def run_inference(checkpoint_path: Path, num_episodes: int = 5, render: bool = True) -> dict:
    """Run inference and analyze real-time adaptation."""
    if not RLLIB_AVAILABLE:
        raise ImportError("Ray RLlib not available")

    # Register environment
    tune.register_env("DBDEnv-v0", lambda cfg: DBDEnv())
    
    # Load algorithm
    print(f"Loading model from: {checkpoint_path}")
    try:
        algo = PPO.from_checkpoint(str(checkpoint_path))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise

    env = DBDEnv()

    episode_rewards = []
    beam_scores = []
    final_populations = []
    adaptation_data = []

    print(f"\n{'='*70}")
    print(f"Running {num_episodes} episodes - Testing Real-Time Adaptation")
    print("=" * 70)

    for episode in range(num_episodes):
        print(f"\n📊 Episode {episode + 1}/{num_episodes}")
        print("-" * 70)

        obs, info = env.reset()
        total_reward = 0.0

        # Track initial parameters
        init_pol = info['initial_polarization_error']
        init_mom = info['initial_momentum_width']
        
        print(f"Scenario: {info['scenario']}")
        print(f"Initial Pol Error: {init_pol:.3f} (randomized)")
        print(f"Initial Mom Width: {init_mom:.3f} (randomized)")
        print("-" * 70)

        # Track parameter changes
        pol_history = [init_pol]
        mom_history = [init_mom]
        beam_history = []

        done = False
        step_index = 0
        
        while not done:
            step_index += 1

            # Get action
            action = algo.compute_single_action(obs, explore=False)
            if np.isscalar(action):
                action = np.array([action], dtype=np.float32)
            else:
                action = np.asarray(action, dtype=np.float32)
            
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Execute
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)

            # Track changes
            pol_history.append(info['current_polarization_error'])
            mom_history.append(info['current_momentum_width'])
            beam_history.append(info['beam_splitting_score'])

            # Print progress
            if render and (step_index % 5 == 0 or done):
                print(f"  Step {step_index:2d}: "
                      f"Reward={float(reward):7.4f}, "
                      f"Beam={info['beam_splitting_score']:.4f}, "
                      f"Pol={info['current_polarization_error']:.3f}, "
                      f"Mom={info['current_momentum_width']:.3f}")

        # Store results
        episode_rewards.append(total_reward)
        beam_scores.append(info['beam_splitting_score'])
        final_populations.append(info['populations'])
        
        # Store adaptation data
        adaptation_data.append({
            'init_pol': init_pol,
            'final_pol': info['current_polarization_error'],
            'init_mom': init_mom,
            'final_mom': info['current_momentum_width'],
            'pol_history': pol_history,
            'mom_history': mom_history,
            'beam_history': beam_history,
            'pol_change': info['current_polarization_error'] - init_pol,
            'mom_change': info['current_momentum_width'] - init_mom
        })

        print(f"\n✓ Episode {episode + 1} completed")
        print(f"  Reward: {total_reward:.4f}")
        print(f"  Final beam score: {info['beam_splitting_score']:.4f}")
        print(f"  Parameter changes:")
        print(f"    Pol: {init_pol:.3f} → {info['current_polarization_error']:.3f} "
              f"(Δ={info['current_polarization_error']-init_pol:+.3f})")
        print(f"    Mom: {init_mom:.3f} → {info['current_momentum_width']:.3f} "
              f"(Δ={info['current_momentum_width']-init_mom:+.3f})")

    # Summary
    print(f"\n{'='*70}")
    print("🎯 Real-Time Adaptation Analysis")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"\nPerformance:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"  Mean Beam Score: {np.mean(beam_scores):.4f} ± {np.std(beam_scores):.4f}")
    print(f"  Best Beam Score: {np.max(beam_scores):.4f}")
    
    # Adaptation analysis
    pol_changes = [d['pol_change'] for d in adaptation_data]
    mom_changes = [d['mom_change'] for d in adaptation_data]
    
    print(f"\nParameter Adaptation:")
    print(f"  Polarization Error Changes:")
    print(f"    Mean: {np.mean(pol_changes):+.4f}")
    print(f"    Std:  {np.std(pol_changes):.4f}")
    print(f"    Max abs: {np.max(np.abs(pol_changes)):.4f}")
    print(f"  Momentum Width Changes:")
    print(f"    Mean: {np.mean(mom_changes):+.4f}")
    print(f"    Std:  {np.std(mom_changes):.4f}")
    print(f"    Max abs: {np.max(np.abs(mom_changes)):.4f}")
    
    # Check adaptation effectiveness
    beam_variance = np.std(beam_scores)
    print(f"\nAdaptation Effectiveness:")
    print(f"  Beam score variance: {beam_variance:.4f}")
    if beam_variance < 0.1:
        print(f"  ✓ Good adaptation - stable performance despite parameter changes")
    else:
        print(f"  ⚠ High variance - agent struggles with parameter changes")
    
    avg_pops = np.mean(final_populations, axis=0)
    print(f"\nAverage Populations:")
    print(f"  |0⟩      = {avg_pops[0]:.4f}")
    print(f"  |+2ℏk⟩  = {avg_pops[1]:.4f}")
    print(f"  |-2ℏk⟩  = {avg_pops[2]:.4f}")
    print(f"  |+4ℏk⟩  = {avg_pops[3]:.4f}")
    print(f"  |-4ℏk⟩  = {avg_pops[4]:.4f}")
    print(f"  ±2ℏk    = {avg_pops[1] + avg_pops[2]:.4f}")
    print("=" * 70)

    algo.stop()
    
    return {
        'episode_rewards': episode_rewards,
        'beam_scores': beam_scores,
        'final_populations': final_populations,
        'adaptation_data': adaptation_data
    }


def main():
    parser = argparse.ArgumentParser(
        description="DBD Inference - Real-Time Adaptation"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint path (auto-finds if not provided)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes (default: 5)")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DBD RL Inference - Real-Time Adaptation Test")
    print("=" * 70)
    
    if not RLLIB_AVAILABLE:
        print("\n✗ Ray RLlib not available")
        print("Install: pip install ray[rllib]")
        return
    
    # Find checkpoint
    if args.checkpoint is None:
        print("\n🔍 Searching for latest checkpoint...")
        try:
            checkpoint_path = find_latest_checkpoint()
            print(f"✓ Found: {checkpoint_path.name}")
        except FileNotFoundError as e:
            print(f"\n✗ {e}")
            return
    else:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.exists():
            print(f"\n✗ Not found: {checkpoint_path}")
            return
        print(f"\n✓ Using: {checkpoint_path}")
    
    try:
        results = run_inference(
            checkpoint_path,
            num_episodes=args.episodes,
            render=not args.no_render
        )
        print("\n✓ Inference completed!")
        print("\n💡 The agent was tested on episodes with:")
        print("   - Random polarization errors")
        print("   - Random momentum widths")
        print("   - Parameter drift during episodes")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()