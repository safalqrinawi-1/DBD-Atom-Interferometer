#!/usr/bin/env python3
"""
DBD RL Environment - RLlib Inference Script

Run inference on a trained PPO policy (RLModule) for the DBDEnv.
Provide the absolute path to the RLlib checkpoint directory via --checkpoint.

Examples:
  python inference_rllib.py --checkpoint "D:/path/to/checkpoint_000050"
  python inference_rllib.py --eps-pol 0.1
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch

# Ensure local imports work
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ray.rllib.core.rl_module import RLModule
from dbd_env import DBDEnv

# Set this to your absolute checkpoint path
DEFAULT_CHECKPOINT: str = r"C:/Users/jbs/ray_results/DBD_PPO_20251013_004631/PPO_DBDEnv-v0_ea1e9_00000_0_2025-10-13_00-46-31/checkpoint_000004"


def resolve_module_dir(checkpoint_path: Path) -> Path:
    """Resolve the RLModule directory (default_policy) from a checkpoint path."""
    # If the user already points to the rl_module default policy directory
    if checkpoint_path.name == "default_policy" and checkpoint_path.is_dir():
        return checkpoint_path

    # If pointing to the rl_module dir directly
    if checkpoint_path.name == "rl_module":
        candidate = checkpoint_path / "default_policy"
        if candidate.is_dir():
            return candidate

    # Typical structure under a checkpoint dir
    candidate = checkpoint_path / "learner_group" / "learner" / "rl_module" / "default_policy"
    if candidate.is_dir():
        return candidate

    # Sometimes checkpoints are nested one level deeper
    nested = checkpoint_path / "checkpoint" / "learner_group" / "learner" / "rl_module" / "default_policy"
    if nested.is_dir():
        return nested

    raise FileNotFoundError(
        f"Could not resolve RLModule 'default_policy' dir from: {checkpoint_path}"
    )


def run_inference(module_dir: Path, render: bool = True, eps_pol: float = 0.0) -> dict:
    """Run one episode of inference and return comprehensive results."""
    # Load RLModule from checkpoint
    rl_module = RLModule.from_checkpoint(module_dir)

    # Create environment with eps_pol
    env = DBDEnv(eps_pol=eps_pol)

    episode_return = 0.0
    done = False
    step_count = 0

    # Track episode data
    episode_data = {
        'steps': [],
        'actions': [],
        'rewards': [],
        'populations': [],
        'symmetric_populations': [],
        'observations': []
    }

    obs, info = env.reset()

    print("🎯 Starting DBD Inference Episode")
    print("=" * 60)
    print(f"Polarization Error (ε_pol): {eps_pol:.4f}")
    print(f"Initial state: {obs}")
    print(f"Initial populations: |0⟩={info['populations'][0]:.3f} |+2⟩={info['populations'][1]:.3f} |-2⟩={info['populations'][2]:.3f} |+4⟩={info['populations'][3]:.3f} |-4⟩={info['populations'][4]:.3f}")
    print(f"Initial symmetric population (DBD efficiency): {info['symmetric_population']:.4f}")
    print("-" * 60)

    while not done:
        step_count += 1

        # Prepare batch (B=1)
        obs_batch = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            model_outputs = rl_module.forward_inference({"obs": obs_batch})

        # Continuous action distribution parameters; take mean
        action_dist_params = model_outputs["action_dist_inputs"][0].cpu().numpy()

        # For Gaussian parameterization [mean, log_std] per dimension, take mean only
        action_mean = action_dist_params[: env.action_space.shape[0]]
        greedy_action = np.clip(
            action_mean,
            a_min=env.action_space.low,
            a_max=env.action_space.high,
        )

        # Store step data
        episode_data['steps'].append(step_count)
        episode_data['actions'].append(greedy_action[0])
        episode_data['observations'].append(obs.copy())
        episode_data['populations'].append(info['populations'].copy())
        episode_data['symmetric_populations'].append(info['symmetric_population'])

        obs, reward, terminated, truncated, info = env.step(greedy_action.astype(np.float32))
        episode_return += float(reward)
        done = bool(terminated or truncated)

        # Store post-step data
        episode_data['rewards'].append(float(reward))

        if render:
            print(f"Step {step_count:2d} | Action: {greedy_action[0]:+.4f} | Reward: {reward:+.4f}")
            print(f"         Populations: |0⟩={info['populations'][0]:.3f} |+2⟩={info['populations'][1]:.3f} |-2⟩={info['populations'][2]:.3f} |+4⟩={info['populations'][3]:.3f} |-4⟩={info['populations'][4]:.3f}")
            print(f"         Symmetric Pop: {info['symmetric_population']:.4f} | Time: {obs[0]*env.pulse_duration*1e3:.1f} μs")
            print("-" * 40)

    # Final results
    print("\n" + "=" * 60)
    print("🎯 FINAL EPISODE RESULTS")
    print("=" * 60)
    print("📊 Final Atomic State Distribution:")
    final_pops = info['populations']
    print(f"   |0⟩  (Ground state):     {final_pops[0]:.4f} ({final_pops[0]*100:.1f}%)")
    print(f"   |+2ℏkL⟩ (Excited +):     {final_pops[1]:.4f} ({final_pops[1]*100:.1f}%)")
    print(f"   |-2ℏkL⟩ (Excited -):     {final_pops[2]:.4f} ({final_pops[2]*100:.1f}%)")
    print(f"   |+4ℏkL⟩ (Higher +):      {final_pops[3]:.4f} ({final_pops[3]*100:.1f}%)")
    print(f"   |-4ℏkL⟩ (Higher -):      {final_pops[4]:.4f} ({final_pops[4]*100:.1f}%)")

    print("\n🎯 DBD EFFICIENCY:")
    print(f"   Polarization Error (ε_pol): {eps_pol:.4f}")
    print(f"   Symmetric population: {info['symmetric_population']:.4f} ({info['symmetric_population']*100:.1f}%)")
    print(f"   Total excited population: {final_pops[1]+final_pops[2]+final_pops[3]+final_pops[4]:.4f} ({(final_pops[1]+final_pops[2]+final_pops[3]+final_pops[4])*100:.1f}%)")
    print(f"   Episode return: {episode_return:.4f}")

    # Performance analysis
    max_symmetric = max(episode_data['symmetric_populations'])
    final_symmetric = info['symmetric_population']
    print("\n📈 Performance Summary:")
    print(f"   Peak symmetric population: {max_symmetric:.4f} (at step {episode_data['symmetric_populations'].index(max_symmetric)+1})")
    print(f"   Final symmetric population: {final_symmetric:.4f}")
    print(f"   Efficiency trend: {'📈 Improved' if final_symmetric > episode_data['symmetric_populations'][0] else '📉 Declined'}")

    print("=" * 60)

    env.close()

    return {
        'episode_return': episode_return,
        'final_populations': final_pops,
        'final_symmetric_population': info['symmetric_population'],
        'episode_data': episode_data,
        'max_symmetric_population': max_symmetric,
        'eps_pol': eps_pol
    }


def run_inference_from_checkpoint(checkpoint: str, render: bool = True, eps_pol: float = 0.0) -> dict:
    """Convenience API: Pass an absolute checkpoint path and run one episode."""
    module_dir = resolve_module_dir(Path(checkpoint).resolve())
    return run_inference(module_dir, render=render, eps_pol=eps_pol)


def main():
    parser = argparse.ArgumentParser(description="DBDEnv RLlib inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Absolute path to RLlib checkpoint dir or to rl_module/default_policy.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Print ANSI render each step (default: True).",
    )
    parser.add_argument(
        "--eps-pol",
        type=float,
        default=0.0,
        help="Polarization error ε_pol [0.0-0.3] (default: 0.0)",
    )
    args = parser.parse_args()

    # Prefer CLI arg; if missing, fall back to DEFAULT_CHECKPOINT constant
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
    elif DEFAULT_CHECKPOINT:
        checkpoint_path = Path(DEFAULT_CHECKPOINT).resolve()
    else:
        print("Please set DEFAULT_CHECKPOINT in this script or pass --checkpoint <ABSOLUTE_PATH>.")
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    module_dir = resolve_module_dir(checkpoint_path)
    results = run_inference(module_dir, render=args.render, eps_pol=args.eps_pol)
    print(f"\nEpisode completed! Final return: {results['episode_return']:.4f}")
    print(f"Final DBD efficiency: {results['final_symmetric_population']:.4f} ({results['final_symmetric_population']*100:.1f}%)")


def run_dbd_inference_simple(checkpoint_path: str = None, eps_pol: float = 0.0) -> dict:
    """Simple function to run DBD inference. Use DEFAULT_CHECKPOINT if no path provided."""
    if checkpoint_path is None:
        if not DEFAULT_CHECKPOINT:
            raise ValueError("No checkpoint provided and DEFAULT_CHECKPOINT is empty")
        checkpoint_path = DEFAULT_CHECKPOINT

    return run_inference_from_checkpoint(checkpoint_path, render=True, eps_pol=eps_pol)


if __name__ == "__main__":
    main()