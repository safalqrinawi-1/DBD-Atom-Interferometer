"""
DBD RL Environment - Multi-Step Version for Detuning Optimization

Gymnasium environment for optimizing time-dependent detuning Δ(t) in Double Bragg Diffraction.
Uses 30-step episodes with single detuning action per step.
Focuses on maximizing symmetric state population (DBD efficiency).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Callable
import sys
import os

# Import physics
try:
    from .dbd_physics import DBDPhysics
except ImportError:
    from dbd_physics import DBDPhysics

class DBDEnv(gym.Env):
    """
    Minimal DBD environment for detuning optimization.

    Action: Time-dependent detuning Δ(t) at discrete control points
    State: [current_time, populations_5_states]
    Reward: Symmetric state population (|+2ℏkL⟩ + |-2ℏkL⟩)
    Termination: After full pulse duration
    """

    def __init__(self,
                 max_detuning: float = 20.0,  # In units of ω_rec (dimensionless)
                 steps_per_episode: int = 30,  # Exactly 30 steps per episode
                 pulse_duration: float = 1.0,  # dimensionless time units (ω_rec^{-1})
                 eps_pol: float = 0.0):  # ← ADDED: polarization error

        super().__init__()

        # Environment parameters
        self.max_detuning = max_detuning
        self.steps_per_episode = steps_per_episode
        self.pulse_duration = pulse_duration
        self.step_duration = pulse_duration / steps_per_episode
        self.eps_pol = eps_pol  # ← ADDED

        # Initialize physics engine with eps_pol
        self.physics = DBDPhysics(
            pulse_duration=self.pulse_duration,
            polarization_error=eps_pol  # ← ADDED
        )

        # Action space: single detuning value [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),  # Single detuning value per step
            dtype=np.float32
        )

        # Observation space: [time, populations_5_states]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),  # time + 5 populations
            dtype=np.float32
        )

        # Episode state
        self.current_time = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.final_reward_calculated = False
        self.previous_symmetric_population = 0.0  # For incremental rewards

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (ignored, for compatibility)

        Returns:
            observation: Initial observation [time, populations]
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset physics
        self.physics.reset()

        # Reset episode state
        self.current_time = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.final_reward_calculated = False
        self.previous_symmetric_population = 0.0  # Reset for incremental rewards

        # Get initial observation
        observation = self._get_observation()

        info = {
            'symmetric_population': self.physics.get_symmetric_population(),
            'populations': self.physics.get_populations().tolist(),
            'eps_pol': self.eps_pol  # ← ADDED
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step with single detuning value.

        Args:
            action: Single detuning value [-1, 1]

        Returns:
            observation: New observation
            reward: Dense reward (incremental improvement + final bonus at step 30)
            terminated: True only after 30 steps
            truncated: False
            info: Additional information
        """
        # Extract single detuning value from action (dimensionless ω_rec units)
        detuning_value = action[0] * self.max_detuning * self.physics.omega_rec

        # Create constant detuning function for this step
        def detuning_function(t: float) -> float:
            return detuning_value

        # Evolve the system for this time step
        evolution_result = self.physics.evolve_system(detuning_function, self.step_duration)

        # Update step count and time
        self.step_count += 1
        self.current_time = min(self.step_count * self.step_duration, self.pulse_duration)

        # Calculate reward - DENSE: incremental improvements + final bonus
        current_symmetric = evolution_result['symmetric_population']

        # Incremental reward: improvement from previous step
        incremental_reward = current_symmetric - self.previous_symmetric_population

        # Final bonus at step 30
        if self.step_count >= self.steps_per_episode:
            # Final step: add bonus equal to final symmetric population
            final_bonus = current_symmetric
            reward = incremental_reward + final_bonus
            self.final_reward_calculated = True
        else:
            # Intermediate steps: only incremental reward
            reward = incremental_reward

        # Update tracking for next step
        self.previous_symmetric_population = current_symmetric
        self.episode_reward += reward  # Track cumulative reward

        # Get observation
        observation = self._get_observation()

        # Check termination
        terminated = (self.step_count >= self.steps_per_episode)
        truncated = False

        info = {
            'symmetric_population': evolution_result['symmetric_population'],
            'populations': evolution_result['final_populations'].tolist(),
            'evolution_success': evolution_result['success'],
            'step_count': self.step_count,
            'current_time': self.current_time,
            'episode_reward': self.episode_reward,
            'eps_pol': self.eps_pol  # ← ADDED
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation [normalized_time, populations]."""
        populations = self.physics.get_populations()
        normalized_time = self.current_time / self.pulse_duration

        # Clip time to [0, 1] range
        normalized_time = np.clip(normalized_time, 0.0, 1.0)

        observation = np.concatenate([[normalized_time], populations])
        return observation.astype(np.float32, copy=False)

    def render(self, mode: str = "ansi"):
        """Render environment state."""
        if mode == "ansi":
            return self._render_ansi()
        else:
            return self._render_ansi()

    def _render_ansi(self) -> str:
        """Text-based rendering."""
        populations = self.physics.get_populations()
        symmetric_pop = self.physics.get_symmetric_population()

        output = []
        output.append(f"DBD Environment (Step {self.step_count}/{self.steps_per_episode})")
        output.append(f"Polarization Error (ε_pol): {self.eps_pol:.4f}")  # ← ADDED
        output.append(f"Time: {self.current_time:.1e}s ({self.current_time/self.pulse_duration*100:.1f}%)")
        output.append(f"Symmetric Population: {symmetric_pop:.4f}")
        if self.final_reward_calculated:
            output.append(f"Episode Reward: {self.episode_reward:.4f}")
        output.append("Populations:")
        output.append(f"  |0⟩:     {populations[0]:.4f}")
        output.append(f"  |+2ℏkL⟩: {populations[1]:.4f}")
        output.append(f"  |-2ℏkL⟩: {populations[2]:.4f}")
        output.append(f"  |+4ℏkL⟩: {populations[3]:.4f}")
        output.append(f"  |-4ℏkL⟩: {populations[4]:.4f}")

        return "\n".join(output)

    def close(self):
        """Clean up environment resources."""
        pass

    def get_episode_statistics(self) -> dict:
        """Get episode statistics."""
        return {
            'episode_reward': self.episode_reward,
            'symmetric_population': self.physics.get_symmetric_population(),
            'final_populations': self.physics.get_populations().tolist(),
            'steps_completed': self.step_count,
            'steps_per_episode': self.steps_per_episode,
            'episode_completed': self.final_reward_calculated,
            'eps_pol': self.eps_pol  # ← ADDED
        }