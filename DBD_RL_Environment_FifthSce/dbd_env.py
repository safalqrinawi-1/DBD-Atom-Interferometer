#!/usr/bin/env python3
"""
DBD RL Environment - SMALL ITERATIONS for Quick Verification

Based on dbd_physics.py - optimized for fast training verification.
Optimizations:
- Steps: 15 per episode (reduced for speed)
- Momentum grid: 21 points (reduced from 121)
- Random polarization errors: [0, 0.2]
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dbd_physics import DBDPhysics, MomentumDist, DBDParams


class DBDEnv(gym.Env):
    """DBD RL Environment - Optimized for quick training verification."""
    
    def __init__(self,
                 max_detuning: float = 20.0,
                 steps_per_episode: int = 15,  # REDUCED for speed
                 pulse_duration: float = 2.0,
                 momentum_p0: float = 0.1,
                 momentum_sigma: float = 0.05):
        
        super().__init__()
        
        # Environment parameters
        self.max_detuning = max_detuning
        self.steps_per_episode = steps_per_episode
        self.pulse_duration = pulse_duration
        self.momentum_p0 = momentum_p0
        self.momentum_sigma = momentum_sigma
        
        # SMALL ITERATIONS: Reduced momentum grid (from 121 to 21)
        self.mdist = MomentumDist(
            p0=momentum_p0,
            sigma_p=momentum_sigma,
            p_grid_pts=21  # Reduced for ~6x speedup
        )
        
        # Initialize DBD parameters from dbd_physics.py
        self.params = DBDParams(
            omega_rec=1.0,
            Omega_R=2.0,
            tau_g=0.47,
            t_center=0.5 * pulse_duration,
            eps_pol=0.1  # Default, will be randomized in reset
        )
        
        # Initialize physics engine
        self.physics = DBDPhysics(
            pulse_duration=pulse_duration,
            polarization_error=0.1,
            mdist=self.mdist,
            params=self.params
        )
        
        # Action space: single detuning value
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: [time, pop_0, pop_+2, pop_-2, pop_+4, pop_-4]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.previous_beam_score = 0.0
        self.episode_reward = 0.0
        
        # Scenario info
        self.scenario_name = "small_iterations_test"
        self.scenario_description = f"SMALL_ITER: p0={momentum_p0}, σp={momentum_sigma}ℏkL, 21pts, 15steps"
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        # Randomize polarization error (0-20%)
        polarization_error = self.np_random.uniform(0.0, 0.2)
        
        # Reset physics with randomized polarization error
        self.physics.reset(polarization_error=polarization_error)
        
        # Reset tracking
        self.current_step = 0
        self.previous_beam_score = 0.0
        self.episode_reward = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'episode_reward': 0.0,
            'populations': self.physics.get_populations(),
            'beam_splitting_score': 0.0,
            'polarization_error': polarization_error,
            'momentum_p0': self.momentum_p0,
            'momentum_sigma': self.momentum_sigma,
            'scenario': self.scenario_name
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        # Convert action to detuning
        detuning = float(action[0] * self.max_detuning)
        
        # Time segment for this step
        time_segment = self.pulse_duration / self.steps_per_episode
        
        # Create constant detuning function for this step
        def detuning_function(t):
            return detuning
        
        # Evolve physics (uses dbd_physics.py evolve method)
        populations = self.physics.evolve(detuning_function, time_segment)
        
        # Calculate beam score
        current_beam_score = self.physics.get_beam_splitting_score()
        
        # Reward: scaled for better learning
        improvement = (current_beam_score - self.previous_beam_score) * 10.0
        absolute = current_beam_score ** 2
        
        reward = 0.5 * improvement + 0.5 * absolute
        
        # Final step bonus
        if self.current_step == self.steps_per_episode - 1:
            reward += 3.0 * current_beam_score
        
        # Update tracking
        self.previous_beam_score = current_beam_score
        self.episode_reward += reward
        self.current_step += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Check termination
        terminated = self.current_step >= self.steps_per_episode
        truncated = False
        
        # Info
        info = {
            'episode_reward': self.episode_reward,
            'populations': populations,
            'beam_splitting_score': current_beam_score,
            'symmetric_population': self.physics.get_symmetric_population(),
            'polarization_error': self.physics.polarization_error,
            'momentum_p0': self.momentum_p0,
            'momentum_sigma': self.momentum_sigma,
            'scenario': self.scenario_name,
            'step': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        normalized_time = self.current_step / self.steps_per_episode
        populations = self.physics.get_populations()
        
        observation = np.array([
            normalized_time,
            populations[0],  # |0⟩
            populations[1],  # |+2ℏkL⟩
            populations[2],  # |-2ℏkL⟩
            populations[3],  # |+4ℏkL⟩
            populations[4]   # |-4ℏkL⟩
        ], dtype=np.float32)
        
        return observation
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment."""
        if mode == 'human':
            populations = self.physics.get_populations()
            beam_score = self.physics.get_beam_splitting_score()
            
            print(f"Step: {self.current_step}/{self.steps_per_episode}")
            print(f"Populations: {populations}")
            print(f"Beam Score: {beam_score:.4f}")
            print(f"Pol Error: {self.physics.polarization_error:.1%}")
            print("-" * 40)
        
        return None
    
    def close(self):
        """Close environment."""
        pass
    
    def get_physics_info(self) -> Dict[str, Any]:
        """Get physics information."""
        return self.physics.get_physics_info()
    
    def get_scenario_info(self) -> Dict[str, Any]:
        """Get scenario information."""
        return {
            'scenario_name': self.scenario_name,
            'scenario_description': self.scenario_description,
            'momentum_p0': self.momentum_p0,
            'momentum_sigma': self.momentum_sigma,
            'momentum_grid_pts': self.mdist.p_grid_pts,
            'steps_per_episode': self.steps_per_episode,
            'polarization_error_range': (0.0, 0.2),
            'target': '50/50 beam split'
        }


# Quick test
if __name__ == "__main__":
    print("Testing DBD Environment - Small Iterations")
    print("=" * 60)
    
    env = DBDEnv()
    print(f"Configuration:")
    print(f"  Steps per episode: {env.steps_per_episode}")
    print(f"  Momentum grid: {env.mdist.p_grid_pts} points")
    print(f"  Polarization: random [0, 20%]")
    print(f"  Physics params: omega_rec={env.params.omega_rec}, Omega_R={env.params.Omega_R}")
    print("=" * 60)
    
    obs, info = env.reset()
    print(f"\nInitial obs shape: {obs.shape}")
    print(f"Running test episode...")
    
    total_reward = 0.0
    for step in range(env.steps_per_episode):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 5 == 0:
            print(f"  Step {step}: Beam Score = {info['beam_splitting_score']:.4f}")
    
    print(f"\nEpisode completed!")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final beam score: {info['beam_splitting_score']:.4f}")
    print(f"Final populations: {info['populations']}")