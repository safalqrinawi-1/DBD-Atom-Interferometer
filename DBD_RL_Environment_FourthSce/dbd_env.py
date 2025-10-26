#!/usr/bin/env python3
"""
DBD RL Environment - Enhanced Scenario (Small Iterations)

Gymnasium environment for DBD (Double Bragg Diffraction) optimization.
Enhanced scenario: Fixed 30% polarization error with momentum distribution.
Optimized for quick training verification.

Configuration:
- Center of mass at zero: p0 = 0.0
- Momentum width: σp = 0.05 ℏkL (fixed)
- Polarization error: 30% (fixed)
- Beam splitting efficiency optimization (50/50 split target)
- Ensemble averaging over momentum distribution
- Single detuning action per step
- 15 steps per episode (reduced for faster training)
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dbd_physics import DBDPhysics, MomentumDist, DBDParams


class DBDEnv(gym.Env):
    """
    DBD RL Environment with fixed momentum distribution and polarization error.
    
    Configuration:
    - p0 = 0.0 (COM at zero)
    - σp = 0.05 ℏkL (momentum width)
    - Polarization error = 30% (fixed)
    - Target: 50/50 split between |+2ℏkL⟩ and |-2ℏkL⟩
    """
    
    def __init__(self,
                 max_detuning: float = 20.0,  # In units of ω_rec
                 steps_per_episode: int = 15,  # Reduced from 30 for faster training
                 pulse_duration: float = 1.0,  # In units of 1/ω_rec (dimensionless)
                 polarization_error: float = 0.3,  # 30% fixed
                 momentum_p0: float = 0.0,  # COM at zero
                 momentum_sigma: float = 0.05):  # σp = 0.05 ℏkL
        
        super().__init__()
        
        # Environment parameters
        self.max_detuning = max_detuning
        self.steps_per_episode = steps_per_episode
        self.pulse_duration = pulse_duration
        self.polarization_error = polarization_error
        self.momentum_p0 = momentum_p0
        self.momentum_sigma = momentum_sigma
        
        # Create momentum distribution (reduced grid points for speed)
        self.mdist = MomentumDist(
            p0=momentum_p0,
            sigma_p=momentum_sigma,
            p_grid_pts=51  # Reduced from 121 for faster computation
        )
        
        # Initialize DBD parameters from dbd_physics.py
        self.params = DBDParams(
            omega_rec=1.0,
            Omega_R=2.0,
            tau_g=0.47,
            t_center=0.5 * pulse_duration,
            eps_pol=polarization_error
        )
        
        # Initialize physics engine
        self.physics = DBDPhysics(
            pulse_duration=pulse_duration,
            polarization_error=polarization_error,
            mdist=self.mdist,
            params=self.params
        )
        
        # Action space: single detuning value per step
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: [normalized_time, pop_0, pop_+2, pop_-2, pop_+4, pop_-4]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.previous_beam_score = 0.0
        self.episode_reward = 0.0
        
        # Scenario information
        self.scenario_name = "enhanced_scenario_small"
        self.scenario_description = f"p0={momentum_p0}, σp={momentum_sigma}ℏkL, ε_pol={polarization_error*100:.0f}%"
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Reset physics (polarization error is fixed)
        self.physics.reset(polarization_error=self.polarization_error)
        
        # Reset episode tracking
        self.current_step = 0
        self.previous_beam_score = 0.0
        self.episode_reward = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        
        # Episode info
        info = {
            'episode_reward': 0.0,
            'populations': self.physics.get_populations(),
            'beam_splitting_score': 0.0,
            'polarization_error': self.polarization_error,
            'momentum_p0': self.momentum_p0,
            'momentum_sigma': self.momentum_sigma,
            'scenario': self.scenario_name
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Detuning action [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to detuning (in ω_rec units)
        detuning = float(action[0] * self.max_detuning)
        
        # Calculate time segment for this step
        time_segment = self.pulse_duration / self.steps_per_episode
        
        # Create constant detuning function for this step
        def detuning_function(t):
            return detuning
        
        # Evolve physics (returns ensemble-averaged populations)
        populations = self.physics.evolve(detuning_function, time_segment)
        
        # Calculate beam splitting score
        current_beam_score = self.physics.get_beam_splitting_score()
        
        # Calculate reward: improvement in beam splitting score (scaled for better learning)
        reward = (current_beam_score - self.previous_beam_score) * 10.0
        
        # Add final step bonus
        if self.current_step == self.steps_per_episode - 1:
            reward += current_beam_score * 5.0  # Final beam splitting score bonus
        
        # Update tracking
        self.previous_beam_score = current_beam_score
        self.episode_reward += reward
        self.current_step += 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Check termination
        terminated = self.current_step >= self.steps_per_episode
        truncated = False
        
        # Episode info
        info = {
            'episode_reward': self.episode_reward,
            'populations': populations,
            'beam_splitting_score': current_beam_score,
            'symmetric_population': self.physics.get_symmetric_population(),
            'polarization_error': self.polarization_error,
            'momentum_p0': self.momentum_p0,
            'momentum_sigma': self.momentum_sigma,
            'scenario': self.scenario_name,
            'step': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Observation vector: [normalized_time, pop_0, pop_+2, pop_-2, pop_+4, pop_-4]
        """
        # Normalized time
        normalized_time = self.current_step / self.steps_per_episode
        
        # Ensemble-averaged population distribution
        populations = self.physics.get_populations()
        
        # Combine into observation vector
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
        """
        Render the environment.
        
        Args:
            mode: Render mode
            
        Returns:
            Rendered frame if applicable
        """
        if mode == 'human':
            populations = self.physics.get_populations()
            beam_score = self.physics.get_beam_splitting_score()
            
            print(f"Step: {self.current_step}/{self.steps_per_episode}")
            print(f"Populations: {populations}")
            print(f"Beam Splitting Score: {beam_score:.4f}")
            print(f"Polarization Error: {self.polarization_error:.1%}")
            print(f"Momentum: p0={self.momentum_p0}, σp={self.momentum_sigma}")
            print("-" * 40)
        
        return None
    
    def close(self):
        """Close the environment."""
        pass
    
    def get_physics_info(self) -> Dict[str, Any]:
        """Get detailed physics information."""
        return self.physics.get_physics_info()
    
    def get_scenario_info(self) -> Dict[str, Any]:
        """Get scenario-specific information."""
        return {
            'scenario_name': self.scenario_name,
            'scenario_description': self.scenario_description,
            'momentum_p0': self.momentum_p0,
            'momentum_sigma': self.momentum_sigma,
            'polarization_error': self.polarization_error,
            'target_metric': 'beam_splitting_efficiency',
            'target_value': '50/50 split between |+2ℏkL⟩ and |-2ℏkL⟩'
        }


# For testing
if __name__ == "__main__":
    env = DBDEnv()
    print("DBD Environment Configuration (Small Iterations):")
    print("=" * 50)
    print(f"Polarization Error: {env.polarization_error:.1%}")
    print(f"Momentum p0: {env.momentum_p0} ℏkL")
    print(f"Momentum σp: {env.momentum_sigma} ℏkL")
    print(f"Steps per episode: {env.steps_per_episode}")
    print(f"Max detuning: {env.max_detuning} ω_rec")
    print(f"Pulse duration: {env.pulse_duration:.2f} (1/ω_rec)")
    print(f"Momentum grid points: {env.mdist.p_grid_pts}")
    print("=" * 50)
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial populations: {info['populations']}")
    print(f"Scenario: {info['scenario']}")
    
    # Quick test episode
    print("\nRunning quick test episode...")
    total_reward = 0.0
    for step in range(env.steps_per_episode):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 5 == 0:
            print(f"  Step {step}: Beam Score = {info['beam_splitting_score']:.4f}")
    
    print(f"\nTest episode completed!")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final beam score: {info['beam_splitting_score']:.4f}")
    print(f"Final populations: {info['populations']}")