"""
DBD RL Environment - Multi-Step Version for Detuning Optimization

Gymnasium environment for optimizing time-dependent detuning Δ(t) in Double Bragg Diffraction.
Uses 30-step episodes with single detuning action per step.
Implements equation D1 cost function for robust beam-splitting optimization.

Supports training with momentum and polarization error sampling for Doppler robustness.
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
    DBD environment implementing equation D1 cost function.

    Action: Time-dependent detuning Δ(t) at discrete control points
    State: [current_time, populations_5_states]
    Reward: Beam-splitter efficiency based on D1 cost function
    Termination: After full pulse duration
    
    Cost (D1) = |0.5 - P_+2| + |0.5 - P_-2| + |P_+2 - P_-2|
    Reward = 1 - Cost (beam-splitter efficiency)
    """

    def __init__(self,
                 max_detuning: float = 20.0,
                 steps_per_episode: int = 30,
                 pulse_duration: float = 1.0,
                 momentum_range: tuple = (-0.3, 0.3),
                 eps_pol_range: tuple = (-0.05, 0.05),
                 sample_conditions: bool = True):
        """
        Initialize DBD environment.
        
        Args:
            max_detuning: Maximum detuning in ω_rec units
            steps_per_episode: Number of discrete time steps (30)
            pulse_duration: Total pulse duration (dimensionless time)
            momentum_range: (min, max) initial momentum in ℏk_L units
            eps_pol_range: (min, max) polarization error
            sample_conditions: If True, sample (p, eps_pol) each episode
        """
        super().__init__()

        # Environment parameters
        self.max_detuning = max_detuning
        self.steps_per_episode = steps_per_episode
        self.pulse_duration = pulse_duration
        self.step_duration = pulse_duration / steps_per_episode
        
        # Ensemble parameters for robustness training
        self.momentum_range = momentum_range
        self.eps_pol_range = eps_pol_range
        self.sample_conditions = sample_conditions
        
        # Current episode conditions (will be sampled in reset)
        self.eps_pol = 0.0
        self.initial_momentum = 0.0

        # Initialize physics engine (will be reset with sampled conditions)
        self.physics = DBDPhysics(
            pulse_duration=self.pulse_duration,
            polarization_error=self.eps_pol,
            initial_momentum=self.initial_momentum
        )

        # Action space: single detuning value [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space: [time, populations_5_states]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        # Episode state
        self.current_time = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.final_reward_calculated = False
        self.previous_efficiency = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state with new sampled conditions.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (can override sampling)

        Returns:
            observation: Initial observation [time, populations]
            info: Additional information
        """
        super().reset(seed=seed)

        # Sample new conditions from distributions (equation D1 ensemble)
        if self.sample_conditions:
            self.initial_momentum = np.random.uniform(*self.momentum_range)
            self.eps_pol = np.random.uniform(*self.eps_pol_range)
        else:
            # Use fixed values for testing
            self.initial_momentum = 0.0
            self.eps_pol = 0.0
        
        # Override with options if provided
        if options is not None:
            if 'initial_momentum' in options:
                self.initial_momentum = options['initial_momentum']
            if 'eps_pol' in options:
                self.eps_pol = options['eps_pol']

        # Reset physics with new sampled conditions
        self.physics = DBDPhysics(
            pulse_duration=self.pulse_duration,
            polarization_error=self.eps_pol,
            initial_momentum=self.initial_momentum
        )
        self.physics.reset()

        # Reset episode state
        self.current_time = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.final_reward_calculated = False
        self.previous_efficiency = 0.0

        # Get initial observation
        observation = self._get_observation()

        info = {
            'beam_splitter_efficiency': 0.0,
            'cost_term1': 0.5,
            'cost_term2': 0.5,
            'cost_term3': 0.0,
            'total_cost': 1.0,
            'symmetric_population': self.physics.get_symmetric_population(),
            'populations': self.physics.get_populations().tolist(),
            'eps_pol': self.eps_pol,
            'initial_momentum': self.initial_momentum
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step with single detuning value.
        Implements equation D1 cost function for reward.

        Args:
            action: Single detuning value [-1, 1]

        Returns:
            observation: New observation
            reward: Dense reward based on beam-splitter efficiency (1 - cost)
            terminated: True only after 30 steps
            truncated: False
            info: Additional information including all D1 components
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

        # ========== GET POPULATIONS ==========
        
        # Bare momentum populations (what's actually measured)
        bare_pops = evolution_result['final_populations']
        P_0 = float(bare_pops[0])       # |p₀⟩
        P_plus2 = float(bare_pops[1])   # |p₀+2ℏk⟩
        P_minus2 = float(bare_pops[2])  # |p₀-2ℏk⟩
        P_plus4 = float(bare_pops[3])   # |p₀+4ℏk⟩
        P_minus4 = float(bare_pops[4])  # |p₀-4ℏk⟩
        
        # Symmetric/antisymmetric populations (theory)
        P_symmetric = float(abs(self.physics.state[1]) ** 2)      # |1,+⟩
        P_antisymmetric = float(abs(self.physics.state[2]) ** 2)  # |1,-⟩
        
        # ========== COMPUTE EQUATION D1 COST ==========
        
        # Term 1: Deviation of P_+2 from ideal 50%
        cost_term1 = abs(0.5 - P_plus2)
        
        # Term 2: Deviation of P_-2 from ideal 50%
        cost_term2 = abs(0.5 - P_minus2)
        
        # Term 3: Imbalance between the two ports
        cost_term3 = abs(P_plus2 - P_minus2)
        
        # Total cost (equation D1)
        total_cost = cost_term1 + cost_term2 + cost_term3
        
        # Beam-splitter efficiency = 1 - cost
        # This is the quantity we want to maximize
        beam_splitter_efficiency = 1.0 - total_cost
        
        # ========== ADDITIONAL METRICS ==========
        
        # Total transfer to ±2ℏk ports
        total_transfer = P_plus2 + P_minus2
        
        # Balance metric (alternative formulation)
        if total_transfer > 1e-8:
            balance = 1.0 - abs(P_plus2 - P_minus2) / total_transfer
        else:
            balance = 1.0
        
        # Leakage to higher orders
        leakage_to_4hk = P_plus4 + P_minus4
        
        # ========== DENSE REWARD CALCULATION ==========
        
        # Incremental reward: improvement in efficiency
        incremental_reward = beam_splitter_efficiency - self.previous_efficiency
        
        # Final bonus at step 30: add current efficiency
        if self.step_count >= self.steps_per_episode:
            final_bonus = beam_splitter_efficiency
            reward = incremental_reward + final_bonus
            self.final_reward_calculated = True
        else:
            reward = incremental_reward
        
        # Update tracking for next step
        self.previous_efficiency = beam_splitter_efficiency
        self.episode_reward += reward

        # Get observation
        observation = self._get_observation()

        # Check termination
        terminated = (self.step_count >= self.steps_per_episode)
        truncated = False

        # ========== COMPREHENSIVE INFO DICT ==========
        info = {
            # D1 cost function components (equation D1)
            'cost_term1': cost_term1,              # |0.5 - P_+2|
            'cost_term2': cost_term2,              # |0.5 - P_-2|
            'cost_term3': cost_term3,              # |P_+2 - P_-2|
            'total_cost': total_cost,              # sum of three terms
            'beam_splitter_efficiency': beam_splitter_efficiency,  # 1 - cost
            
            # Bare momentum populations (experimental observables)
            'populations': bare_pops.tolist(),
            'P_0': P_0,
            'P_plus2': P_plus2,
            'P_minus2': P_minus2,
            'P_plus4': P_plus4,
            'P_minus4': P_minus4,
            
            # Symmetric/antisymmetric basis (theory)
            'symmetric_population': P_symmetric,
            'antisymmetric_population': P_antisymmetric,
            
            # Additional performance metrics
            'total_transfer': total_transfer,
            'balance': balance,
            'leakage_to_4hk': leakage_to_4hk,
            
            # Episode tracking
            'evolution_success': evolution_result['success'],
            'step_count': self.step_count,
            'current_time': self.current_time,
            'episode_reward': self.episode_reward,
            
            # Experimental conditions (sampled from ensemble)
            'eps_pol': self.eps_pol,
            'initial_momentum': self.initial_momentum
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
        """Text-based rendering with D1 cost breakdown."""
        populations = self.physics.get_populations()
        
        # Compute current metrics
        P_plus2 = float(populations[1])
        P_minus2 = float(populations[2])
        
        cost_term1 = abs(0.5 - P_plus2)
        cost_term2 = abs(0.5 - P_minus2)
        cost_term3 = abs(P_plus2 - P_minus2)
        total_cost = cost_term1 + cost_term2 + cost_term3
        efficiency = 1.0 - total_cost
        
        symmetric_pop = self.physics.get_symmetric_population()

        output = []
        output.append(f"DBD Environment (Step {self.step_count}/{self.steps_per_episode})")
        output.append("=" * 60)
        output.append(f"Experimental Conditions:")
        output.append(f"  Polarization Error (ε_pol): {self.eps_pol:.4f}")
        output.append(f"  Initial Momentum (p):       {self.initial_momentum:.4f} ℏkL")
        output.append(f"  Time: {self.current_time:.3f} / {self.pulse_duration:.3f} ({self.current_time/self.pulse_duration*100:.1f}%)")
        output.append("")
        output.append(f"Populations (Bare Momentum Basis):")
        output.append(f"  |0⟩:      {populations[0]:.4f} ({populations[0]*100:.1f}%)")
        output.append(f"  |+2ℏkL⟩:  {P_plus2:.4f} ({P_plus2*100:.1f}%)")
        output.append(f"  |-2ℏkL⟩:  {P_minus2:.4f} ({P_minus2*100:.1f}%)")
        output.append(f"  |+4ℏkL⟩:  {populations[3]:.4f} ({populations[3]*100:.1f}%)")
        output.append(f"  |-4ℏkL⟩:  {populations[4]:.4f} ({populations[4]*100:.1f}%)")
        output.append("")
        output.append(f"D1 Cost Function Components:")
        output.append(f"  Term 1 |0.5 - P_+2|:   {cost_term1:.4f}")
        output.append(f"  Term 2 |0.5 - P_-2|:   {cost_term2:.4f}")
        output.append(f"  Term 3 |P_+2 - P_-2|:  {cost_term3:.4f}")
        output.append(f"  Total Cost:            {total_cost:.4f}")
        output.append(f"  Beam-Splitter Efficiency (1-Cost): {efficiency:.4f} ({efficiency*100:.1f}%)")
        output.append("")
        output.append(f"Theoretical Metrics:")
        output.append(f"  Symmetric Population:  {symmetric_pop:.4f}")
        
        if self.final_reward_calculated:
            output.append("")
            output.append(f"Episode Reward (Total): {self.episode_reward:.4f}")
        
        output.append("=" * 60)

        return "\n".join(output)

    def close(self):
        """Clean up environment resources."""
        pass

    def get_episode_statistics(self) -> dict:
        """Get comprehensive episode statistics."""
        populations = self.physics.get_populations()
        P_plus2 = float(populations[1])
        P_minus2 = float(populations[2])
        
        # Compute final D1 cost
        cost_term1 = abs(0.5 - P_plus2)
        cost_term2 = abs(0.5 - P_minus2)
        cost_term3 = abs(P_plus2 - P_minus2)
        total_cost = cost_term1 + cost_term2 + cost_term3
        
        return {
            'episode_reward': self.episode_reward,
            'beam_splitter_efficiency': 1.0 - total_cost,
            'total_cost': total_cost,
            'cost_term1': cost_term1,
            'cost_term2': cost_term2,
            'cost_term3': cost_term3,
            'symmetric_population': self.physics.get_symmetric_population(),
            'antisymmetric_population': float(abs(self.physics.state[2]) ** 2),
            'final_populations': populations.tolist(),
            'P_plus2': P_plus2,
            'P_minus2': P_minus2,
            'total_transfer': P_plus2 + P_minus2,
            'steps_completed': self.step_count,
            'steps_per_episode': self.steps_per_episode,
            'episode_completed': self.final_reward_calculated,
            'eps_pol': self.eps_pol,
            'initial_momentum': self.initial_momentum
        }