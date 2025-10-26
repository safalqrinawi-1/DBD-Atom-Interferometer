from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


# ---------- Parameters ----------

@dataclass
class DBDParams:
    omega_rec: float = 1.0      # recoil frequency (dimensionless units)
    Omega_R: float = 2.0        # peak Rabi frequency (in ω_rec units)
    tau_g: float = 0.47         # Gaussian width τ (1/ω_rec)
    t_center: float = 0.5       # pulse center t0 (1/ω_rec)
    eps_pol: float = 0.3        # polarization error term (30%)

# Momentum-distribution parameters (in units of ℏ k_L)
@dataclass
class MomentumDist:
    p0: float = 0.0          # mean momentum ⟨p⟩/(ℏ k_L) - COM at zero
    sigma_p: float = 0.05    # width σ_p/(ℏ k_L) - as requested
    p_grid_pts: int = 121    # number of grid points over ~±6σ


# ---------- Gaussian pulse envelope Ω(t) (Eq. 21) ----------

def gaussian_Omega(t: float, pars: DBDParams) -> float:
    """Ω(t) = Ω_R * exp( - (t - t0)^2 / (2 τ^2) )"""
    return pars.Omega_R * np.exp(-((t - pars.t_center) ** 2) / (2.0 * pars.tau_g ** 2))


# ---------- Hamiltonian (Equation 16) ----------

def H5_matrix(t: float, p: float, Delta_t: float, pars: DBDParams) -> np.ndarray:
    """
    Five-level Double Bragg Hamiltonian (Eq. 16) in basis:
      |p>, |1,+>, |1,->, |2,+>, |2,->
    p in ℏ k_L, energies in ℏ ω_rec, time in 1/ω_rec.
    """
    w = pars.omega_rec
    Om = gaussian_Omega(t, pars)

    # C(t, ε_pol) = cos[(4 ω_rec + Δ(t)) t] + ε_pol
    C = np.cos((4.0 * w + Delta_t) * t) + pars.eps_pol

    H = np.zeros((5, 5), dtype=np.complex128)

    # Diagonals (kinetic + Bragg zone energies)
    kin_base = (p ** 2) / 2.0
    H[0, 0] = kin_base
    H[1, 1] = kin_base + 4.0 * w
    H[2, 2] = kin_base + 4.0 * w
    H[3, 3] = kin_base + 16.0 * w
    H[4, 4] = kin_base + 16.0 * w

    # Couplings
    H[0, 1] = np.sqrt(2.0) * Om * C
    H[1, 0] = H[0, 1]

    doppler_12 = 4.0 * p * w
    H[1, 2] = doppler_12
    H[2, 1] = doppler_12

    H[1, 3] = Om * C
    H[3, 1] = H[1, 3]

    H[2, 4] = Om * C
    H[4, 2] = H[2, 4]

    doppler_45 = 8.0 * p * w
    H[3, 4] = doppler_45
    H[4, 3] = doppler_45

    return H


# ---------- Time evolution for a single p ----------

def step_schrodinger(c0, t0, t1, p, detuning_function, pars: DBDParams,
                     rtol=1e-9, atol=1e-11):
    """Evolve from t0→t1 with fully time-dependent Δ(t)."""
    def rhs(t, y):
        Delta_t = float(detuning_function(t))
        return -1j * (H5_matrix(t, p, Delta_t, pars) @ y)

    sol = solve_ivp(rhs, (t0, t1), c0, t_eval=[t1],
                    method='DOP853', rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, 0]


# ---------- Momentum distribution utilities ----------

def psi_p(p: np.ndarray, mdist: MomentumDist) -> np.ndarray:
    """
    Normalized Gaussian wavefunction in momentum space:
      psi(p) = (2πσ_p^2)^(-1/4) * exp(-(p-p0)^2 / (4σ_p^2))
    Returns complex amplitudes ψ(p). Units of p, p0, σ_p are ℏ k_L.
    """
    s2 = mdist.sigma_p ** 2
    norm = (1.0 / (2.0 * np.pi * s2)) ** 0.25
    return norm * np.exp(-((p - mdist.p0) ** 2) / (4.0 * s2))


def gaussian_p_grid(mdist: MomentumDist):
    """Uniform grid over [p0-6σ, p0+6σ] with mdist.p_grid_pts points."""
    lo = mdist.p0 - 6.0 * mdist.sigma_p
    hi = mdist.p0 + 6.0 * mdist.sigma_p
    p_grid = np.linspace(lo, hi, mdist.p_grid_pts)
    dp = p_grid[1] - p_grid[0] if len(p_grid) > 1 else 1.0
    return p_grid, dp


# ---------- Core DBD engine (ensemble-aware) ----------

class DBDPhysics:
    """
    DBD Physics engine with momentum ensemble support.
    
    Parameters configured for:
    - Initial momentum: p0 = 0.0 (COM at zero)
    - Momentum width: σp = 0.05 ℏkL
    - Polarization error: 30%
    """
    
    def __init__(self, 
                 pulse_duration=1.0, 
                 dt=1e-3,
                 polarization_error=0.3,  # 30% polarization error
                 mdist: MomentumDist = None,
                 params: DBDParams = None):
        
        self.pulse_duration = float(pulse_duration)
        self.dt = float(dt)
        self.polarization_error = float(polarization_error)
        
        # Initialize momentum distribution parameters
        if mdist is None:
            self.mdist = MomentumDist(p0=0.0, sigma_p=0.05)  # COM at zero, σp=0.05
        else:
            self.mdist = mdist
        
        # Initialize DBD parameters with polarization error
        self.pars = params if params is not None else DBDParams(
            t_center=0.5 * pulse_duration,
            eps_pol=polarization_error
        )
        
        # Add omega_rec attribute for RL environment
        self.omega_rec = self.pars.omega_rec
        
        # Build momentum grid and weights (FIXED NORMALIZATION)
        self.p_grid, self.dp = gaussian_p_grid(self.mdist)
        self.psi_vals = psi_p(self.p_grid, self.mdist)
        self.weights = np.abs(self.psi_vals) ** 2 * self.dp  # Include dp in weights
        Z = np.sum(self.weights)  # Now sum without dp
        self.weights /= Z  # Normalize so Σ weights = 1
        
        # Initialize state ensemble (one state per momentum value)
        self.num_p_samples = len(self.p_grid)
        self.states = np.zeros((self.num_p_samples, 5), dtype=np.complex128)
        self.states[:, 0] = 1.0  # Each state normalized: starts in |p⟩
        
        self.current_time = 0.0
    
    def reset(self, polarization_error=None):
        """Reset to initial state."""
        if polarization_error is not None:
            self.polarization_error = float(polarization_error)
            self.pars.eps_pol = self.polarization_error
        
        # Reset all momentum states (each normalized to 1)
        self.states[:] = 0.0
        self.states[:, 0] = 1.0  # Each state in |p⟩
        self.current_time = 0.0
        self.pars.t_center = 0.5 * self.pulse_duration
    
    def evolve(self, detuning_function, step_duration: float):
        """
        Evolve ensemble for one time step with time-dependent detuning.
        
        Args:
            detuning_function: Callable that takes time t and returns Δ(t)
            step_duration: Duration of this step
            
        Returns:
            Ensemble-averaged populations
        """
        # Evolve each momentum component with the full time-dependent Δ(t)
        for i, p in enumerate(self.p_grid):
            self.states[i] = step_schrodinger(
                self.states[i], 
                self.current_time,
                self.current_time + step_duration,
                p, 
                detuning_function,  # ← Pass function directly
                self.pars
            )
        
        self.current_time += step_duration
        
        return self.get_populations()
    
    def get_populations(self) -> np.ndarray:
        """
        Get ensemble-averaged populations in bare momentum basis.
        
        Returns:
            Array [P(|p⟩), P(|p+2ℏk⟩), P(|p-2ℏk⟩), P(|p+4ℏk⟩), P(|p-4ℏk⟩)]
        """
        populations = np.zeros(5, dtype=np.float32)
        
        for i, wgt in enumerate(self.weights):
            c_p, c1p, c1m, c2p, c2m = self.states[i]
            
            # Convert to bare momentum basis
            c_plus2 = (c1p + c1m) / np.sqrt(2.0)
            c_minus2 = (c1p - c1m) / np.sqrt(2.0)
            c_plus4 = (c2p + c2m) / np.sqrt(2.0)
            c_minus4 = (c2p - c2m) / np.sqrt(2.0)
            
            # Accumulate weighted populations
            populations[0] += wgt * abs(c_p) ** 2      # |p⟩
            populations[1] += wgt * abs(c_plus2) ** 2  # |p+2ℏk⟩
            populations[2] += wgt * abs(c_minus2) ** 2 # |p-2ℏk⟩
            populations[3] += wgt * abs(c_plus4) ** 2  # |p+4ℏk⟩
            populations[4] += wgt * abs(c_minus4) ** 2 # |p-4ℏk⟩
        
        return populations
    
    def get_symmetric_population(self) -> float:
        """
        Get ensemble-averaged symmetric state population |1,+⟩.
        
        Returns:
            Averaged population in symmetric ±2ℏk state
        """
        sym_pop = 0.0
        for i, wgt in enumerate(self.weights):
            sym_pop += wgt * abs(self.states[i, 1]) ** 2
        return float(sym_pop)
    
    def get_beam_splitting_score(self) -> float:
        """
        Calculate beam-splitter efficiency score.
        Measures how close the split is to ideal 50/50 between ±2ℏk ports.
        
        Returns:
            Score in [0, 1] where 1 = perfect 50/50 split
        """
        pops = self.get_populations()
        p_plus2 = pops[1]   # |p+2ℏk⟩
        p_minus2 = pops[2]  # |p-2ℏk⟩
        
        total_split = p_plus2 + p_minus2
        
        if total_split < 1e-10:
            return 0.0
        
        # Ideal is 0.5 each, so difference from 0.5 shows imbalance
        imbalance = abs(p_plus2 - 0.5 * total_split)
        
        # Score: 1 - normalized imbalance
        score = total_split * (1.0 - 2.0 * imbalance / total_split)
        
        return float(np.clip(score, 0.0, 1.0))
    
    def get_physics_info(self) -> dict:
        """Get detailed physics information."""
        return {
            'pulse_duration': self.pulse_duration,
            'dt': self.dt,
            'current_time': self.current_time,
            'polarization_error': self.polarization_error,
            'omega_rec': self.omega_rec,
            'momentum_distribution': {
                'p0': self.mdist.p0,
                'sigma_p': self.mdist.sigma_p,
                'p_grid_pts': self.mdist.p_grid_pts,
                'p_range': [float(self.p_grid[0]), float(self.p_grid[-1])]
            }
        }


# ---------- Standalone ensemble runner ----------

def run_ensemble_gaussian(detuning_function,
                          physics_template: DBDPhysics,
                          mdist: MomentumDist):
    """
    Evolve the distribution ψ(p) under the DBD dynamics and
    return momentum-averaged output populations.
    """
    # Build grid and weights (FIXED NORMALIZATION)
    p_grid, dp = gaussian_p_grid(mdist)
    psi_vals = psi_p(p_grid, mdist)
    weights = np.abs(psi_vals) ** 2 * dp  # Include dp in weights
    Z = np.sum(weights)
    weights /= Z  # Normalize so Σ weights = 1

    # Storage for weighted populations
    keys = ["|p⟩", "|p+2ℏk⟩", "|p-2ℏk⟩", "|p+4ℏk⟩", "|p-4ℏk⟩"]
    accum = {k: 0.0 for k in keys}

    # Evolve each p independently and accumulate
    for p, wgt in zip(p_grid, weights):
        eng = DBDPhysics(
            pulse_duration=physics_template.pulse_duration,
            dt=physics_template.dt,
            polarization_error=physics_template.pars.eps_pol,
            mdist=mdist,
            params=DBDParams(**vars(physics_template.pars))
        )
        eng.reset()
        eng.pars.t_center = 0.5 * eng.pulse_duration
        nsteps = max(1, int(round(eng.pulse_duration / eng.dt)))
        step = eng.pulse_duration / nsteps
        
        for _ in range(nsteps):
            state = eng.states[0]  # Single momentum case
            state = step_schrodinger(
                state, eng.current_time,
                eng.current_time + step,
                p, detuning_function, eng.pars
            )
            eng.states[0] = state
            eng.current_time += step
        
        # Get populations for this p
        c_p, c1p, c1m, c2p, c2m = eng.states[0]
        c_plus2 = (c1p + c1m) / np.sqrt(2.0)
        c_minus2 = (c1p - c1m) / np.sqrt(2.0)
        c_plus4 = (c2p + c2m) / np.sqrt(2.0)
        c_minus4 = (c2p - c2m) / np.sqrt(2.0)
        
        pops = {
            "|p⟩": abs(c_p) ** 2,
            "|p+2ℏk⟩": abs(c_plus2) ** 2,
            "|p-2ℏk⟩": abs(c_minus2) ** 2,
            "|p+4ℏk⟩": abs(c_plus4) ** 2,
            "|p-4ℏk⟩": abs(c_minus4) ** 2,
        }
        
        for k in keys:
            accum[k] += wgt * pops[k]

    return {
        "averaged_bare_populations": accum,
        "p_grid": p_grid,
        "weights": weights,
        "norm_check": float(np.sum(weights))  # Should be 1.0
    }