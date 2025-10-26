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
    eps_pol: float = 0.0        # polarization error term

# Momentum-distribution parameters (in units of ℏ k_L)
@dataclass
class MomentumDist:
    p0: float = 0.0          # mean momentum ⟨p⟩/(ℏ k_L)
    sigma_p: float = 0.05    # width σ_p/(ℏ k_L)
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


# ---------- Core DBD engine (single-p) ----------

class DBDPhysics:
    def __init__(self, pulse_duration=1.0, dt=1e-3,
                 polarization_error=0.0, initial_momentum=0.0, params: DBDParams = None):
        self.pulse_duration = float(pulse_duration)
        self.dt = float(dt)
        self.initial_momentum = float(initial_momentum)  # ← NEW

        self.pars = params if params is not None else DBDParams(
            t_center=0.5 * pulse_duration,
            eps_pol=polarization_error
        )
        
        # ← NEW: Add omega_rec attribute for RL environment
        self.omega_rec = self.pars.omega_rec
        
        self.state = np.zeros(5, dtype=np.complex128)
        self.state[0] = 1.0  # Start in |p⟩
        self.current_time = 0.0

    def reset(self, polarization_error=None):
        """Reset to initial state."""
        if polarization_error is not None:
            self.pars.eps_pol = float(polarization_error)
        self.state[:] = 0.0
        self.state[0] = 1.0  # Reset to |p⟩
        self.current_time = 0.0
        self.pars.t_center = 0.5 * self.pulse_duration  # ← NEW: Re-center pulse

    def evolve_system(self, detuning_function, step_duration, p=None):
        """Evolve system for one time step."""
        if p is None:
            p = self.initial_momentum
        
        # Single step with full time-dependent Δ(t)
        self.state = step_schrodinger(
            self.state, self.current_time,
            self.current_time + step_duration,
            p, detuning_function, self.pars
        )
        self.current_time += step_duration
        
        # ← NEW: Return dict compatible with RL environment
        return {
            'success': True,
            'time': np.array([self.current_time], dtype=np.float64),
            'states': self.state[None, :],
            'final_populations': self.get_populations(),
            'symmetric_population': self.get_symmetric_population()
        }

    def run_gaussian_pulse(self, detuning_function, p=0.0):
        """Run complete Gaussian pulse (for standalone use)."""
        self.pars.t_center = self.current_time + 0.5 * self.pulse_duration
        nsteps = max(1, int(round(self.pulse_duration / self.dt)))
        step = self.pulse_duration / nsteps
        for _ in range(nsteps):
            self.evolve_system(detuning_function, step, p=p)
        return self.get_results()

    # ========== NEW METHODS FOR RL ENVIRONMENT ==========
    
    def get_populations(self) -> np.ndarray:
        """
        Get populations in bare momentum basis as numpy array.
        Required by RL environment.
        
        Returns:
            Array [P(|p⟩), P(|p+2ℏk⟩), P(|p-2ℏk⟩), P(|p+4ℏk⟩), P(|p-4ℏk⟩)]
        """
        bare_pops = self._bare_momentum_populations()
        return np.array([
            bare_pops["|p⟩"],
            bare_pops["|p+2ℏk⟩"],
            bare_pops["|p-2ℏk⟩"],
            bare_pops["|p+4ℏk⟩"],
            bare_pops["|p-4ℏk⟩"]
        ], dtype=np.float32)
    
    def get_symmetric_population(self) -> float:
        """
        Get symmetric state population |1,+⟩.
        This is the key metric for DBD beam-splitter efficiency.
        Required by RL environment.
        
        Returns:
            Population in symmetric ±2ℏk state
        """
        return float(abs(self.state[1]) ** 2)
    
    def get_beamsplitter_efficiency(self) -> float:
        """
        Get beam-splitter efficiency: P(|p+2ℏk⟩) + P(|p-2ℏk⟩)
        
        Returns:
            Total population in ±2ℏk ports
        """
        bare_pops = self._bare_momentum_populations()
        return float(bare_pops["|p+2ℏk⟩"] + bare_pops["|p-2ℏk⟩"])
    
    # ====================================================

    def _bare_momentum_populations(self):
        """Convert from symmetric/antisymmetric to bare momentum states."""
        c_p, c1p, c1m, c2p, c2m = self.state
        c_plus2  = (c1p + c1m) / np.sqrt(2.0)
        c_minus2 = (c1p - c1m) / np.sqrt(2.0)
        c_plus4  = (c2p + c2m) / np.sqrt(2.0)
        c_minus4 = (c2p - c2m) / np.sqrt(2.0)
        return {
            "|p⟩": abs(c_p) ** 2,
            "|p+2ℏk⟩": abs(c_plus2) ** 2,
            "|p-2ℏk⟩": abs(c_minus2) ** 2,
            "|p+4ℏk⟩": abs(c_plus4) ** 2,
            "|p-4ℏk⟩": abs(c_minus4) ** 2,
        }

    def get_results(self):
        """Get results in dict format (for standalone use)."""
        bare_pops = self._bare_momentum_populations()
        sym_antisym_pops = {
            "|1,+⟩ (±2ℏk, sym)": abs(self.state[1]) ** 2,
            "|1,-⟩ (±2ℏk, anti)": abs(self.state[2]) ** 2,
            "|2,+⟩ (±4ℏk, sym)": abs(self.state[3]) ** 2,
            "|2,-⟩ (±4ℏk, anti)": abs(self.state[4]) ** 2,
        }
        return {
            "time": self.current_time,
            "bare_populations": bare_pops,
            "sym_antisym_populations": sym_antisym_pops
        }


# ---------- Gaussian momentum ensemble runner ----------

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


def run_ensemble_gaussian(detuning_function,
                          physics_template: DBDPhysics,
                          mdist: MomentumDist):
    """
    Evolve the distribution ψ(p) under the DBD dynamics and
    return momentum-averaged output populations.
    """
    # Build grid and weights
    p_grid, dp = gaussian_p_grid(mdist)
    psi_vals = psi_p(p_grid, mdist)
    weights = np.abs(psi_vals) ** 2
    # Normalize numerically over the discrete grid
    Z = np.sum(weights) * dp
    weights /= Z

    # Storage for weighted populations
    keys = ["|p⟩", "|p+2ℏk⟩", "|p-2ℏk⟩", "|p+4ℏk⟩", "|p-4ℏk⟩"]
    accum = {k: 0.0 for k in keys}

    # Evolve each p independently and accumulate
    for p, wgt in zip(p_grid, weights):
        eng = DBDPhysics(pulse_duration=physics_template.pulse_duration,
                         dt=physics_template.dt,
                         polarization_error=physics_template.pars.eps_pol,
                         params=DBDParams(**vars(physics_template.pars)))
        eng.reset()
        eng.pars.t_center = 0.5 * eng.pulse_duration
        nsteps = max(1, int(round(eng.pulse_duration / eng.dt)))
        step = eng.pulse_duration / nsteps
        for _ in range(nsteps):
            eng.evolve_system(detuning_function, step, p=p)
        pops = eng._bare_momentum_populations()
        for k in keys:
            accum[k] += wgt * pops[k]

    return {
        "averaged_bare_populations": accum,
        "p_grid": p_grid,
        "weights": weights,
        "norm_check": float(np.sum(weights) * dp)
    }