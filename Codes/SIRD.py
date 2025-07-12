"""Stochastic (Gillespie) and deterministic SIR epidemic model simulation with mortality.

© 2024 — Iker Marginet <ikergenki@gmail.com>
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional
import argparse
import logging
import math
import csv
from pathlib import Path
import sys

import numpy as np
from numpy.random import Generator, default_rng
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ------------------------------------------------------------------------------
# Parameters and Types
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SIRParams:
    """Epidemiological parameters for the SIR model with mortality.

    Attributes
    ----------
    N : int
        Initial population size.
    beta : float
        Transmission rate (β). Units: 1/time.
    gamma : float
        Recovery rate (γ). Units: 1/time.
    mu : float
        Death rate (μ). Units: 1/time.
    """
    N: int
    beta: float
    gamma: float
    mu: float = 0.0

    def __post_init__(self):
        """Validate parameters."""
        if self.N <= 0:
            raise ValueError("Population size N must be positive")
        if self.beta < 0 or self.gamma < 0 or self.mu < 0:
            raise ValueError("Rates β, γ, and μ must be non-negative")

    def R0(self) -> float:
        """Basic reproduction number R₀ = β / (γ + μ)."""
        return self.beta / (self.gamma + self.mu)

    def herd_immunity_threshold(self) -> float:
        """Herd immunity threshold (1 – 1/R₀)."""
        r0 = self.R0()
        return 1.0 - (1.0 / r0) if r0 > 1.0 else float("nan")

@dataclass(slots=True)
class Trajectory:
    """Container for a temporal trajectory (stochastic or deterministic)."""
    t: np.ndarray  # (n,)
    S: np.ndarray  # (n,)
    I: np.ndarray  # (n,)
    R: np.ndarray  # (n,)
    D: np.ndarray  # (n,) Deaths

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {"time": self.t, "S": self.S, "I": self.I, "R": self.R, "D": self.D}

    def peak_infections(self) -> Tuple[float, float]:
        """Return (t_peak, I_max)."""
        idx = int(np.argmax(self.I))
        return float(self.t[idx]), float(self.I[idx])

    def final_size(self) -> float:
        """Cumulative infected/recovered fraction at t → ∞."""
        initial_pop = self.S[0] + self.I[0] + self.R[0] + self.D[0]
        return float(self.R[-1]) / float(initial_pop) if initial_pop > 0 else 0.0

    def epidemic_duration(self) -> float:
        """Time until I(t) < 1 (stochastic) or I(t) < 0.01 (deterministic)."""
        threshold = 1.0 if np.all(self.I % 1 == 0) else 0.01
        idx = np.where(self.I < threshold)[0]
        return float(self.t[idx[0]]) if idx.size > 0 else float(self.t[-1])

    def save_to_csv(self, filename: str | Path) -> None:
        """Save trajectory to CSV file with consistent float formatting."""
        filename = Path(filename)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "S", "I", "R", "D"])
            for t, s, i, r, d in zip(self.t, self.S, self.I, self.R, self.D):
                formatted_row = [
                    f"{t:.6f}" if isinstance(t, float) else t,
                    f"{s:.6f}" if isinstance(s, float) else s,
                    f"{i:.6f}" if isinstance(i, float) else i,
                    f"{r:.6f}" if isinstance(r, float) else r,
                    f"{d:.6f}" if isinstance(d, float) else d
                ]
                writer.writerow(formatted_row)
        logger.info("Trajectory saved to %s", filename)

# ------------------------------------------------------------------------------
# Stochastic Simulation (Gillespie SSA)
# ------------------------------------------------------------------------------

def gillespie_sir(params: SIRParams, S0: int, I0: int, R0: int, D0: int, t_max: float, *, rng: Generator | None = None) -> Trajectory:
    """Stochastic SIR simulation with mortality using the Gillespie algorithm.

    Parameters
    ----------
    params : SIRParams
        Model parameters (N, β, γ, μ).
    S0, I0, R0, D0 : int
        Initial conditions (S + I + R + D = N).
    t_max : float
        Maximum simulation time.
    rng : np.random.Generator, optional
        Random number generator. If None, a non-deterministic RNG is created.

    Returns
    -------
    Trajectory
        Temporal trajectory (time, S, I, R, D) as numpy arrays.
    """
    if S0 + I0 + R0 + D0 != params.N:
        raise ValueError(f"S0 ({S0}) + I0 ({I0}) + R0 ({R0}) + D0 ({D0}) must equal N ({params.N})")
    if S0 < 0 or I0 < 0 or R0 < 0 or D0 < 0:
        raise ValueError("Initial conditions must be non-negative")
    if t_max <= 0:
        raise ValueError("t_max must be positive")

    rng = default_rng() if rng is None else rng
    t_list: List[float] = [0.0]
    S_list: List[int] = [S0]
    I_list: List[int] = [I0]
    R_list: List[int] = [R0]
    D_list: List[int] = [D0]

    t, S, I, R, D = 0.0, S0, I0, R0, D0
    while t < t_max and I > 0:
        rate_infection = params.beta * S * I / (params.N - D)  # Active population is N - D
        rate_recovery = params.gamma * I
        rate_death = params.mu * I
        total_rate = rate_infection + rate_recovery + rate_death

        if total_rate <= 0:
            logger.debug("No events remaining (I=0). Terminating simulation at t=%.2f", t)
            break

        t += rng.exponential(1.0 / total_rate)
        rand = rng.random()
        if rand < rate_infection / total_rate:
            S, I = S - 1, I + 1
        elif rand < (rate_infection + rate_recovery) / total_rate:
            I, R = I - 1, R + 1
        else:
            I, D = I - 1, D + 1

        t_list.append(t)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
        D_list.append(D)

    return Trajectory(
        np.asarray(t_list),
        np.asarray(S_list),
        np.asarray(I_list),
        np.asarray(R_list),
        np.asarray(D_list)
    )

# ------------------------------------------------------------------------------
# Deterministic Model (ODE)
# ------------------------------------------------------------------------------

def sir_ode(t: float, y: Sequence[float], params: SIRParams) -> Tuple[float, float, float, float]:
    """ODE system for the SIR model with mortality.

    Equations:
        dS/dt = -β S I / (N - D)
        dI/dt = β S I / (N - D) - (γ + μ) I
        dR/dt = γ I
        dD/dt = μ I
    """
    S, I, R, D = y
    N_active = params.N - D
    dSdt = -params.beta * S * I / N_active if N_active > 0 else 0.0
    dIdt = (params.beta * S * I / N_active if N_active > 0 else 0.0) - (params.gamma + params.mu) * I
    dRdt = params.gamma * I
    dDdt = params.mu * I
    return dSdt, dIdt, dRdt, dDdt

def integrate_deterministic(params: SIRParams, S0: int, I0: int, R0: int, D0: int, t_max: float, *, n_points: int = 1000) -> Trajectory:
    """Integrate the deterministic SIR model with mortality.

    Parameters
    ----------
    n_points : int, default=1000
        Number of time points for evaluation.
    """
    if S0 + I0 + R0 + D0 != params.N:
        raise ValueError(f"S0 ({S0}) + I0 ({I0}) + R0 ({R0}) + D0 ({D0}) must equal N ({params.N})")
    if S0 < 0 or I0 < 0 or R0 < 0 or D0 < 0:
        raise ValueError("Initial conditions must be non-negative")
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(
        sir_ode,
        (0.0, t_max),
        (S0, I0, R0, D0),
        args=(params,),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return Trajectory(sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3])

# ------------------------------------------------------------------------------
# Ensemble Simulation
# ------------------------------------------------------------------------------

def simulate_ensemble(params: SIRParams, S0: int, I0: int, R0: int, D0: int, t_max: float, n_runs: int = 1000, *, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Simulate multiple stochastic trajectories and compute quantiles.

    Returns
    -------
    t_grid : np.ndarray
        Common time grid for interpolated trajectories.
    quantiles : Dict[str, np.ndarray]
        Quantiles (2.5%, 50%, 97.5%) for S, I, R, D.
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    rng_master = default_rng(seed)
    t_grid = np.linspace(0.0, t_max, 1000)
    S_samples = np.empty((n_runs, t_grid.size))
    I_samples = np.empty((n_runs, t_grid.size))
    R_samples = np.empty((n_runs, t_grid.size))
    D_samples = np.empty((n_runs, t_grid.size))

    for k in tqdm(range(n_runs), desc="Running ensemble simulations", unit="run"):
        traj = gillespie_sir(params, S0, I0, R0, D0, t_max, rng=rng_master)
        S_samples[k, :] = np.interp(t_grid, traj.t, traj.S)
        I_samples[k, :] = np.interp(t_grid, traj.t, traj.I)
        R_samples[k, :] = np.interp(t_grid, traj.t, traj.R)
        D_samples[k, :] = np.interp(t_grid, traj.t, traj.D)

    quantiles = {
        "S": np.percentile(S_samples, [2.5, 50.0, 97.5], axis=0),
        "I": np.percentile(I_samples, [2.5, 50.0, 97.5], axis=0),
        "R": np.percentile(R_samples, [2.5, 50.0, 97.5], axis=0),
        "D": np.percentile(D_samples, [2.5, 50.0, 97.5], axis=0)
    }
    return t_grid, quantiles

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

def plot_trajectories(
    stoch: Trajectory,
    det: Trajectory,
    params: SIRParams,
    *,
    filename: str | Path | None = None,
    n_runs: int = 1000,
    seed: int | None = None,
    log_scale: bool = False
) -> None:
    """Plot stochastic vs deterministic trajectories and ensemble quantiles."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")

    # Stochastic trajectory
    ax = axes[0, 0]
    ax.plot(stoch.t, stoch.S, "b", label="S (stochastic)")
    ax.plot(stoch.t, stoch.I, "r", label="I (stochastic)")
    ax.plot(stoch.t, stoch.R, "g", label="R (stochastic)")
    ax.plot(stoch.t, stoch.D, "k", label="D (stochastic)")
    ax.set(title="Stochastic Simulation (Gillespie)", xlabel="Time", ylabel="Population")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
    ax.legend()

    # Deterministic trajectory
    ax = axes[0, 1]
    ax.plot(det.t, det.S, "b", label="S (ODE)")
    ax.plot(det.t, det.I, "r", label="I (ODE)")
    ax.plot(det.t, det.R, "g", label="R (ODE)")
    ax.plot(det.t, det.D, "k", label="D (ODE)")
    ax.set(title="Deterministic Model (ODE)", xlabel="Time")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
    ax.legend()

    # Ensemble comparison for I(t) and D(t)
    ax = axes[1, 0]
    t_grid, quantiles = simulate_ensemble(params, stoch.S[0], stoch.I[0], stoch.R[0], stoch.D[0], stoch.t[-1], n_runs=n_runs, seed=seed)
    ax.plot(stoch.t, stoch.I, "r", alpha=0.6, label="I (stochastic)")
    ax.plot(det.t, det.I, "k--", label="I (ODE)")
    ax.plot(t_grid, quantiles["I"][1], "b-", label="I (median)")
    ax.fill_between(t_grid, quantiles["I"][0], quantiles["I"][2], color="blue", alpha=0.2, label="I (95% CI)")
    ax.set(title="Infected Comparison", xlabel="Time", ylabel="Count")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
    ax.legend()

    # Deaths comparison
    ax = axes[1, 1]
    ax.plot(stoch.t, stoch.D, "k", alpha=0.6, label="D (stochastic)")
    ax.plot(det.t, det.D, "k--", label="D (ODE)")
    ax.plot(t_grid, quantiles["D"][1], "m-", label="D (median)")
    ax.fill_between(t_grid, quantiles["D"][0], quantiles["D"][2], color="magenta", alpha=0.2, label="D (95% CI)")
    ax.set(title="Deaths Comparison", xlabel="Time", ylabel="Deaths")
    ax.legend()

    # Add text box with key metrics
    fig.text(0.5, 0.02,
        f"Parameters: N={params.N}, β={params.beta:.3f}, γ={params.gamma:.3f}, μ={params.mu:.3f}\n"
        f"R₀={params.R0():.2f}, Herd immunity threshold={params.herd_immunity_threshold():.1%}\n"
        f"Final size: {det.final_size():.1%} (ODE), {stoch.final_size():.1%} (stochastic)\n"
        f"Total deaths: {det.D[-1]:.0f} (ODE), {stoch.D[-1]:.0f} (stochastic)",
        ha="center", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    if filename:
        filename = Path(filename)
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info("Figure saved to %s", filename)
    else:
        plt.show()
    plt.close(fig)

# ------------------------------------------------------------------------------
# Interactive Menu
# ------------------------------------------------------------------------------

def interactive_menu() -> argparse.Namespace:
    """Prompt user for simulation parameters interactively."""
    print("\n=== SIR Epidemic Model with Mortality ===")
    print("Enter values for the simulation parameters (press Enter for recommended values).")
    print("Invalid inputs will prompt re-entry.\n")

    # Helper function to get valid input
    def get_input(prompt: str, default: any, type_func: callable, valid_range: Tuple[any, any] | None = None, custom_validate: callable | None = None) -> any:
        while True:
            try:
                value = input(prompt).strip()
                if value == "":
                    return default
                value = type_func(value)
                if valid_range and not (valid_range[0] <= value <= valid_range[1]):
                    print(f"Value must be in range [{valid_range[0]}, {valid_range[1]}]. Try again.")
                    continue
                if custom_validate and not custom_validate(value):
                    print("Invalid value. Try again.")
                    continue
                return value
            except ValueError:
                print(f"Invalid {type_func.__name__} value. Try again.")

    # Get parameters
    N = get_input(
        "Total population (N, int, default=1000, range=[100, 100000]): ",
        default=1000,
        type_func=int,
        valid_range=(100, 100000)
    )
    I0 = get_input(
        f"Initial infected (I0, int, default=10, range=[1, {N}]): ",
        default=10,
        type_func=int,
        valid_range=(1, N)
    )
    beta = get_input(
        "Transmission rate (β, float, default=0.4, range=[0.0, 2.0]): ",
        default=0.4,
        type_func=float,
        valid_range=(0.0, 2.0)
    )
    gamma = get_input(
        "Recovery rate (γ, float, default=0.1, range=[0.01, 1.0]): ",
        default=0.1,
        type_func=float,
        valid_range=(0.01, 1.0)
    )
    mu = get_input(
        "Death rate (μ, float, default=0.01, range=[0.0, 1.0]): ",
        default=0.01,
        type_func=float,
        valid_range=(0.0, 1.0)
    )
    tmax = get_input(
        "Simulation duration (tmax, float, default=120.0, range=[1.0, 1000.0]): ",
        default=120.0,
        type_func=float,
        valid_range=(1.0, 1000.0)
    )
    nruns = get_input(
        "Number of stochastic runs (nruns, int, default=1000, range=[10, 10000]): ",
        default=1000,
        type_func=int,
        valid_range=(10, 10000)
    )
    seed = get_input(
        "Random seed (seed, int or empty for None, default=None, range=[0, 4294967295]): ",
        default=None,
        type_func=lambda x: None if x == "" else int(x),
        custom_validate=lambda x: x is None or 0 <= x <= 2**32 - 1
    )
    outfile = get_input(
        "Output figure file (outfile, str or empty for display, default=None): ",
        default=None,
        type_func=str
    )
    datafile = get_input(
        "Output CSV file (datafile, str or empty for none, default=None): ",
        default=None,
        type_func=str
    )
    log_scale = get_input(
        "Use logarithmic scale for plots? (y/n, default=n): ",
        default=False,
        type_func=lambda x: x.lower() in ('y', 'yes', 'true'),
    )

    # Create argparse.Namespace to mimic command-line arguments
    args = argparse.Namespace(
        N=N,
        I0=I0,
        beta=beta,
        gamma=gamma,
        mu=mu,
        tmax=tmax,
        nruns=nruns,
        seed=seed,
        outfile=outfile if outfile else None,
        datafile=datafile if datafile else None,
        log_scale=log_scale
    )
    return args

# ------------------------------------------------------------------------------
# Command-Line Interface
# ------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIR epidemic simulation with mortality")
    parser.add_argument("--N", type=int, default=1000, help="Total population")
    parser.add_argument("--I0", type=int, default=10, help="Initial infected")
    parser.add_argument("--beta", type=float, default=0.4, help="Transmission rate β")
    parser.add_argument("--gamma", type=float, default=0.1, help="Recovery rate γ")
    parser.add_argument("--mu", type=float, default=0.01, help="Death rate μ")
    parser.add_argument("--tmax", type=float, default=120.0, help="Simulation duration")
    parser.add_argument("--nruns", type=int, default=1000, help="Number of stochastic runs for ensemble")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--outfile", type=str, default=None, help="Path to save figure (PNG/PDF)")
    parser.add_argument("--datafile", type=str, default=None, help="Path to save trajectory data (CSV)")
    parser.add_argument("--log-scale", action="store_true", help="Use logarithmic scale for plots")
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> None:
    # Use interactive menu if no command-line arguments are provided
    if not argv or len(argv) == 1:
        args = interactive_menu()
    else:
        args = parse_args(argv)

    try:
        params = SIRParams(N=args.N, beta=args.beta, gamma=args.gamma, mu=args.mu)
    except ValueError as e:
        logger.error("Invalid parameters: %s", e)
        return

    S0 = args.N - args.I0
    if S0 < 0:
        logger.error("Initial infected (I0=%d) cannot exceed population (N=%d)", args.I0, args.N)
        return

    rng = default_rng(args.seed)
    try:
        stoch = gillespie_sir(params, S0, args.I0, 0, 0, args.tmax, rng=rng)
        det = integrate_deterministic(params, S0, args.I0, 0, 0, args.tmax)
    except ValueError as e:
        logger.error("Simulation failed: %s", e)
        return

    if args.datafile:
        try:
            stoch.save_to_csv(Path(args.datafile).with_suffix(".stoch.csv"))
            det.save_to_csv(Path(args.datafile).with_suffix(".det.csv"))
        except Exception as e:
            logger.error("Failed to save CSV files: %s", e)
            return

    try:
        plot_trajectories(stoch, det, params, filename=args.outfile, n_runs=args.nruns, seed=args.seed, log_scale=args.log_scale)
    except Exception as e:
        logger.error("Plotting failed: %s", e)

if __name__ == "__main__":
    main(sys.argv)