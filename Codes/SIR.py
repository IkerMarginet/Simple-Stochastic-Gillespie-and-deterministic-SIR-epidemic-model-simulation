
"""Stochastic (Gillespie) and deterministic SIR epidemic model simulation.

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
    """Epidemiological parameters for the SIR model.

    Attributes
    ----------
    N : int
        Total population size (constant).
    beta : float
        Transmission rate (β). Units: 1/time.
    gamma : float
        Recovery rate (γ). Units: 1/time.
    """
    N: int
    beta: float
    gamma: float

    def __post_init__(self):
        """Validate parameters."""
        if self.N <= 0:
            raise ValueError("Population size N must be positive")
        if self.beta < 0 or self.gamma < 0:
            raise ValueError("Rates β and γ must be non-negative")

    def R0(self) -> float:
        """Basic reproduction number R₀ = β / γ."""
        return self.beta / self.gamma

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

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {"time": self.t, "S": self.S, "I": self.I, "R": self.R}

    def peak_infections(self) -> Tuple[float, float]:
        """Return (t_peak, I_max)."""
        idx = int(np.argmax(self.I))
        return float(self.t[idx]), float(self.I[idx])

    def final_size(self) -> float:
        """Cumulative infected/recovered fraction at t → ∞."""
        initial_pop = self.S[0] + self.I[0] + self.R[0]
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
            writer.writerow(["time", "S", "I", "R"])
            for t, s, i, r in zip(self.t, self.S, self.I, self.R):
                formatted_row = [
                    f"{t:.6f}" if isinstance(t, float) else t,
                    f"{s:.6f}" if isinstance(s, float) else s,
                    f"{i:.6f}" if isinstance(i, float) else i,
                    f"{r:.6f}" if isinstance(r, float) else r
                ]
                writer.writerow(formatted_row)
        logger.info("Trajectory saved to %s", filename)

# ------------------------------------------------------------------------------
# Stochastic Simulation (Gillespie SSA)
# ------------------------------------------------------------------------------

def gillespie_sir(params: SIRParams, S0: int, I0: int, R0: int, t_max: float, *, rng: Generator | None = None) -> Trajectory:
    """Stochastic SIR simulation using the Gillespie algorithm.

    Parameters
    ----------
    params : SIRParams
        Model parameters (N, β, γ).
    S0, I0, R0 : int
        Initial conditions (S + I + R = N).
    t_max : float
        Maximum simulation time.
    rng : np.random.Generator, optional
        Random number generator. If None, a non-deterministic RNG is created.

    Returns
    -------
    Trajectory
        Temporal trajectory (time, S, I, R) as numpy arrays.
    """
    if S0 + I0 + R0 != params.N:
        raise ValueError(f"S0 ({S0}) + I0 ({I0}) + R0 ({R0}) must equal N ({params.N})")
    if S0 < 0 or I0 < 0 or R0 < 0:
        raise ValueError("Initial conditions must be non-negative")
    if t_max <= 0:
        raise ValueError("t_max must be positive")

    rng = default_rng() if rng is None else rng
    t_list: List[float] = [0.0]
    S_list: List[int] = [S0]
    I_list: List[int] = [I0]
    R_list: List[int] = [R0]

    t, S, I, R = 0.0, S0, I0, R0
    while t < t_max and I > 0:
        rate_infection = params.beta * S * I / params.N
        rate_recovery = params.gamma * I
        total_rate = rate_infection + rate_recovery

        if total_rate <= 0:
            logger.debug("No infections remaining (I=0). Terminating simulation at t=%.2f", t)
            break

        t += rng.exponential(1.0 / total_rate)
        if rng.random() < rate_infection / total_rate:
            S, I = S - 1, I + 1
        else:
            I, R = I - 1, R + 1

        t_list.append(t)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)

    return Trajectory(np.asarray(t_list), np.asarray(S_list), np.asarray(I_list), np.asarray(R_list))

# ------------------------------------------------------------------------------
# Deterministic Model (ODE)
# ------------------------------------------------------------------------------

def sir_ode(t: float, y: Sequence[float], params: SIRParams) -> Tuple[float, float, float]:
    """ODE system for the SIR model.

    Equations:
        dS/dt = -β S I / N
        dI/dt =  β S I / N - γ I
        dR/dt =  γ I
    """
    S, I, R = y
    dSdt = -params.beta * S * I / params.N
    dIdt = params.beta * S * I / params.N - params.gamma * I
    dRdt = params.gamma * I
    return dSdt, dIdt, dRdt

def integrate_deterministic(params: SIRParams, S0: int, I0: int, R0: int, t_max: float, *, n_points: int = 1000) -> Trajectory:
    """Integrate the deterministic SIR model.

    Parameters
    ----------
    n_points : int, default=1000
        Number of time points for evaluation.
    """
    if S0 + I0 + R0 != params.N:
        raise ValueError(f"S0 ({S0}) + I0 ({I0}) + R0 ({R0}) must equal N ({params.N})")
    if S0 < 0 or I0 < 0 or R0 < 0:
        raise ValueError("Initial conditions must be non-negative")
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(sir_ode, (0.0, t_max), (S0, I0, R0), args=(params,), t_eval=t_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return Trajectory(sol.t, sol.y[0], sol.y[1], sol.y[2])

# ------------------------------------------------------------------------------
# Ensemble Simulation
# ------------------------------------------------------------------------------

def simulate_ensemble(params: SIRParams, S0: int, I0: int, R0: int, t_max: float, n_runs: int = 1000, *, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Simulate multiple stochastic trajectories and compute quantiles.

    Returns
    -------
    t_grid : np.ndarray
        Common time grid for interpolated trajectories.
    quantiles : Dict[str, np.ndarray]
        Quantiles (2.5%, 50%, 97.5%) for S, I, R.
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    rng_master = default_rng(seed)
    t_grid = np.linspace(0.0, t_max, 1000)
    S_samples = np.empty((n_runs, t_grid.size))
    I_samples = np.empty((n_runs, t_grid.size))
    R_samples = np.empty((n_runs, t_grid.size))

    for k in tqdm(range(n_runs), desc="Running ensemble simulations", unit="run"):
        traj = gillespie_sir(params, S0, I0, R0, t_max, rng=rng_master)
        S_samples[k, :] = np.interp(t_grid, traj.t, traj.S)
        I_samples[k, :] = np.interp(t_grid, traj.t, traj.I)
        R_samples[k, :] = np.interp(t_grid, traj.t, traj.R)

    quantiles = {
        "S": np.percentile(S_samples, [2.5, 50.0, 97.5], axis=0),
        "I": np.percentile(I_samples, [2.5, 50.0, 97.5], axis=0),
        "R": np.percentile(R_samples, [2.5, 50.0, 97.5], axis=0)
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")

    # Stochastic trajectory
    ax = axes[0, 0]
    ax.plot(stoch.t, stoch.S, "b", label="S (stochastic)")
    ax.plot(stoch.t, stoch.I, "r", label="I (stochastic)")
    ax.plot(stoch.t, stoch.R, "g", label="R (stochastic)")
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
    ax.set(title="Deterministic Model (ODE)", xlabel="Time")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
    ax.legend()

    # Ensemble comparison for I(t)
    ax = axes[1, 0]
    t_grid, quantiles = simulate_ensemble(params, stoch.S[0], stoch.I[0], stoch.R[0], stoch.t[-1], n_runs=n_runs, seed=seed)
    ax.plot(stoch.t, stoch.I, "r", alpha=0.6, label="I (stochastic)")
    ax.plot(det.t, det.I, "k--", label="I (ODE)")
    ax.plot(t_grid, quantiles["I"][1], "b-", label="I (median)")
    ax.fill_between(t_grid, quantiles["I"][0], quantiles["I"][2], color="blue", alpha=0.2, label="I (95% CI)")
    ax.set(title="Infected Comparison", xlabel="Time", ylabel="I")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
    ax.legend()

    # Key metrics
    ax = axes[1, 1]
    ax.axis("off")
    text = [
        "Parameters:",
        f"N = {params.N}",
        f"β = {params.beta:.3f}",
        f"γ = {params.gamma:.3f}",
        f"R₀ = {params.R0():.2f}",
        f"Herd immunity = {params.herd_immunity_threshold():.1%}",
        f"I_max (stoch) = {stoch.peak_infections()[1]:.0f} at t = {stoch.peak_infections()[0]:.1f}",
        f"I_max (ODE) = {det.peak_infections()[1]:.0f} at t = {det.peak_infections()[0]:.1f}",
        f"Duration (stoch) = {stoch.epidemic_duration():.1f}",
        f"Duration (ODE) = {det.epidemic_duration():.1f}"
    ]
    ax.text(0.02, 0.98, "\n".join(text), va="top", fontsize=11)

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
    print("\n=== SIR Epidemic Model Interactive Menu ===")
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

    # Helper function to validate file paths
    def valid_path(value: str) -> bool:
        if not value:
            return True
        path = Path(value)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    # Helper function to validate seed
    def valid_seed(value: any) -> bool:
        if value is None:
            return True
        return 0 <= value <= 2**32 - 1

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
        custom_validate=valid_seed
    )
    outfile = get_input(
        "Output figure file (outfile, str or empty for display, default=None): ",
        default=None,
        type_func=str,
        custom_validate=valid_path
    )
    datafile = get_input(
        "Output CSV file (datafile, str or empty for none, default=None): ",
        default=None,
        type_func=str,
        custom_validate=valid_path
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
    parser = argparse.ArgumentParser(description="Stochastic and deterministic SIR epidemic simulation")
    parser.add_argument("--N", type=int, default=1000, help="Total population")
    parser.add_argument("--I0", type=int, default=10, help="Initial infected")
    parser.add_argument("--beta", type=float, default=0.4, help="Transmission rate β")
    parser.add_argument("--gamma", type=float, default=0.1, help="Recovery rate γ")
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
        params = SIRParams(N=args.N, beta=args.beta, gamma=args.gamma)
    except ValueError as e:
        logger.error("Invalid parameters: %s", e)
        return

    S0 = args.N - args.I0
    if S0 < 0:
        logger.error("Initial infected (I0=%d) cannot exceed population (N=%d)", args.I0, args.N)
        return

    rng = default_rng(args.seed)
    try:
        stoch = gillespie_sir(params, S0, args.I0, 0, args.tmax, rng=rng)
        det = integrate_deterministic(params, S0, args.I0, 0, args.tmax)
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

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------

def _test_mass_conservation() -> None:
    params = SIRParams(N=100, beta=0.4, gamma=0.1)
    traj = integrate_deterministic(params, 90, 10, 0, 30)
    total = traj.S + traj.I + traj.R
    assert np.allclose(total, params.N), "Mass conservation failed"

def _test_R0() -> None:
    params = SIRParams(N=1, beta=0.4, gamma=0.1)
    assert math.isclose(params.R0(), 4.0), "R₀ calculation incorrect"

def _test_edge_cases() -> None:
    params = SIRParams(N=100, beta=0.0, gamma=0.1)
    traj = gillespie_sir(params, 90, 10, 0, 10.0)
    assert np.all(traj.I <= 10), "Infections should not increase with β=0"

def _test_csv_output(tmp_path: Path) -> None:
    params = SIRParams(N=100, beta=0.4, gamma=0.1)
    traj = integrate_deterministic(params, 90, 10, 0, 10.0)
    filename = tmp_path / "test.csv"
    traj.save_to_csv(filename)
    assert filename.exists(), "CSV file was not created"
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["time", "S", "I", "R"], "Incorrect CSV header"

if __name__ == "__main__":
    main(sys.argv)