"""
gabriel_4cc_model.py
=====================
4-Cole-Cole dielectric model (Gabriel et al., 1996) and the impedance
calculations derived from it.
"""

import numpy as np
from dataclasses import dataclass

EPS0 = 8.8541878128e-12   # vacuum permittivity [F/m]


@dataclass
class FourColeColeParams:
    tissue_name: str
    eps_inf:   float
    sigma_s:   float
    delta_eps: list   # [d_eps1, d_eps2, d_eps3, d_eps4]
    tau:       list   # [tau1, tau2, tau3, tau4] in seconds
    alpha:     list   # [alpha1, alpha2, alpha3, alpha4]
    source:    str = "Gabriel et al. (1996) -- verify via FCC/IFAC-CNR"
    # True only once all 14 parameters have been confirmed against the
    # official source. Prevents unverified demo parameters from silently
    # being mistaken for validated ones downstream.
    verified:  bool = False

    def __post_init__(self):
        assert len(self.delta_eps) == 4, "Expected 4 dispersions (delta_eps)"
        assert len(self.tau) == 4, "Expected 4 dispersions (tau)"
        assert len(self.alpha) == 4, "Expected 4 dispersions (alpha)"


def complex_permittivity_4cc(f: np.ndarray, p: FourColeColeParams) -> np.ndarray:
    """eps*(f) = eps_inf + sum_n d_eps_n/(1+(j*omega*tau_n)^(1-alpha_n)) + sigma_s/(j*omega*EPS0)."""
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    eps = np.full_like(omega, p.eps_inf, dtype=complex)
    for d_eps, tau_n, alpha_n in zip(p.delta_eps, p.tau, p.alpha):
        eps += d_eps / (1.0 + (1j * omega * tau_n) ** (1.0 - alpha_n))
    eps += p.sigma_s / (1j * omega * EPS0)   # ionic conduction term
    return eps


def conductivity_and_permittivity(f: np.ndarray, p: FourColeColeParams):
    """Return (sigma, eps_real) derived from the complex permittivity."""
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    eps_star = complex_permittivity_4cc(f, p)
    return -omega * EPS0 * eps_star.imag, eps_star.real


def _admittance_per_kgeo(f: np.ndarray, p: FourColeColeParams) -> np.ndarray:
    """Admittance per unit cell-geometry factor: Y/K_geo = j*omega*EPS0*eps*(f).

    Shared by impedance_from_4cc and impedance_from_4cc_calibrated so the
    sign convention only needs to be correct in one place.
    """
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    return 1j * omega * EPS0 * complex_permittivity_4cc(f, p)


def impedance_from_4cc(f: np.ndarray, p: FourColeColeParams,
                        cell_length_m: float = 0.01,
                        cell_area_m2: float = 1e-3) -> np.ndarray:
    """Z(f) = K_geo / Y(f), K_geo = length/area. Sign convention: Y = +j*omega*EPS0*eps*(f)."""
    k_geo = cell_length_m / cell_area_m2
    return k_geo / _admittance_per_kgeo(f, p)


def calibrate_measurement_geometry(f: np.ndarray, p: FourColeColeParams,
                                    target_r_inf: float = 50.0,
                                    f_ref: float = 1e6) -> float:
    """Solve for K_geo such that |Z(f_ref)| == target_r_inf (unknown cell geometry)."""
    y_ref = _admittance_per_kgeo(np.array([f_ref]), p)[0]
    return float(target_r_inf * np.abs(y_ref))


def impedance_from_4cc_calibrated(f: np.ndarray, p: FourColeColeParams,
                                   target_r_inf: float = 50.0,
                                   f_ref: float = 1e6) -> np.ndarray:
    """Impedance calibrated so |Z(f_ref)| == target_r_inf."""
    k_geo = calibrate_measurement_geometry(f, p, target_r_inf, f_ref)
    return k_geo / _admittance_per_kgeo(f, p)


def impedance_from_eps_sigma(f: np.ndarray, eps_prime: np.ndarray, sigma: np.ndarray,
                              cell_length_m: float = 0.01,
                              cell_area_m2: float = 1e-3) -> np.ndarray:
    """Impedance directly from measured (eps', sigma) pairs, e.g. FCC calculator output."""
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    y_per_kgeo = (np.asarray(sigma, dtype=float)
                  + 1j * omega * EPS0 * np.asarray(eps_prime, dtype=float))
    k_geo = cell_length_m / cell_area_m2
    return k_geo / y_per_kgeo


MUSCLE_4CC_PARTIAL = FourColeColeParams(
    tissue_name="muscle",
    eps_inf=4.0,   # confirmed (Gabriel 1996; cross-checked 2024)
    sigma_s=0.2,   # confirmed (Gabriel 1996; cross-checked 2024)
    # Remaining 12 values are placeholders for exercising the code path
    # only (see `verified=False` below) -- they are not claimed to be in
    # physically consistent tau order and must not be used for the paper.
    delta_eps=[5.0e7, 8.0e3, 2.5e5, 5.0e3],
    tau=[7.234e-6, 353.7e-9, 318.7e-3, 7.96e-12],
    alpha=[0.10, 0.10, 0.10, 0.20],
    source="Gabriel et al. (1996) -- eps_inf and sigma_s confirmed; remaining "
           "parameters MUST be verified at "
           "fcc.gov/general/body-tissue-dielectric-parameters before "
           "publication use",
    verified=False,
)


def select_beta_band(f: np.ndarray, Z: np.ndarray,
                      f_low: float = 1e3, f_high: float = 1e6):
    """
    Select only the beta-dispersion sub-band -- the region where a
    single-dispersion Cole-Cole model is physically identifiable from
    bench measurements.

    The beta dispersion typically spans ~1 kHz to a few MHz for most
    soft tissues (cell membranes). Below 1 kHz the alpha dispersion
    (electrode/interface) dominates; above a few MHz the delta
    dispersion (bound-water relaxation) takes over.

    Returns (f_band, Z_band) -- only the points inside the band.
    """
    mask = (f >= f_low) & (f <= f_high)
    return f[mask], Z[mask]


def fit_single_cole_cole_subband(f_full: np.ndarray, Z_full: np.ndarray,
                                  f_low: float = 1e3,
                                  f_high: float = 1e6,
                                  snr_db: float = 30.0,
                                  n_samples: int = 4000,
                                  n_warmup: int = 2000,
                                  seed: int = 42):
    """
    Fit a single-dispersion Cole-Cole model restricted to the beta
    sub-band, honestly reporting the limits of identifiability.

    Returns a dict with:
        'f_band', 'Z_band'  : data used in the fit (sub-band)
        'mcmc_samples'      : MCMC samples of the simple Cole-Cole model
        'r_squared_subband' : fit quality WITHIN THE SUB-BAND
        'coverage_fraction' : fraction of the full spectrum covered by the band
    """
    from .ls_fitting import nlls_fit
    from .mcmc_sampler import AdaptiveMCMC, log_posterior
    from .data_generation import snr_to_sigma
    from .cole_model import cole_cole_impedance

    f_band, z_clean_band = select_beta_band(f_full, Z_full, f_low, f_high)
    if len(f_band) < 5:
        raise ValueError(
            f"Beta sub-band [{f_low:.0f}, {f_high:.0f}] Hz contains only "
            f"{len(f_band)} points from the original grid -- not enough for "
            f"fitting. Use a denser frequency grid in this region."
        )

    # Add realistic measurement noise only within the sub-band
    rng = np.random.default_rng(seed)
    noise_scale = snr_to_sigma(snr_db) * np.abs(z_clean_band)
    z_obs_band = (z_clean_band.real + rng.normal(0, noise_scale)
                  + 1j * (z_clean_band.imag + rng.normal(0, noise_scale)))

    # NLLS for initialization
    ls_result = nlls_fit(f_band, z_obs_band, n_restarts=5)
    if ls_result.get("converged"):
        p = ls_result["params"]
        theta0 = np.array([max(p["R_inf"], 1.),
                            max(p["R0"] - p["R_inf"], 1.),
                            np.log(max(p["tau"], 1e-10)),
                            np.clip(p["alpha"], 0.05, 0.98)])
    else:
        r_inf_est = float(np.abs(z_obs_band[-3:]).mean())
        r0_est = float(np.abs(z_obs_band[:3]).mean())
        theta0 = np.array([max(r_inf_est, 1.),
                            max(r0_est - r_inf_est, 1.),
                            np.log(1.0 / (2 * np.pi * np.sqrt(f_low * f_high))),
                            0.75])

    log_post = lambda th: log_posterior(th, f_band, z_obs_band, snr_db)
    sampler = AdaptiveMCMC(n_samples=n_samples, n_warmup=n_warmup, seed=seed)
    mcmc_result = sampler.run(log_post, theta0, verbose=True)

    samples = mcmc_result["samples"]
    r_inf_med, dr_med = np.median(samples[:, 0]), np.median(samples[:, 1])
    tau_med, alpha_med = np.exp(np.median(samples[:, 2])), np.median(samples[:, 3])
    z_fit = cole_cole_impedance(f_band, r_inf_med, r_inf_med + dr_med, tau_med, alpha_med)
    ss_res = np.sum(np.abs(z_obs_band - z_fit) ** 2)
    ss_tot = np.sum(np.abs(z_obs_band - z_obs_band.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "f_band": f_band, "Z_band": z_obs_band, "Z_band_clean": z_clean_band,
        "ls_result": ls_result, "mcmc_samples": samples,
        "mcmc_warmup": mcmc_result["warmup"],
        "r_squared_subband": r_squared,
        "coverage_fraction": len(f_band) / len(f_full),
        "band_range": (f_low, f_high),
        "params_median": {"R_inf": r_inf_med, "R0": r_inf_med + dr_med,
                           "tau": tau_med, "alpha": alpha_med},
    }
