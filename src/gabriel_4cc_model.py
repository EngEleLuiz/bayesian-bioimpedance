import numpy as np
from dataclasses import dataclass, field
from typing import Optional

EPS0 = 8.8541878128e-12   # permissividade vácuo [F/m]


# ===========================================================================
# Modelo de 4-Cole-Cole — estrutura matemática (Gabriel et al., 1996)
# ===========================================================================

@dataclass
class FourColeColeParams:
    tissue_name: str
    eps_inf:   float
    sigma_s:   float
    delta_eps: list   # [Δε1, Δε2, Δε3, Δε4]
    tau:       list   # [τ1, τ2, τ3, τ4] em segundos
    alpha:     list   # [α1, α2, α3, α4]
    source:    str = "Gabriel et al. (1996) — verificar via FCC/IFAC-CNR"

    def __post_init__(self):
        assert len(self.delta_eps) == 4, "Esperado 4 dispersões (Δε)"
        assert len(self.tau)       == 4, "Esperado 4 dispersões (τ)"
        assert len(self.alpha)     == 4, "Esperado 4 dispersões (α)"


def complex_permittivity_4cc(f: np.ndarray, p: FourColeColeParams) -> np.ndarray:

    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    eps = np.full_like(omega, p.eps_inf, dtype=complex)

    for dEps, tau_n, alpha_n in zip(p.delta_eps, p.tau, p.alpha):
        eps += dEps / (1.0 + (1j * omega * tau_n) ** (1.0 - alpha_n))

    # Termo de condução iônica (contribui para a parte imaginária)
    eps += p.sigma_s / (1j * omega * EPS0)

    return eps


def conductivity_and_permittivity(f: np.ndarray, p: FourColeColeParams):
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    eps_star = complex_permittivity_4cc(f, p)
    sigma = -omega * EPS0 * eps_star.imag
    eps_real = eps_star.real
    return sigma, eps_real


def impedance_from_4cc(f: np.ndarray, p: FourColeColeParams,
                        cell_length_m: float = 0.01,
                        cell_area_m2: float = 1e-3) -> np.ndarray:
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    eps_star = complex_permittivity_4cc(f, p)
    # CORRIGIDO: sinal positivo (+1j), não negativo
    Y_per_Kgeo = 1j * omega * EPS0 * eps_star
    K_geo = cell_length_m / cell_area_m2
    Z = K_geo / Y_per_Kgeo
    return Z


def calibrar_geometria_medicao(f: np.ndarray, p: FourColeColeParams,
                                R_inf_desejado: float = 50.0,
                                f_ref: float = 1e6) -> float:
    omega_ref = 2.0 * np.pi * f_ref
    eps_star_ref = complex_permittivity_4cc(np.array([f_ref]), p)[0]
    Y_per_Kgeo_ref = 1j * omega_ref * EPS0 * eps_star_ref
    # |Z_ref| = K_geo / |Y_per_Kgeo_ref|  =>  K_geo = R_inf_desejado * |Y_per_Kgeo_ref|
    K_geo = R_inf_desejado * np.abs(Y_per_Kgeo_ref)
    return float(K_geo)


def impedance_from_4cc_calibrated(f: np.ndarray, p: FourColeColeParams,
                                   R_inf_desejado: float = 50.0,
                                   f_ref: float = 1e6) -> np.ndarray:
    K_geo = calibrar_geometria_medicao(f, p, R_inf_desejado, f_ref)
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    eps_star = complex_permittivity_4cc(f, p)
    Y_per_Kgeo = 1j * omega * EPS0 * eps_star
    return K_geo / Y_per_Kgeo



def impedance_from_eps_sigma(f: np.ndarray, eps_prime: np.ndarray,
                              sigma: np.ndarray,
                              cell_length_m: float = 0.01,
                              cell_area_m2: float = 1e-3) -> np.ndarray:
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    eps_prime = np.asarray(eps_prime, dtype=float)
    sigma     = np.asarray(sigma, dtype=float)

    Y_per_Kgeo = sigma + 1j * omega * EPS0 * eps_prime
    K_geo = cell_length_m / cell_area_m2
    Z = K_geo / Y_per_Kgeo
    return Z

MUSCLE_4CC_PARTIAL = FourColeColeParams(
    tissue_name="muscle",
    eps_inf=4.0,        # ✅ confirmado (Gabriel 1996; múltiplas fontes 2024)
    sigma_s=0.2,        # ✅ confirmado (Gabriel 1996; múltiplas fontes 2024)
    delta_eps=[5.0e7, 8.0e3, 2.5e5, 5.0e3],   
    tau=[7.234e-6, 353.7e-9, 318.7e-3, 7.96e-12], 
    alpha=[0.10, 0.10, 0.10, 0.20],            
    source="Gabriel et al. (1996) — ε∞ e σ_s confirmados; demais "
           "parâmetros DEVEM ser verificados em "
           "fcc.gov/general/body-tissue-dielectric-parameters "
           "antes de uso em publicação"
)


# ===========================================================================
# Ajuste honesto na sub-banda β (1 kHz – 1 MHz)
# ===========================================================================

def select_beta_band(f: np.ndarray, Z: np.ndarray,
                      f_low: float = 1e3, f_high: float = 1e6):
    """
    Seleciona apenas a sub-banda β do espectro — a região onde um
    Cole-Cole de 1 dispersão é fisicamente identificável a partir de
    medições de bancada.

    A dispersão β ocorre tipicamente entre ~1 kHz e ~1-10 MHz para a
    maioria dos tecidos moles (membranas celulares). Abaixo de 1 kHz
    domina a dispersão α (eletrodo/interface); acima de poucos MHz
    começa a dispersão δ (relaxação de água ligada).

    Retorna (f_band, Z_band) — apenas os pontos dentro da banda.
    """
    mask = (f >= f_low) & (f <= f_high)
    return f[mask], Z[mask]


def fit_single_cole_cole_subband(f_full: np.ndarray, Z_full: np.ndarray,
                                   f_low: float = 1e3,
                                   f_high: float = 1e6,
                                   snr_db: float = 30.0,
                                   n_amostras: int = 4000,
                                   n_warmup: int = 2000,
                                   seed: int = 42):
    """
    Ajusta um Cole-Cole de 1 dispersão apenas na sub-banda β,
    reportando honestamente os limites de identificabilidade.

    Retorna dict com:
        'f_band', 'Z_band'  : dados usados no ajuste (sub-banda)
        'mcmc_samples'      : amostras MCMC do Cole-Cole simples
        'r_squared_subband' : qualidade do ajuste NA SUB-BANDA
        'coverage_fraction' : fração do espectro total coberta pela banda
    """
    from .ls_fitting import nlls_fit
    from .mcmc_sampler import AdaptiveMCMC, log_posterior
    from .data_generation import generate_eis_data, snr_to_sigma

    f_band, Z_clean_band = select_beta_band(f_full, Z_full, f_low, f_high)

    if len(f_band) < 5:
        raise ValueError(
            f"Sub-banda β [{f_low:.0f}, {f_high:.0f}] Hz contém apenas "
            f"{len(f_band)} pontos da grade original — insuficiente para "
            f"ajuste. Use uma grade de frequência mais densa nesta região."
        )

    # Adicionar ruído de medição realista apenas na sub-banda
    rng = np.random.default_rng(seed)
    sigma_noise = snr_to_sigma(snr_db)
    noise_scale = sigma_noise * np.abs(Z_clean_band)
    Z_obs_band = (Z_clean_band.real + rng.normal(0, noise_scale)
                  + 1j * (Z_clean_band.imag + rng.normal(0, noise_scale)))

    # NLLS para inicialização
    ls_result = nlls_fit(f_band, Z_obs_band, n_restarts=5)

    if ls_result.get("converged"):
        p = ls_result["params"]
        theta0 = np.array([max(p["R_inf"], 1.),
                           max(p["R0"] - p["R_inf"], 1.),
                           np.log(max(p["tau"], 1e-10)),
                           np.clip(p["alpha"], 0.05, 0.98)])
    else:
        R_inf_est = float(np.abs(Z_obs_band[-3:]).mean())
        R0_est    = float(np.abs(Z_obs_band[:3]).mean())
        theta0 = np.array([max(R_inf_est, 1.),
                           max(R0_est - R_inf_est, 1.),
                           np.log(1.0 / (2*np.pi*np.sqrt(f_low*f_high))),
                           0.75])

    fn = lambda th: log_posterior(th, f_band, Z_obs_band, snr_db)
    sampler = AdaptiveMCMC(n_samples=n_amostras, n_warmup=n_warmup, seed=seed)
    mcmc_result = sampler.run(fn, theta0, verbose=True)

    # R² na sub-banda (qualidade do ajuste local)
    from .cole_model import cole_cole_impedance
    samples = mcmc_result["samples"]
    R_inf_med = np.median(samples[:, 0])
    dR_med    = np.median(samples[:, 1])
    tau_med   = np.exp(np.median(samples[:, 2]))
    alpha_med = np.median(samples[:, 3])
    Z_fit = cole_cole_impedance(f_band, R_inf_med, R_inf_med + dR_med,
                                 tau_med, alpha_med)
    ss_res = np.sum(np.abs(Z_obs_band - Z_fit) ** 2)
    ss_tot = np.sum(np.abs(Z_obs_band - Z_obs_band.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    coverage = len(f_band) / len(f_full)

    return {
        "f_band":            f_band,
        "Z_band":            Z_obs_band,
        "Z_band_clean":      Z_clean_band,
        "ls_result":         ls_result,
        "mcmc_samples":      samples,
        "mcmc_warmup":       mcmc_result["warmup"],
        "r_squared_subband": r_squared,
        "coverage_fraction": coverage,
        "band_range":        (f_low, f_high),
        "params_median": {
            "R_inf": R_inf_med, "R0": R_inf_med + dR_med,
            "tau": tau_med, "alpha": alpha_med
        }
    }
