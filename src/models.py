"""
models.py
=========
Definição formal dos 4 modelos de circuito equivalente para comparação bayesiana.

Modelos implementados:
    M1 — Debye:            caso especial do Cole-Cole com α=1 fixo (3 parâmetros)
    M2 — Cole-Cole:        modelo base da disciplina (4 parâmetros)
    M3 — Double Cole-Cole: duas dispersões independentes (7 parâmetros)
    M4 — Randles + CPE:    topologia diferente, elemento de fase constante (4 parâmetros)

A comparação formal M1 vs M2 vs M3 vs M4 via WAIC/LOO responde a pergunta
central do artigo: qual modelo é mais suportado pelos dados, penalizando
complexidade desnecessária?

Referências:
    Herencsar, N. et al. (2020). A Comparative Study of Two Fractional-Order
    Equivalent Electrical Circuits for Modeling the Electrical Impedance of
    Dental Tissues. Entropy, 22(10), 1117.

    arXiv:2407.20297 — Bayesian EIS: An Assessment of Commonly Used Equivalent
    Circuit Models (Corrosion Analysis). Apresenta framework análogo para
    comparação formal de ECMs via Bayesian Inference.

    Barsoukov & Macdonald (2018), Cap. 3 — topologias de circuitos equivalentes.

    Vehtari, A., Gelman, A. & Gabry, J. (2017). Practical Bayesian model
    evaluation using leave-one-out cross-validation and WAIC.
    Stat Comput 27, 1413–1432.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from .data_generation import snr_to_sigma


# ===========================================================================
# Funções de impedância
# ===========================================================================

def impedance_debye(f, R_inf, delta_R, tau):
    """
    Modelo Debye (Cole-Cole com α=1 — capacitor ideal).

    Z(ω) = R_inf + ΔR / (1 + jωτ)

    Limites:
        R_inf > 0,  delta_R > 0,  tau > 0
    """
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    return R_inf + delta_R / (1.0 + 1j * omega * tau)


def impedance_cole_cole(f, R_inf, delta_R, tau, alpha):
    """
    Modelo Cole-Cole — base da disciplina.

    Z(ω) = R_inf + ΔR / (1 + (jωτ)^α)

    α ∈ (0,1]: α=1 → Debye; α<1 → distribuição de tempos de relaxação.
    """
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    return R_inf + delta_R / (1.0 + (1j * omega * tau) ** alpha)


def impedance_double_cole_cole(f, R_inf, dR1, tau1, alpha1, dR2, tau2, alpha2):
    """
    Modelo Double Cole-Cole — duas dispersões independentes.

    Z(ω) = R_inf + ΔR1/(1+(jωτ1)^α1) + ΔR2/(1+(jωτ2)^α2)

    Motivação: tecidos com dois processos de relaxação distintos
    (ex.: dispersão-α e dispersão-β simultâneas em tecido composto).

    Referência: Grimnes & Martinsen (2015), Cap. 4 — dispersões múltiplas.
    """
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    Z1 = dR1 / (1.0 + (1j * omega * tau1) ** alpha1)
    Z2 = dR2 / (1.0 + (1j * omega * tau2) ** alpha2)
    return R_inf + Z1 + Z2


def impedance_randles_cpe(f, Rs, Rp, Q, n):
    """
    Circuito de Randles modificado com CPE (Constant Phase Element).

    Topologia: Rs em série com (Rp || CPE)
    Z_CPE = 1 / (Q·(jω)^n)
    Z(ω) = Rs + Rp / (1 + Rp·Q·(jω)^n)

    Rs: resistência de solução/bulk [Ω]
    Rp: resistência de polarização/transferência de carga [Ω]
    Q:  coeficiente do CPE [S·s^n]
    n:  expoente do CPE ∈ (0,1]  (n=1 → capacitor puro)

    Referência: Barsoukov & Macdonald (2018), Cap. 2;
                Metrohm Application Note AN-EIS-004.
    """
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    Z_CPE_inv = Q * (1j * omega) ** n     # admitância CPE
    denom = 1.0 + Rp * Z_CPE_inv
    return Rs + Rp / denom


# ===========================================================================
# Verossimilhança comum (Gaussiana complexa proporcional ao módulo)
# ===========================================================================

def _log_likelihood_complex(Z_pred, Z_obs, snr_db):
    """
    Log-verossimilhança Gaussiana complexa com σ(f) = σ_noise·|Z_pred(f)|.
    Compartilhado por todos os modelos.
    """
    sigma_noise = snr_to_sigma(snr_db)
    sigma_f = sigma_noise * np.abs(Z_pred)
    sigma_f = np.maximum(sigma_f, 1e-12)
    res_R = Z_obs.real - Z_pred.real
    res_X = Z_obs.imag - Z_pred.imag
    ll = -np.sum(np.log(sigma_f)) - 0.5 * np.sum((res_R / sigma_f) ** 2)
    ll += -np.sum(np.log(sigma_f)) - 0.5 * np.sum((res_X / sigma_f) ** 2)
    return ll


def _log_likelihood_pointwise(Z_pred, Z_obs, snr_db):
    """
    Log-verossimilhança ponto a ponto (por frequência) — necessária para WAIC/LOO.

    Retorna array de shape (2*n_freq,) com log-likelihood de cada ponto de dado.
    Os 2*n_freq pontos são: [parte real f0...fN, parte imag f0...fN].
    """
    sigma_noise = snr_to_sigma(snr_db)
    sigma_f = sigma_noise * np.abs(Z_pred)
    sigma_f = np.maximum(sigma_f, 1e-12)
    res_R = Z_obs.real - Z_pred.real
    res_X = Z_obs.imag - Z_pred.imag
    ll_R = -np.log(sigma_f) - 0.5 * (res_R / sigma_f) ** 2
    ll_X = -np.log(sigma_f) - 0.5 * (res_X / sigma_f) ** 2
    return np.concatenate([ll_R, ll_X])   # shape: (2*n_freq,)


# ===========================================================================
# Definição formal de cada modelo (priors + log-posterior + impedância)
# ===========================================================================

@dataclass
class ImpedanceModel:
    """Container formal para cada modelo de circuito equivalente."""
    name:        str
    short_name:  str
    n_params:    int
    param_names: list
    impedance_fn: Callable
    log_prior_fn: Callable
    theta0_fn:   Callable    # retorna ponto inicial dado (f, Z_obs)
    bounds_lower: list
    bounds_upper: list
    description: str = ""


# ---------------------------------------------------------------------------
# M1 — Debye
# ---------------------------------------------------------------------------

def _prior_debye(theta):
    R_inf, dR, log_tau = theta
    if R_inf <= 0 or dR <= 0:
        return -np.inf
    lp = 0.0
    lp += -0.5 * ((np.log(R_inf) - np.log(50.0)) / 0.5) ** 2 - np.log(R_inf * 0.5)
    lp += -0.5 * ((np.log(dR)   - np.log(150.0)) / 0.5) ** 2 - np.log(dR * 0.5)
    lp += -0.5 * ((log_tau - (-12.0)) / 2.0) ** 2
    return lp


def _logpost_debye(theta, f, Z_obs, snr_db):
    R_inf, dR, log_tau = theta
    lp = _prior_debye(theta)
    if not np.isfinite(lp):
        return -np.inf
    tau = np.exp(log_tau)
    R0 = R_inf + dR
    if R_inf <= 0 or dR <= 0 or tau <= 0 or R0 <= R_inf:
        return -np.inf
    Z_pred = impedance_debye(f, R_inf, dR, tau)
    ll = _log_likelihood_complex(Z_pred, Z_obs, snr_db)
    return lp + ll if np.isfinite(ll) else -np.inf


def _theta0_debye(f, Z_obs, snr_db):
    from .ls_fitting import nlls_fit
    r = nlls_fit(f, Z_obs)
    if r.get("converged"):
        p = r["params"]
        return np.array([max(p["R_inf"], 1.0),
                         max(p["R0"] - p["R_inf"], 1.0),
                         np.log(max(p["tau"], 1e-10))])
    return np.array([50.0, 150.0, np.log(7.96e-6)])


MODEL_DEBYE = ImpedanceModel(
    name="Debye (α=1)",
    short_name="M1_Debye",
    n_params=3,
    param_names=["R_inf", "delta_R", "log_tau"],
    impedance_fn=lambda f, th: impedance_debye(f, th[0], th[1], np.exp(th[2])),
    log_prior_fn=_prior_debye,
    theta0_fn=_theta0_debye,
    bounds_lower=[0.1, 0.1, np.log(1e-9)],
    bounds_upper=[2000., 5000., np.log(1.0)],
    description="Caso especial de Cole-Cole com α=1 (capacitor ideal, 3 parâmetros)."
)


# ---------------------------------------------------------------------------
# M2 — Cole-Cole
# ---------------------------------------------------------------------------

def _prior_cole_cole(theta):
    R_inf, dR, log_tau, alpha = theta
    if R_inf <= 0 or dR <= 0 or alpha <= 0 or alpha > 1.0:
        return -np.inf
    lp = 0.0
    lp += -0.5 * ((np.log(R_inf) - np.log(50.0)) / 0.5) ** 2 - np.log(R_inf * 0.5)
    lp += -0.5 * ((np.log(dR)   - np.log(150.0)) / 0.5) ** 2 - np.log(dR * 0.5)
    lp += -0.5 * ((log_tau - (-12.0)) / 2.0) ** 2
    a_a, b_a = 8.0, 2.5
    lp += (a_a - 1.0) * np.log(alpha) + (b_a - 1.0) * np.log(1.0 - alpha)
    return lp


def _logpost_cole_cole(theta, f, Z_obs, snr_db):
    R_inf, dR, log_tau, alpha = theta
    lp = _prior_cole_cole(theta)
    if not np.isfinite(lp):
        return -np.inf
    tau = np.exp(log_tau)
    if R_inf <= 0 or dR <= 0 or tau <= 0:
        return -np.inf
    Z_pred = impedance_cole_cole(f, R_inf, dR, tau, alpha)
    ll = _log_likelihood_complex(Z_pred, Z_obs, snr_db)
    return lp + ll if np.isfinite(ll) else -np.inf


def _theta0_cole_cole(f, Z_obs, snr_db):
    from .ls_fitting import nlls_fit
    r = nlls_fit(f, Z_obs)
    if r.get("converged"):
        p = r["params"]
        return np.array([max(p["R_inf"], 1.0),
                         max(p["R0"] - p["R_inf"], 1.0),
                         np.log(max(p["tau"], 1e-10)),
                         np.clip(p["alpha"], 0.05, 0.98)])
    return np.array([50.0, 150.0, np.log(7.96e-6), 0.75])


MODEL_COLE_COLE = ImpedanceModel(
    name="Cole-Cole",
    short_name="M2_ColeCole",
    n_params=4,
    param_names=["R_inf", "delta_R", "log_tau", "alpha"],
    impedance_fn=lambda f, th: impedance_cole_cole(f, th[0], th[1], np.exp(th[2]), th[3]),
    log_prior_fn=_prior_cole_cole,
    theta0_fn=_theta0_cole_cole,
    bounds_lower=[0.1, 0.1, np.log(1e-9), 0.05],
    bounds_upper=[2000., 5000., np.log(1.0), 1.0],
    description="Modelo Cole-Cole padrão com α livre (4 parâmetros)."
)


# ---------------------------------------------------------------------------
# M3 — Double Cole-Cole
# ---------------------------------------------------------------------------

def _prior_double_cc(theta):
    R_inf, dR1, log_tau1, alpha1, dR2, log_tau2, alpha2 = theta
    if R_inf <= 0 or dR1 <= 0 or dR2 <= 0:
        return -np.inf
    if alpha1 <= 0 or alpha1 > 1.0 or alpha2 <= 0 or alpha2 > 1.0:
        return -np.inf
    lp = 0.0
    lp += -0.5 * ((np.log(R_inf) - np.log(50.0)) / 0.5) ** 2 - np.log(R_inf * 0.5)
    # Duas dispersões: ambas com prior LogNormal mas centros diferentes
    lp += -0.5 * ((np.log(dR1) - np.log(100.0)) / 0.6) ** 2 - np.log(dR1 * 0.6)
    lp += -0.5 * ((log_tau1 - (-12.0)) / 1.5) ** 2
    a_a, b_a = 8.0, 2.5
    lp += (a_a - 1.0) * np.log(alpha1) + (b_a - 1.0) * np.log(1.0 - alpha1)
    lp += -0.5 * ((np.log(dR2) - np.log(50.0)) / 0.6) ** 2 - np.log(dR2 * 0.6)
    lp += -0.5 * ((log_tau2 - (-14.0)) / 1.5) ** 2  # segunda dispersão em freq. mais alta
    lp += (a_a - 1.0) * np.log(alpha2) + (b_a - 1.0) * np.log(1.0 - alpha2)
    return lp


def _logpost_double_cc(theta, f, Z_obs, snr_db):
    lp = _prior_double_cc(theta)
    if not np.isfinite(lp):
        return -np.inf
    R_inf, dR1, log_tau1, alpha1, dR2, log_tau2, alpha2 = theta
    tau1, tau2 = np.exp(log_tau1), np.exp(log_tau2)
    Z_pred = impedance_double_cole_cole(f, R_inf, dR1, tau1, alpha1, dR2, tau2, alpha2)
    if not np.all(np.isfinite(Z_pred)):
        return -np.inf
    ll = _log_likelihood_complex(Z_pred, Z_obs, snr_db)
    return lp + ll if np.isfinite(ll) else -np.inf


def _theta0_double_cc(f, Z_obs, snr_db):
    # Inicializa com duas dispersões baseadas no Cole-Cole ajustado
    from .ls_fitting import nlls_fit
    r = nlls_fit(f, Z_obs)
    if r.get("converged"):
        p = r["params"]
        dR = max(p["R0"] - p["R_inf"], 2.0)
        return np.array([
            max(p["R_inf"], 1.0),
            dR * 0.7,                          # maior fração para 1ª dispersão
            np.log(max(p["tau"], 1e-10)),
            np.clip(p["alpha"], 0.05, 0.98),
            dR * 0.3,                          # menor fração para 2ª dispersão
            np.log(max(p["tau"], 1e-10)) - 2.0,  # tau2 menor (freq. mais alta)
            np.clip(p["alpha"] + 0.05, 0.05, 0.98)
        ])
    return np.array([50., 100., np.log(7.96e-6), 0.75, 50., np.log(1e-7), 0.80])


MODEL_DOUBLE_CC = ImpedanceModel(
    name="Double Cole-Cole",
    short_name="M3_DoubleCole",
    n_params=7,
    param_names=["R_inf", "dR1", "log_tau1", "alpha1", "dR2", "log_tau2", "alpha2"],
    impedance_fn=lambda f, th: impedance_double_cole_cole(
        f, th[0], th[1], np.exp(th[2]), th[3], th[4], np.exp(th[5]), th[6]),
    log_prior_fn=_prior_double_cc,
    theta0_fn=_theta0_double_cc,
    bounds_lower=[0.1, 0.1, np.log(1e-9), 0.05, 0.1, np.log(1e-9), 0.05],
    bounds_upper=[2000., 5000., np.log(1.0), 1.0, 5000., np.log(1.0), 1.0],
    description="Duas dispersões Cole-Cole independentes (7 parâmetros)."
)


# ---------------------------------------------------------------------------
# M4 — Randles + CPE
# ---------------------------------------------------------------------------

def _prior_randles(theta):
    Rs, Rp, log_Q, n = theta
    if Rs <= 0 or Rp <= 0 or n <= 0 or n > 1.0:
        return -np.inf
    lp = 0.0
    # Rs ~ LogNormal(ln(50), 0.5)
    lp += -0.5 * ((np.log(Rs) - np.log(50.0)) / 0.5) ** 2 - np.log(Rs * 0.5)
    # Rp ~ LogNormal(ln(200), 0.5)
    lp += -0.5 * ((np.log(Rp) - np.log(200.0)) / 0.5) ** 2 - np.log(Rp * 0.5)
    # log_Q ~ Normal(-10, 2) → Q ∈ [~10nS·s^n, ~10μS·s^n]
    lp += -0.5 * ((log_Q - (-10.0)) / 2.0) ** 2
    # n ~ Beta(8, 2.5) — similar ao alpha do Cole-Cole
    a_n, b_n = 8.0, 2.5
    lp += (a_n - 1.0) * np.log(n) + (b_n - 1.0) * np.log(1.0 - n)
    return lp


def _logpost_randles(theta, f, Z_obs, snr_db):
    lp = _prior_randles(theta)
    if not np.isfinite(lp):
        return -np.inf
    Rs, Rp, log_Q, n = theta
    Q = np.exp(log_Q)
    if Rs <= 0 or Rp <= 0 or Q <= 0 or n <= 0 or n > 1.0:
        return -np.inf
    Z_pred = impedance_randles_cpe(f, Rs, Rp, Q, n)
    if not np.all(np.isfinite(Z_pred)):
        return -np.inf
    ll = _log_likelihood_complex(Z_pred, Z_obs, snr_db)
    return lp + ll if np.isfinite(ll) else -np.inf


def _theta0_randles(f, Z_obs, snr_db):
    from .ls_fitting import nlls_fit
    r = nlls_fit(f, Z_obs)
    if r.get("converged"):
        p = r["params"]
        Rs  = max(p["R_inf"], 1.0)
        Rp  = max(p["R0"] - p["R_inf"], 1.0)
        tau = max(p["tau"], 1e-10)
        # Q ≈ tau / Rp^(1-n)  para n≈alpha
        n   = np.clip(p["alpha"], 0.05, 0.98)
        Q   = tau / (Rp ** (1.0 - n))
        return np.array([Rs, Rp, np.log(max(Q, 1e-15)), n])
    return np.array([50., 150., np.log(1e-8), 0.75])


MODEL_RANDLES_CPE = ImpedanceModel(
    name="Randles + CPE",
    short_name="M4_Randles",
    n_params=4,
    param_names=["Rs", "Rp", "log_Q", "n"],
    impedance_fn=lambda f, th: impedance_randles_cpe(f, th[0], th[1], np.exp(th[2]), th[3]),
    log_prior_fn=_prior_randles,
    theta0_fn=_theta0_randles,
    bounds_lower=[0.1, 0.1, np.log(1e-15), 0.05],
    bounds_upper=[2000., 5000., np.log(1e-2), 1.0],
    description="Randles com CPE: Rs + Rp||CPE (4 parâmetros, topologia diferente)."
)


# ---------------------------------------------------------------------------
# Registro de todos os modelos
# ---------------------------------------------------------------------------

ALL_MODELS = {
    "M1": MODEL_DEBYE,
    "M2": MODEL_COLE_COLE,
    "M3": MODEL_DOUBLE_CC,
    "M4": MODEL_RANDLES_CPE,
}


# ---------------------------------------------------------------------------
# Função auxiliar: log-likelihood ponto-a-ponto (para WAIC/LOO)
# ---------------------------------------------------------------------------

def pointwise_log_likelihood(model: ImpedanceModel,
                              samples: np.ndarray,
                              f: np.ndarray,
                              Z_obs: np.ndarray,
                              snr_db: float) -> np.ndarray:
    """
    Computa a log-verossimilhança ponto-a-ponto para cada amostra MCMC.

    Retorna
    -------
    ll_matrix : array de shape (n_samples, 2*n_freq)
                ll_matrix[s, i] = log p(Z_obs_i | θ_s)
    """
    n_samples = len(samples)
    n_data = 2 * len(f)   # Re e Im separados
    ll_matrix = np.full((n_samples, n_data), -np.inf)

    for s, theta in enumerate(samples):
        try:
            Z_pred = model.impedance_fn(f, theta)
            if np.all(np.isfinite(Z_pred)):
                ll_matrix[s] = _log_likelihood_pointwise(Z_pred, Z_obs, snr_db)
        except Exception:
            pass

    return ll_matrix
