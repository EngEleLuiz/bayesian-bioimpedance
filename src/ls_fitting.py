"""
ls_fitting.py
=============
Ajuste por Mínimos Quadrados Não-Lineares (NLLS) — referência comparativa.

O NLLS é o método padrão na literatura de EIS (Barsoukov & Macdonald, 2018;
Grimnes & Martinsen, 2015). Ele fornece estimativas pontuais e incertezas
via matriz de covariância da Hessiana local — válidas somente no regime
de ruído Gaussiano pequeno e modelo corretamente especificado.

Comparação com Bayes:
    NLLS fornece: θ_MAP (estimativa de ponto único)
    Bayes fornece: p(θ|Z_obs) (distribuição completa posterior)

    A diferença é mais pronunciada quando:
    - SNR é baixo (não-linearidades do posterior)
    - Parâmetros são correlacionados (ex.: R0 e tau)
    - O modelo é ligeiramente mal-especificado

Referências:
    Macdonald, J.R. & Potter Jr, L.D. (1987). A flexible procedure for
    analyzing impedance spectroscopy results. Solid State Ionics, 24(1), 61-79.

    Huang, J. (2023). Generalized distribution of relaxation times and its
    application to Cole-Cole fitting. J. Electrochem. Soc., 170(3).
"""

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.linalg import inv
from .cole_model import (cole_cole_impedance, cole_cole_modulus,
                         characteristic_frequency, delta_R)


# ---------------------------------------------------------------------------
# Funções de custo para NLLS
# ---------------------------------------------------------------------------

def _residuals_complex(params, f, Z_obs):
    """
    Resíduos no espaço complexo: [R_residuais, X_residuais].
    Fitting simultâneo de parte real e imaginária — mais robusto
    do que fitting apenas do módulo (Srinivasan & Sanchez, 2021).
    """
    R_inf, R0, log_tau, alpha = params
    tau = np.exp(log_tau)
    Z_pred = cole_cole_impedance(f, R_inf, R0, tau, alpha)
    res_R = Z_obs.real - Z_pred.real
    res_X = Z_obs.imag - Z_pred.imag
    return np.concatenate([res_R, res_X])


def _residuals_modulus(params, f, Z_obs_mod):
    """Resíduos apenas no módulo — útil quando a fase não é confiável."""
    R_inf, R0, log_tau, alpha = params
    tau = np.exp(log_tau)
    Z_pred_mod = cole_cole_modulus(f, R_inf, R0, tau, alpha)
    return Z_obs_mod - Z_pred_mod


# ---------------------------------------------------------------------------
# Estimativa inicial automática (inicialização robusta)
# ---------------------------------------------------------------------------

def _initial_guess(f, Z_obs):
    """
    Estimativa inicial a partir de propriedades geométricas do espectro.

    - R_inf  ≈ min(Re[Z])  (resistência em alta frequência)
    - R0     ≈ max(Re[Z])  (resistência em baixa frequência)
    - tau    ≈ 1/(2π·f_pico_Im), onde f_pico é onde Im[Z] é mínimo
    - alpha  = 0.7         (valor neutro fisiologicamente razoável)
    """
    R_inf_guess = np.percentile(Z_obs.real, 5)   # 5th percentile ≈ R_inf
    R0_guess    = np.percentile(Z_obs.real, 95)  # 95th percentile ≈ R0
    R_inf_guess = max(R_inf_guess, 1.0)
    R0_guess    = max(R0_guess, R_inf_guess + 10.0)

    # Frequência do pico da reatância (Im[Z] mais negativo)
    idx_peak = np.argmin(Z_obs.imag)
    f_peak = f[idx_peak]
    tau_guess = 1.0 / (2.0 * np.pi * f_peak) if f_peak > 0 else 1e-5
    tau_guess = np.clip(tau_guess, 1e-8, 1e-2)

    alpha_guess = 0.7
    return [R_inf_guess, R0_guess, np.log(tau_guess), alpha_guess]


# ---------------------------------------------------------------------------
# Ajuste NLLS principal
# ---------------------------------------------------------------------------

def nlls_fit(f: np.ndarray,
             Z_obs: np.ndarray,
             method: str = "complex",
             n_restarts: int = 5,
             seed: int = 0) -> dict:
    """
    Ajuste NLLS do modelo Cole-Cole por Levenberg-Marquardt.

    Parâmetros
    ----------
    f        : frequências [Hz]
    Z_obs    : impedância observada (array complexo)
    method   : 'complex' (Re+Im) ou 'modulus' (|Z|)
    n_restarts: número de reinicializações aleatórias (robustez global)
    seed     : semente para reprodutibilidade

    Retorna
    -------
    dict com:
        'params'   : {'R_inf', 'R0', 'tau', 'alpha'}
        'std'      : desvios padrão (da Hessiana local)
        'Z_fitted' : impedância ajustada
        'residuals': resíduos finais
        'chi2'     : χ² normalizado
        'converged': bool
    """
    rng = np.random.default_rng(seed)

    # Limites físicos dos parâmetros: [R_inf, R0, log_tau, alpha]
    lower = [0.1,    1.0,    np.log(1e-8), 0.05]
    upper = [2000.0, 5000.0, np.log(1.0),  1.0 ]

    best_cost = np.inf
    best_result = None

    for restart in range(n_restarts):
        if restart == 0:
            x0 = _initial_guess(f, Z_obs)
        else:
            # Perturbação aleatória da estimativa inicial
            x0 = _initial_guess(f, Z_obs)
            x0[0] *= rng.uniform(0.8, 1.2)
            x0[1] *= rng.uniform(0.8, 1.2)
            x0[2] += rng.uniform(-0.5, 0.5)
            x0[3] = rng.uniform(0.5, 0.95)

        # Clampar dentro dos limites
        x0 = np.clip(x0, [l + 1e-6 for l in lower],
                         [u - 1e-6 for u in upper])

        if method == "complex":
            fun = lambda p: _residuals_complex(p, f, Z_obs)
        else:
            fun = lambda p: _residuals_modulus(p, f, np.abs(Z_obs))

        try:
            res = least_squares(fun, x0,
                                bounds=(lower, upper),
                                method="trf",
                                ftol=1e-12, xtol=1e-12, gtol=1e-12,
                                max_nfev=10000)
            if res.cost < best_cost:
                best_cost = res.cost
                best_result = res
        except Exception:
            continue

    if best_result is None:
        return {"converged": False}

    res = best_result
    R_inf_fit, R0_fit, log_tau_fit, alpha_fit = res.x
    tau_fit = np.exp(log_tau_fit)

    # --- Incerteza via Hessiana (propagação delta method para tau) ---
    try:
        J = res.jac
        cov_log = inv(J.T @ J) * (2.0 * res.cost / max(len(res.fun) - 4, 1))
        std_log = np.sqrt(np.diag(cov_log))
        # Propagação para tau: std(tau) = tau * std(log_tau)
        std_params = {
            "R_inf":  std_log[0],
            "R0":     std_log[1],
            "tau":    tau_fit * std_log[2],
            "alpha":  std_log[3],
        }
    except Exception:
        std_params = {"R_inf": np.nan, "R0": np.nan, "tau": np.nan, "alpha": np.nan}

    Z_fitted = cole_cole_impedance(f, R_inf_fit, R0_fit, tau_fit, alpha_fit)
    residuals = np.concatenate([Z_obs.real - Z_fitted.real,
                                Z_obs.imag - Z_fitted.imag])
    chi2 = np.sum(residuals**2) / (2 * len(f) - 4)

    params_fit = dict(R_inf=R_inf_fit, R0=R0_fit, tau=tau_fit, alpha=alpha_fit)

    return {
        "params":    params_fit,
        "std":       std_params,
        "Z_fitted":  Z_fitted,
        "residuals": residuals,
        "chi2":      chi2,
        "converged": res.success or res.cost < 1.0,
        "derived": {
            "f_c":    characteristic_frequency(tau_fit, alpha_fit),
            "delta_R": delta_R(R_inf_fit, R0_fit),
        }
    }
