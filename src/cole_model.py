"""
cole_model.py
=============
Implementação do modelo de Cole-Cole para bioimpedância.

Referência principal:
    Cole, K.S. & Cole, R.H. (1941). Dispersion and absorption in dielectrics I.
    Alternating current characteristics. J. Chem. Phys., 9(4), 341-351.

    Grimnes, S. & Martinsen, Ø.G. (2015). Bioimpedance and Bioelectricity Basics
    (3rd ed.). Academic Press. [Capítulos 1 e 2 — base da disciplina EEL410279]

Modelo Cole-Cole:
    Z(ω) = R_inf + (R0 - R_inf) / (1 + (jωτ)^α)

    Parâmetros:
        R_inf  : resistência a frequência infinita [Ω]
        R0     : resistência a frequência zero (DC) [Ω]
        tau    : tempo característico de relaxação [s]
        alpha  : parâmetro de depressão ∈ (0, 1]
                 α=1 → modelo Debye clássico
                 α<1 → distribuição de tempos de relaxação (mais realista para tecidos)

Faixas fisiológicas (tecido muscular):
    R_inf  ∈ [30,  80] Ω          (fluido extracelular)
    R0     ∈ [100, 300] Ω         (fluido extra + intracelular)
    tau    ∈ [1e-6, 1e-4] s       (dispersão β)
    alpha  ∈ [0.5,  0.9]          (heterogeneidade celular)

    Fontes: Gabriel et al. (1996), Physiol. Meas.; IT'IS Foundation database.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Modelo Cole-Cole (impedância complexa)
# ---------------------------------------------------------------------------

def cole_cole_impedance(f: np.ndarray,
                        R_inf: float,
                        R0: float,
                        tau: float,
                        alpha: float) -> np.ndarray:
    """
    Calcula a impedância complexa pelo modelo de Cole-Cole.

    Z(ω) = R_inf + (R0 - R_inf) / (1 + (jωτ)^α)

    Parâmetros
    ----------
    f      : array de frequências [Hz]
    R_inf  : resistência a freq. infinita [Ω]  (R_inf > 0)
    R0     : resistência DC [Ω]                (R0 > R_inf)
    tau    : tempo de relaxação [s]            (tau > 0)
    alpha  : parâmetro de depressão            (0 < alpha ≤ 1)

    Retorna
    -------
    Z : array complexo [Ω]
    """
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    delta_R = R0 - R_inf
    denominator = 1.0 + (1j * omega * tau) ** alpha
    return R_inf + delta_R / denominator


def cole_cole_modulus(f, R_inf, R0, tau, alpha):
    """Módulo de Z(ω) — usado no fitting por módulo (Grimnes & Martinsen, Cap.2)."""
    return np.abs(cole_cole_impedance(f, R_inf, R0, tau, alpha))


def cole_cole_phase_deg(f, R_inf, R0, tau, alpha):
    """Fase de Z(ω) em graus."""
    return np.angle(cole_cole_impedance(f, R_inf, R0, tau, alpha), deg=True)


def cole_cole_real(f, R_inf, R0, tau, alpha):
    """Parte real de Z(ω) — resistência R [Ω]."""
    return cole_cole_impedance(f, R_inf, R0, tau, alpha).real


def cole_cole_imag(f, R_inf, R0, tau, alpha):
    """Parte imaginária de Z(ω) — reatância X [Ω]."""
    return cole_cole_impedance(f, R_inf, R0, tau, alpha).imag


# ---------------------------------------------------------------------------
# Grandezas derivadas — relevantes para interpretação clínica
# ---------------------------------------------------------------------------

def characteristic_frequency(tau: float, alpha: float) -> float:
    """
    Frequência característica do pico de reatância [Hz].

    f_c = 1 / (2π · τ)

    Nota: esta é a frequência de relaxação central. Para α < 1, o pico
    real de |X| ocorre próximo mas não exatamente em f_c (Barsoukov & Macdonald, 2018).
    """
    return 1.0 / (2.0 * np.pi * tau)


def delta_R(R_inf: float, R0: float) -> float:
    """
    ΔR = R0 - R_inf: diferença de resistências.
    Relacionada ao volume intracelular e à integridade de membrana.
    Queda de ΔR indica lesão celular ou morte celular (edema, isquemia).
    """
    return R0 - R_inf


def membrane_capacitance_approx(tau: float, R_inf: float, R0: float) -> float:
    """
    Aproximação da capacitância de membrana para o modelo de Debye (α=1):
        C_m ≈ τ / (R_inf · R0 / (R0 - R_inf))
              = τ · ΔR / (R_inf · R0)

    Apenas indicativa — para α≠1 usar o CPE completo.
    """
    dR = delta_R(R_inf, R0)
    return tau * dR / (R_inf * R0)


# ---------------------------------------------------------------------------
# Validação de parâmetros
# ---------------------------------------------------------------------------

def validate_params(R_inf, R0, tau, alpha):
    """
    Verifica restrições físicas dos parâmetros do modelo Cole-Cole.
    Levanta ValueError se alguma restrição for violada.
    """
    errors = []
    if R_inf <= 0:
        errors.append(f"R_inf deve ser > 0, recebido: {R_inf:.4g}")
    if R0 <= R_inf:
        errors.append(f"R0 deve ser > R_inf, recebido R0={R0:.4g}, R_inf={R_inf:.4g}")
    if tau <= 0:
        errors.append(f"tau deve ser > 0, recebido: {tau:.4g}")
    if not (0 < alpha <= 1):
        errors.append(f"alpha deve estar em (0,1], recebido: {alpha:.4f}")
    if errors:
        raise ValueError("Parâmetros inválidos:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# Parâmetros de referência — tecido muscular (Gabriel et al., 1996)
# ---------------------------------------------------------------------------

TISSUE_PARAMS = {
    "musculo": {
        "R_inf": 50.0,    # Ω
        "R0":    200.0,   # Ω
        "tau":   7.96e-6, # s  → f_c ≈ 20 kHz
        "alpha": 0.75,
        "fonte": "Gabriel et al. (1996) — Phys. Med. Biol."
    },
    "gordura": {
        "R_inf": 200.0,
        "R0":    600.0,
        "tau":   1.59e-5,
        "alpha": 0.55,
        "fonte": "IT'IS Foundation Tissue Properties Database"
    },
    "sangue": {
        "R_inf": 60.0,
        "R0":    100.0,
        "tau":   1.06e-6,
        "alpha": 0.90,
        "fonte": "Grimnes & Martinsen (2015), Cap. 4"
    },
}
