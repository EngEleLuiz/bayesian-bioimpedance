"""
data_generation.py
==================
Geração de dados sintéticos de EIS com modelos de ruído realistas.

Modelo de ruído:
    A medição de bioimpedância é afetada por ruído em amplitude e fase
    (Srinivasan & Tran, 2021; Grimnes & Martinsen, 2015, Cap.7).

    Modelo adotado (ruído multiplicativo em módulo + aditivo em fase):
        |Z_med(f)| = |Z_true(f)| · (1 + ε_mod)    ε_mod ~ N(0, σ_mod²)
        φ_med(f)   = φ_true(f) + ε_fase            ε_fase ~ N(0, σ_fase²)

    Equivalentemente em componentes complexas:
        Z_med = (R + ε_R) + j(X + ε_X)
        com σ_R ≈ σ_X ≈ σ_noise · |Z|

    Esta escolha é consistente com instrumentos reais como AD5933/AD5941
    e com o modelo de ruído usado por Huang et al. (2021) e pelo grupo
    de Barsoukov & Macdonald (2018).

Referências:
    Barsoukov, E. & Macdonald, J.R. (2018). Impedance Spectroscopy: Theory,
    Experiment, and Applications (3rd ed.). Wiley.

    Huang, J., Papac, M. & O'Hayre, R. (2021). Towards robust autonomous
    impedance spectroscopy analysis: a calibrated hierarchical Bayesian
    approach for EIS inversion. Electrochim. Acta, 367, 137493.
"""

import numpy as np
from .cole_model import cole_cole_impedance, validate_params


# ---------------------------------------------------------------------------
# Grade de frequências padrão (100 Hz – 1 MHz, escala log)
# ---------------------------------------------------------------------------

def frequency_grid(f_min: float = 100.0,
                   f_max: float = 1e6,
                   n_points: int = 30,
                   log: bool = True) -> np.ndarray:
    """
    Retorna uma grade de frequências para EIS.

    Parâmetros
    ----------
    f_min    : frequência mínima [Hz]
    f_max    : frequência máxima [Hz]
    n_points : número de pontos
    log      : True → escala logarítmica (padrão em EIS)

    Retorna
    -------
    f : array [Hz]
    """
    if log:
        return np.logspace(np.log10(f_min), np.log10(f_max), n_points)
    return np.linspace(f_min, f_max, n_points)


# ---------------------------------------------------------------------------
# Conversão SNR → σ_noise
# ---------------------------------------------------------------------------

def snr_to_sigma(snr_db: float) -> float:
    """
    Converte SNR em dB para desvio padrão relativo σ.

    SNR (dB) = -20·log10(σ)  →  σ = 10^(-SNR/20)

    Exemplos:
        SNR=20 dB → σ ≈ 0.100 (10%)
        SNR=30 dB → σ ≈ 0.032 (3.2%)
        SNR=40 dB → σ ≈ 0.010 (1.0%)
    """
    return 10.0 ** (-snr_db / 20.0)


# ---------------------------------------------------------------------------
# Gerador principal de dados sintéticos
# ---------------------------------------------------------------------------

def generate_eis_data(f: np.ndarray,
                      R_inf: float,
                      R0: float,
                      tau: float,
                      alpha: float,
                      snr_db: float = 30.0,
                      seed: int = None) -> dict:
    """
    Gera espectro EIS sintético com ruído Gaussiano complexo.

    Ruído aplicado independentemente na parte real e imaginária:
        R_med(f) = R_true(f) + N(0, σ·|Z_true|)
        X_med(f) = X_true(f) + N(0, σ·|Z_true|)

    Esta escolha é motivada pelo modelo de ruído proporcional ao módulo,
    consistente com medições por AD5941 e LCR meters (Barsoukov & Macdonald, 2018).

    Parâmetros
    ----------
    f      : array de frequências [Hz]
    R_inf, R0, tau, alpha : parâmetros Cole-Cole
    snr_db : relação sinal-ruído em dB
    seed   : semente para reprodutibilidade

    Retorna
    -------
    dict com chaves:
        'f'        : frequências [Hz]
        'Z_true'   : impedância sem ruído (complexo)
        'Z_noisy'  : impedância com ruído (complexo)
        'sigma'    : desvio padrão relativo (fração de |Z|)
        'snr_db'   : SNR em dB
        'params'   : dicionário com parâmetros verdadeiros
    """
    validate_params(R_inf, R0, tau, alpha)
    rng = np.random.default_rng(seed)

    Z_true = cole_cole_impedance(f, R_inf, R0, tau, alpha)
    sigma = snr_to_sigma(snr_db)
    noise_scale = sigma * np.abs(Z_true)

    noise_R = rng.normal(0.0, noise_scale)
    noise_X = rng.normal(0.0, noise_scale)
    Z_noisy = (Z_true.real + noise_R) + 1j * (Z_true.imag + noise_X)

    return {
        "f":        f,
        "Z_true":   Z_true,
        "Z_noisy":  Z_noisy,
        "sigma":    sigma,
        "snr_db":   snr_db,
        "params":   dict(R_inf=R_inf, R0=R0, tau=tau, alpha=alpha),
    }


def generate_multiple_snr(f: np.ndarray,
                          R_inf: float,
                          R0: float,
                          tau: float,
                          alpha: float,
                          snr_levels_db: list = None,
                          n_realizations: int = 50,
                          seed: int = 42) -> dict:
    """
    Gera múltiplas realizações para diferentes níveis de SNR.
    Usado no estudo de sensibilidade (Etapa 4 do pipeline).

    Parâmetros
    ----------
    snr_levels_db   : lista de SNR [dB], ex: [15, 20, 25, 30, 35, 40]
    n_realizations  : número de realizações por nível de SNR

    Retorna
    -------
    dict: {snr_db: lista de dicts de dados}
    """
    if snr_levels_db is None:
        snr_levels_db = [15, 20, 25, 30, 35, 40]

    rng = np.random.default_rng(seed)
    result = {}
    for snr in snr_levels_db:
        realizations = []
        for i in range(n_realizations):
            seed_i = int(rng.integers(0, 2**31))
            data = generate_eis_data(f, R_inf, R0, tau, alpha,
                                     snr_db=snr, seed=seed_i)
            realizations.append(data)
        result[snr] = realizations
    return result
