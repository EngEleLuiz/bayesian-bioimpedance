"""
mcmc_sampler.py
===============
Amostrador MCMC — Metropolis-Hastings Adaptativo (AM-MCMC).

Algoritmo implementado: Adaptive Metropolis (AM) de Haario et al. (2001).
A proposta se adapta à covariância empírica das amostras aceitas, acelerando
a convergência e evitando o ajuste manual de hiperparâmetros de proposta.

Referências fundamentais:
    Haario, H., Saksman, E. & Tamminen, J. (2001). An adaptive Metropolis
    algorithm. Bernoulli, 7(2), 223-242.

    Gelman, A., Carlin, J.B., Stern, H.S. et al. (2013). Bayesian Data
    Analysis (3rd ed.). CRC Press. — Capítulos 11-12 (MCMC).

    Huang, J., Papac, M. & O'Hayre, R. (2021). Towards robust autonomous
    impedance spectroscopy analysis: a calibrated hierarchical Bayesian
    approach for EIS inversion. Electrochim. Acta, 367, 137493.

    Žnidarič, L. et al. (2021). Evaluating uncertainties in electrochemical
    impedance spectra of solid oxide fuel cells. arXiv:2101.08049.

Modelo probabilístico:
    Prior (baseado em faixas fisiológicas de Gabriel et al., 1996):
        R_inf  ~ LogNormal(μ=ln(50),  σ=0.5)    [Ω]
        ΔR     ~ LogNormal(μ=ln(150), σ=0.5)    [Ω]   → R0 = R_inf + ΔR
        log_τ  ~ Normal(μ=-12, σ=2.0)           [ln(s)]
        α      ~ Beta(a=8, b=2.5)               ≈ concentrado em ~0.76

    Verossimilhança (ruído Gaussiano complexo):
        R_obs(f) ~ N(R_true(f), σ(f)²)
        X_obs(f) ~ N(X_true(f), σ(f)²)
        com σ(f) = σ_noise · |Z_true(f)|  (ruído proporcional ao módulo)

    Posterior (por Bayes):
        p(θ | Z_obs) ∝ p(Z_obs | θ) · p(θ)
"""

import numpy as np
from .cole_model import cole_cole_impedance, validate_params
from .data_generation import snr_to_sigma


# ---------------------------------------------------------------------------
# Log-prior (log-escala para estabilidade numérica)
# ---------------------------------------------------------------------------

def log_prior(R_inf: float, delta_R: float, log_tau: float, alpha: float) -> float:
    """
    Log-densidade a priori conjunta.

    Parametrização:
        θ = (R_inf, ΔR=R0-R_inf, log_tau, alpha)

    Priors escolhidos para refletir conhecimento fisiológico:
        - LogNormal para R_inf e ΔR: garante positividade e R0 > R_inf
        - Normal para log_tau: escala logarítmica natural para constantes de tempo
        - Beta para alpha: concentrado em 0.5-1.0, modo ~0.75

    Retorna -inf se os parâmetros violam restrições físicas.
    """
    # --- Restrições físicas rígidas ---
    if R_inf <= 0 or delta_R <= 0 or alpha <= 0 or alpha > 1.0:
        return -np.inf
    tau = np.exp(log_tau)
    if tau <= 0:
        return -np.inf

    lp = 0.0

    # R_inf ~ LogNormal(ln(50), 0.5²)  → modo ≈ 45 Ω, P(10<R_inf<200) ≈ 0.96
    mu_Rinf, sigma_Rinf = np.log(50.0), 0.5
    lp += -0.5 * ((np.log(R_inf) - mu_Rinf) / sigma_Rinf) ** 2 - np.log(R_inf * sigma_Rinf)

    # ΔR ~ LogNormal(ln(150), 0.5²)  → modo ≈ 136 Ω
    mu_dR, sigma_dR = np.log(150.0), 0.5
    lp += -0.5 * ((np.log(delta_R) - mu_dR) / sigma_dR) ** 2 - np.log(delta_R * sigma_dR)

    # log_tau ~ Normal(-12.0, 2.0²)  → τ ∈ [1ns, 1ms] com alta probabilidade
    mu_lt, sigma_lt = -12.0, 2.0
    lp += -0.5 * ((log_tau - mu_lt) / sigma_lt) ** 2

    # alpha ~ Beta(8, 2.5)  → modo = (8-1)/(8+2.5-2) = 7/8.5 ≈ 0.82
    a_alpha, b_alpha = 8.0, 2.5
    lp += (a_alpha - 1.0) * np.log(alpha) + (b_alpha - 1.0) * np.log(1.0 - alpha)

    return lp


# ---------------------------------------------------------------------------
# Log-verossimilhança
# ---------------------------------------------------------------------------

def log_likelihood(R_inf: float, delta_R_val: float, log_tau: float,
                   alpha: float, f: np.ndarray,
                   Z_obs: np.ndarray, snr_db: float) -> float:
    """
    Log-verossimilhança Gaussiana complexa.

    L(θ) = Σ_f [ -ln(σ(f)) - (R_obs - R_pred)²/(2σ²) ]
           + Σ_f [ -ln(σ(f)) - (X_obs - X_pred)²/(2σ²) ]

    onde σ(f) = σ_noise · |Z_pred(f)|  (ruído proporcional ao módulo).

    Nota: usamos σ baseado no modelo (σ proporcional a |Z_pred|), não em
    |Z_obs|, para evitar heterocedasticidade mal-especificada no fitting.
    """
    tau = np.exp(log_tau)
    R0 = R_inf + delta_R_val
    try:
        validate_params(R_inf, R0, tau, alpha)
    except ValueError:
        return -np.inf

    Z_pred = cole_cole_impedance(f, R_inf, R0, tau, alpha)
    sigma_noise = snr_to_sigma(snr_db)
    sigma_f = sigma_noise * np.abs(Z_pred)  # σ(f): ruído proporcional

    # Evitar divisão por zero
    sigma_f = np.maximum(sigma_f, 1e-12)

    res_R = Z_obs.real - Z_pred.real
    res_X = Z_obs.imag - Z_pred.imag

    ll = -np.sum(np.log(sigma_f)) - 0.5 * np.sum((res_R / sigma_f) ** 2)
    ll += -np.sum(np.log(sigma_f)) - 0.5 * np.sum((res_X / sigma_f) ** 2)
    return ll


# ---------------------------------------------------------------------------
# Log-posterior
# ---------------------------------------------------------------------------

def log_posterior(theta: np.ndarray, f: np.ndarray,
                  Z_obs: np.ndarray, snr_db: float) -> float:
    """
    Log-posterior não normalizado: log p(θ|Z_obs) ∝ log L + log prior.

    θ = [R_inf, delta_R, log_tau, alpha]
    """
    R_inf, dR, log_tau, alpha = theta
    lp = log_prior(R_inf, dR, log_tau, alpha)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(R_inf, dR, log_tau, alpha, f, Z_obs, snr_db)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ---------------------------------------------------------------------------
# Amostrador Metropolis-Hastings Adaptativo (Haario et al., 2001)
# ---------------------------------------------------------------------------

class AdaptiveMCMC:
    """
    Metropolis-Hastings com proposta adaptativa (AM-MCMC).

    Fase 1 (burn-in inicial): proposta diagonal fixa com escala s0.
    Fase 2 (adaptação):       proposta atualizada a cada `adapt_interval`
                              passos usando a covariância empírica.

    Escala ótima de Gelman (1996): s = 2.38²/d  para distribuições Gaussianas.
    """

    def __init__(self,
                 n_samples: int = 5000,
                 n_warmup:  int = 2000,
                 adapt_start: int = 500,
                 adapt_interval: int = 100,
                 seed: int = 42):
        """
        Parâmetros
        ----------
        n_samples     : amostras totais (após warm-up)
        n_warmup      : amostras de warm-up (descartadas)
        adapt_start   : passo em que inicia a adaptação da proposta
        adapt_interval: intervalo de atualização da covariância
        seed          : semente para reprodutibilidade
        """
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.adapt_start = adapt_start
        self.adapt_interval = adapt_interval
        self.rng = np.random.default_rng(seed)

    def _initial_proposal_cov(self, theta0: np.ndarray) -> np.ndarray:
        """
        Covariância diagonal inicial genérica para qualquer dimensão.
        Heurística: 5% do valor absoluto de cada parâmetro (mínimo 0.01).
        """
        scales = np.maximum(np.abs(theta0) * 0.05, 0.01)
        return np.diag(scales ** 2)

    def run(self, log_post_fn, theta0: np.ndarray,
            verbose: bool = True) -> dict:
        """
        Executa o amostrador AM-MCMC.

        Parâmetros
        ----------
        log_post_fn : função log_posterior(theta) → float
        theta0      : ponto inicial [R_inf, delta_R, log_tau, alpha]
        verbose     : imprime progresso

        Retorna
        -------
        dict com:
            'samples'      : array (n_samples, 4) — amostras pós warm-up
            'warmup'       : array (n_warmup, 4)  — amostras de warm-up
            'log_post'     : log-posterior de cada amostra
            'accept_rate'  : taxa de aceitação global
            'accept_warmup': taxa de aceitação no warm-up
            'final_cov'    : covariância final da proposta
        """
        d = len(theta0)
        n_total = self.n_warmup + self.n_samples

        # Escala ótima de Gelman et al. (1996) para RW-MH em d dimensões
        sd = (2.38 ** 2) / d

        # Proposta inicial
        Sigma_prop = self._initial_proposal_cov(theta0)

        # Buffers
        all_samples = np.zeros((n_total, d))
        all_log_post = np.zeros(n_total)

        theta_curr = np.array(theta0, dtype=float)
        lp_curr = log_post_fn(theta_curr)

        if not np.isfinite(lp_curr):
            raise ValueError(
                f"Log-posterior inválido no ponto inicial θ0={theta0}. "
                "Verifique os priors e a inicialização."
            )

        n_accept = 0
        all_samples[0] = theta_curr
        all_log_post[0] = lp_curr

        # Running mean e covariância para adaptação
        run_mean = theta_curr.copy()
        eps = 1e-6 * np.eye(d)   # regularização numérica

        for i in range(1, n_total):
            # --- Proposta ---
            theta_prop = self.rng.multivariate_normal(theta_curr, sd * Sigma_prop)

            # --- Aceitação Metropolis ---
            lp_prop = log_post_fn(theta_prop)
            log_alpha_accept = lp_prop - lp_curr

            if np.log(self.rng.uniform()) < log_alpha_accept:
                theta_curr = theta_prop
                lp_curr = lp_prop
                n_accept += 1

            all_samples[i] = theta_curr
            all_log_post[i] = lp_curr

            # --- Adaptação da covariância (Haario et al., 2001) ---
            if i >= self.adapt_start and i % self.adapt_interval == 0:
                recent = all_samples[max(0, i - 2000):i]
                run_mean = recent.mean(axis=0)
                centered = recent - run_mean
                Sigma_prop = (centered.T @ centered) / len(recent) + eps

            # Progresso
            if verbose and (i % (n_total // 5) == 0):
                ar = n_accept / i * 100
                phase = "warm-up" if i < self.n_warmup else "amostragem"
                print(f"  [{phase}] passo {i:5d}/{n_total} | "
                      f"aceitação: {ar:.1f}% | "
                      f"log-post: {lp_curr:.2f}")

        warmup_samples = all_samples[:self.n_warmup]
        post_samples = all_samples[self.n_warmup:]
        accept_warmup = (np.diff(warmup_samples, axis=0) != 0).any(axis=1).mean()
        accept_post = (np.diff(post_samples, axis=0) != 0).any(axis=1).mean()

        return {
            "samples":       post_samples,
            "warmup":        warmup_samples,
            "log_post":      all_log_post[self.n_warmup:],
            "accept_rate":   accept_post,
            "accept_warmup": accept_warmup,
            "final_cov":     Sigma_prop,
            "n_samples":     self.n_samples,
            "n_warmup":      self.n_warmup,
        }


# ---------------------------------------------------------------------------
# Ponto inicial automático (MAP aproximado via NLLS)
# ---------------------------------------------------------------------------

def initial_theta_from_nlls(f, Z_obs, snr_db):
    """
    Usa o resultado NLLS como ponto de partida para o MCMC.
    Isso melhora substancialmente a convergência (inicializar no MAP).

    Retorna θ0 = [R_inf, delta_R, log_tau, alpha]
    """
    from .ls_fitting import nlls_fit
    result = nlls_fit(f, Z_obs, method="complex")
    if result.get("converged") and result.get("params"):
        p = result["params"]
        R_inf  = max(p["R_inf"], 1.0)
        dR     = max(p["R0"] - p["R_inf"], 1.0)
        log_tau = np.log(max(p["tau"], 1e-10))
        alpha  = np.clip(p["alpha"], 0.05, 0.98)
        return np.array([R_inf, dR, log_tau, alpha])
    else:
        # Fallback: valores fisiológicos genéricos
        return np.array([50.0, 150.0, np.log(7.96e-6), 0.75])


# ---------------------------------------------------------------------------
# Diagnósticos de convergência MCMC
# ---------------------------------------------------------------------------

def compute_ess(chain: np.ndarray) -> float:
    """
    Tamanho efetivo da amostra (ESS) via autocorrelação.

    ESS = N / (1 + 2·Σ_k ρ_k)

    Trunca quando a autocorrelação cruza zero (Geyer, 1992).
    Referência: Gelman et al. (2013), BDA3, Cap. 11.
    """
    n = len(chain)
    chain_centered = chain - chain.mean()
    acf = np.correlate(chain_centered, chain_centered, mode="full")
    acf = acf[n-1:]
    acf /= acf[0]

    # Truncar na primeira autocorrelação negativa
    cutoff = np.argmax(acf < 0)
    if cutoff == 0:
        cutoff = n

    ess = n / (1.0 + 2.0 * acf[1:cutoff].sum())
    return min(ess, float(n))


def compute_rhat(chains: list) -> np.ndarray:
    """
    R-hat de Gelman-Rubin para múltiplas cadeias (diagnóstico de convergência).

    R-hat < 1.01 indica convergência satisfatória (Vehtari et al., 2021).

    Referência:
        Vehtari, A. et al. (2021). Rank-normalization, folding, and
        localization: An improved R-hat for assessing convergence of MCMC.
        Bayesian Analysis, 16(2), 667-718.

    chains : lista de arrays (n_samples, n_params)
    """
    M = len(chains)
    N = chains[0].shape[0]
    n_params = chains[0].shape[1]
    rhat = np.zeros(n_params)

    for p in range(n_params):
        chain_means = np.array([c[:, p].mean() for c in chains])
        chain_vars  = np.array([c[:, p].var(ddof=1) for c in chains])
        grand_mean  = chain_means.mean()

        B = N / (M - 1) * np.sum((chain_means - grand_mean) ** 2)
        W = chain_vars.mean()
        var_plus = (N - 1) / N * W + B / N
        rhat[p] = np.sqrt(var_plus / W) if W > 0 else np.nan

    return rhat


# ---------------------------------------------------------------------------
# Prior adaptativo (Empirical Bayes) para dados reais
# ---------------------------------------------------------------------------

def _empirical_bayes_log_posterior(theta: np.ndarray,
                                    f: np.ndarray,
                                    Z_obs: np.ndarray,
                                    snr_db: float,
                                    theta_nlls: np.ndarray) -> float:
    """
    Log-posterior com prior adaptativo centrado na estimativa NLLS.

    Útil para dados reais onde os parâmetros podem estar fora do
    prior fixo (ex.: tecidos com R_inf >> 50Ω como gordura).

    Prior: LogNormal centrado em theta_nlls com σ_log=0.7 para cada param.
    Isso cobre ±2x ao redor do ponto NLLS com 95% de probabilidade,
    acomodando qualquer tipo de tecido biologicamente razoável.
    """
    R_inf, dR, log_tau, alpha = theta
    if R_inf <= 0 or dR <= 0 or alpha <= 0 or alpha > 1.0:
        return -np.inf

    # Prior LogNormal adaptativo (sigma_log = 0.7 para R_inf, dR, tau)
    sig = 0.7  # largo o suficiente para todos os tecidos
    lp = 0.0
    # R_inf
    mu_Rinf = np.log(max(theta_nlls[0], 1.))
    lp += -0.5 * ((np.log(R_inf) - mu_Rinf) / sig) ** 2 - np.log(R_inf * sig)
    # delta_R
    mu_dR = np.log(max(theta_nlls[1], 1.))
    lp += -0.5 * ((np.log(dR) - mu_dR) / sig) ** 2 - np.log(dR * sig)
    # log_tau
    mu_lt = theta_nlls[2]
    lp += -0.5 * ((log_tau - mu_lt) / 1.5) ** 2
    # alpha Beta(8, 2.5) — igual ao prior base
    a_a, b_a = 8.0, 2.5
    lp += (a_a - 1.0) * np.log(alpha) + (b_a - 1.0) * np.log(1.0 - alpha)

    if not np.isfinite(lp):
        return -np.inf

    from .data_generation import snr_to_sigma
    from .cole_model import cole_cole_impedance
    tau = np.exp(log_tau)
    R0 = R_inf + dR
    try:
        Z_pred = cole_cole_impedance(f, R_inf, R0, tau, alpha)
        sigma_noise = snr_to_sigma(snr_db)
        sigma_f = np.maximum(sigma_noise * np.abs(Z_pred), 1e-12)
        res_R = Z_obs.real - Z_pred.real
        res_X = Z_obs.imag - Z_pred.imag
        ll = (-np.sum(np.log(sigma_f)) - 0.5 * np.sum((res_R / sigma_f) ** 2)
              - np.sum(np.log(sigma_f)) - 0.5 * np.sum((res_X / sigma_f) ** 2))
        return lp + ll if np.isfinite(ll) else -np.inf
    except Exception:
        return -np.inf
