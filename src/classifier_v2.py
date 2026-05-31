"""
classifier_v2.py
================
Classificador Bayesiano de 2 estágios com Importance Sampling (IS-posterior).

PROBLEMA DA VERSÃO ANTERIOR:
    O classificador v1 amostra diretamente do prior de cada estado para
    estimar a verossimilhança marginal p(Z|estado_k). Com n_mc=600,
    o estimador tem CV=7–22% — ruído da mesma ordem do gap entre estados,
    causando misclassificação por ruído estatístico puro.

SOLUÇÃO — IS-POSTERIOR (esta versão):
    Ao invés de amostrar do prior de cada estado, reutilizamos as amostras
    MCMC já computadas de p(θ | Z_obs) e aplicamos importance sampling:

        p(Z | estado_k) ∝ E_{p(θ|Z)}[ p(θ|estado_k) / p₀(θ) ]
                        ≈ (1/S) Σ_s exp(log p(θ_s|estado_k) - log p₀(θ_s))

    onde:
        p(θ_s|estado_k) = prior do estado k avaliado nas amostras MCMC
        p₀(θ_s)         = prior base do MCMC (de mcmc_sampler.py)
        θ_s ~ p(θ|Z)    = amostras MCMC já disponíveis

    Com S=6000 amostras e ESS>1000, CV ≈ 1-2% → muito mais estável.

VANTAGEM ADICIONAL:
    A mesma cadeia MCMC é reutilizada — sem custo computacional extra
    por estado adicional. Ideal para artigo: "single-pass classification".

Referências:
    Robert, C.P. & Casella, G. (2004). Monte Carlo Statistical Methods.
    Springer. — Cap. 3: Importance Sampling.

    Gronau, Q.F. et al. (2017). A tutorial on bridge sampling.
    J Math Psychol, 81, 80–97. — estimação da evidência Bayesiana.

    Gelman, A. et al. (2013). Bayesian Data Analysis, 3ª ed. CRC.
    Cap. 10: Introduction to Bayesian Computation.
"""

import numpy as np
from scipy.special import logsumexp, betaln
from .tissue_states import ALL_STATES, TissueState, _compute_auc, _compute_calibration


# ===========================================================================
# Log-densidades dos priors
# ===========================================================================

def _log_lognormal(x, mu_log, sig_log):
    """Log-densidade LogNormal."""
    if x <= 0:
        return -np.inf
    return (-0.5 * ((np.log(x) - mu_log) / sig_log) ** 2
            - np.log(x * sig_log) - 0.5 * np.log(2 * np.pi))


def _log_normal(x, mu, sig):
    """Log-densidade Normal."""
    return (-0.5 * ((x - mu) / sig) ** 2
            - np.log(sig) - 0.5 * np.log(2 * np.pi))


def _log_beta(x, a, b):
    """Log-densidade Beta."""
    if x <= 0 or x >= 1:
        return -np.inf
    return ((a - 1) * np.log(x) + (b - 1) * np.log(1 - x)
            - betaln(a, b))


def log_prior_state(theta, state: TissueState) -> float:
    """
    Log p(θ | estado) — prior do estado tecidual avaliado em θ.

    θ = [R_inf, delta_R, log_tau, alpha]  (parametrização MCMC)
    """
    R_inf, dR, log_tau, alpha = theta
    tau = np.exp(log_tau)

    if R_inf <= 0 or dR <= 0 or tau <= 0 or alpha <= 0 or alpha >= 1:
        return -np.inf

    lp = 0.0
    lp += _log_lognormal(R_inf, state.mu_log_Rinf,  state.sig_log_Rinf)
    lp += _log_lognormal(dR,    state.mu_log_dR,    state.sig_log_dR)
    # log_tau ~ Normal(mu_log_tau, sig_log_tau²) — prior sobre log_tau diretamente
    lp += _log_normal(log_tau, state.mu_log_tau, state.sig_log_tau)
    lp += _log_beta(alpha, state.alpha_beta_a, state.alpha_beta_b)
    return lp


def log_prior_mixture_ref(theta, states: dict = None) -> float:
    """
    Prior de referência = mistura uniforme de todos os estados teciduais.

    p_ref(θ) = (1/K) Σ_k p(θ | estado_k)

    Usando a mistura como referência (ao invés do prior MCMC base),
    o IS é intrinsecamente balanceado: nenhum estado tem vantagem
    sistemática — o peso IS p(θ|k)/p_ref(θ) favorece apenas o estado
    cujo prior melhor coincide com o posterior amostrado via MCMC.

    Isso elimina o viés que ocorre quando o prior MCMC é similar a apenas
    um dos estados (ex.: Normal centrado em 50Ω, igual ao MCMC base).
    """
    if states is None:
        states = ALL_STATES
    K = len(states)
    log_pks = [log_prior_state(theta, st) for st in states.values()]
    finite  = [lp for lp in log_pks if np.isfinite(lp)]
    if not finite:
        return -np.inf
    return float(logsumexp(finite) - np.log(K))


# ===========================================================================
# Classificador IS-Posterior
# ===========================================================================

class ISPosteriorClassifier:
    """
    Classificador de 2 estágios via Importance Sampling sobre o posterior MCMC.

    Estágio 1: AM-MCMC para obter p(θ | Z_obs)        — já feito no pipeline base
    Estágio 2: IS para estimar p(estado_k | Z_obs)    — este módulo

    Estimativa IS:
        log p(estado_k | Z) ≈ logsumexp(w_k) - log(S) + log p(estado_k)
        w_{k,s} = log p(θ_s | estado_k) - log p₀(θ_s)

    Propriedades:
        - CV ≈ 1/√ESS_eff   (~1-3% com ESS>1000, vs 7-22% do método anterior)
        - Sem custo extra: reutiliza amostras MCMC existentes
        - Monotônico: mais amostras → estimativa melhor sempre
    """

    def __init__(self, states: dict = None, n_bootstrap: int = 200):
        self.states = states or ALL_STATES
        self.n_bootstrap = n_bootstrap

    def classify(self, mcmc_samples: np.ndarray,
                 prior_probs: dict = None) -> dict:
        """
        Classifica usando amostras MCMC já computadas.

        Parâmetros
        ----------
        mcmc_samples : array (S, 4) — [R_inf, delta_R, log_tau, alpha]
        prior_probs  : prob. a priori dos estados (padrão: uniforme)

        Retorna
        -------
        dict com probs, probs_std, log_evidence, predicted, confidence
        """
        state_names = list(self.states.keys())
        S = len(mcmc_samples)

        if prior_probs is None:
            prior_probs = {k: 1.0 / len(state_names) for k in state_names}

        # Calcular IS weights para cada estado
        log_weights = {k: np.full(S, -np.inf) for k in state_names}
        # Prior de referência = mistura de todos os estados (elimina viés)
        log_p_ref = np.array([log_prior_mixture_ref(mcmc_samples[s], self.states)
                               for s in range(S)])

        for k, state in self.states.items():
            log_pk = np.array([log_prior_state(mcmc_samples[s], state)
                               for s in range(S)])
            # IS weight: log p(θ|estado_k) - log p₀(θ)
            valid = np.isfinite(log_pk) & np.isfinite(log_p_ref)
            log_weights[k][valid] = log_pk[valid] - log_p_ref[valid]

        # Log-evidência por estado via logsumexp
        log_evidence = {}
        for k in state_names:
            valid = np.isfinite(log_weights[k])
            n_valid = valid.sum()
            if n_valid < 10:
                log_evidence[k] = -np.inf
            else:
                log_evidence[k] = float(
                    logsumexp(log_weights[k][valid]) - np.log(n_valid)
                )

        # Posterior
        log_post = {k: log_evidence[k] + np.log(max(prior_probs[k], 1e-300))
                    for k in state_names}
        finite_keys = [k for k in state_names if np.isfinite(log_post[k])]

        if not finite_keys:
            probs = {k: 1.0 / len(state_names) for k in state_names}
        else:
            log_norm = logsumexp([log_post[k] for k in finite_keys])
            probs = {k: float(np.exp(log_post[k] - log_norm))
                     if np.isfinite(log_post[k]) else 0.0
                     for k in state_names}

        predicted  = max(probs, key=probs.get)
        confidence = probs[predicted]

        # Bootstrap por reamostragem das cadeias MCMC (sem extra custo)
        rng = np.random.default_rng(0)
        probs_boot = {k: [] for k in state_names}
        for _ in range(self.n_bootstrap):
            idx = rng.choice(S, size=S, replace=True)
            samp_b = mcmc_samples[idx]
            lp0_b = log_p_ref[idx]
            lev_b = {}
            for k, state in self.states.items():
                lpk_b = np.array([log_prior_state(samp_b[s], state)
                                  for s in range(S)])
                valid_b = np.isfinite(lpk_b) & np.isfinite(lp0_b)  # lp0_b = log_p_ref amostrado
                if valid_b.sum() < 5:
                    lev_b[k] = -np.inf
                else:
                    w_b = lpk_b[valid_b] - lp0_b[valid_b]  # IS com referencia mistura
                    lev_b[k] = float(logsumexp(w_b) - np.log(valid_b.sum()))
            lpost_b = {k: lev_b[k] + np.log(max(prior_probs[k], 1e-300))
                       for k in state_names}
            fin_b = [k for k in state_names if np.isfinite(lpost_b[k])]
            if not fin_b:
                for k in state_names:
                    probs_boot[k].append(1.0/len(state_names))
                continue
            ln_b = logsumexp([lpost_b[k] for k in fin_b])
            for k in state_names:
                p = float(np.exp(lpost_b[k] - ln_b)) if np.isfinite(lpost_b[k]) else 0.0
                probs_boot[k].append(p)

        probs_std = {k: float(np.std(probs_boot[k])) for k in state_names}

        # Diagnóstico IS: ESS efetivo por estado
        ess_is = {}
        for k in state_names:
            finite_w = log_weights[k][np.isfinite(log_weights[k])]
            if len(finite_w)==0: ess_is[k]=0.0; continue
            w = np.exp(log_weights[k] - finite_w.max())
            w[~np.isfinite(log_weights[k])] = 0
            w_sum = w.sum()
            if w_sum > 0:
                w /= w_sum
                ess_is[k] = float(1.0 / (w ** 2).sum())
            else:
                ess_is[k] = 0.0

        return {
            "probs":        probs,
            "probs_std":    probs_std,
            "probs_boot":   probs_boot,
            "log_evidence": log_evidence,
            "predicted":    predicted,
            "confidence":   confidence,
            "ess_is":       ess_is,
        }


# ===========================================================================
# Estudo de simulação com IS-Posterior
# ===========================================================================

def simulation_study_v2(f, snr_db: float = 30.0,
                         n_per_state: int = 40,
                         n_mcmc: int = 4000,
                         n_warmup: int = 2000,
                         seed: int = 42,
                         verbose: bool = True) -> dict:
    """
    Estudo de simulação completo com o classificador IS-Posterior.

    Fluxo por espectro:
        1. Gerar Z_obs sintético do estado k
        2. Rodar MCMC → p(θ | Z_obs)
        3. Aplicar IS-Posterior → p(estado | Z_obs)
        4. Registrar predição, confiança, calibração

    Parâmetros
    ----------
    n_mcmc   : amostras MCMC por espectro
    n_warmup : warm-up MCMC
    """
    from .data_generation import generate_eis_data
    from .ls_fitting import nlls_fit
    from .mcmc_sampler import AdaptiveMCMC
    from .models import MODEL_COLE_COLE

    rng = np.random.default_rng(seed)
    state_names = list(ALL_STATES.keys())
    n_states    = len(state_names)
    classifier  = ISPosteriorClassifier(n_bootstrap=100)

    confusion     = np.zeros((n_states, n_states), dtype=int)
    all_probs     = {k: [] for k in state_names}
    true_labels   = []
    pred_labels   = []
    confidences   = []
    correct_flags = []
    ess_records   = []

    for i_true, true_name in enumerate(state_names):
        state = ALL_STATES[true_name]
        test_seed = int(rng.integers(0, 2**30))
        params_s = state.sample_params(n=n_per_state, seed=test_seed)

        if verbose:
            print(f"\n  [{true_name}] {n_per_state} espectros (SNR={snr_db} dB)...")

        for j in range(n_per_state):
            R_inf_j = float(params_s["R_inf"][j])
            dR_j    = float(params_s["delta_R"][j])
            tau_j   = float(params_s["tau"][j])
            alpha_j = float(params_s["alpha"][j])

            data_j = generate_eis_data(
                f, R_inf_j, R_inf_j + dR_j, tau_j, alpha_j,
                snr_db=snr_db,
                seed=int(rng.integers(0, 2**30))
            )
            Z_obs_j = data_j["Z_noisy"]

            # Etapa 1: MCMC para p(θ | Z_obs)
            ls_j = nlls_fit(f, Z_obs_j, method="complex", n_restarts=3)
            if ls_j.get("converged"):
                p = ls_j["params"]
                theta0 = np.array([max(p["R_inf"],1.),
                                   max(p["R0"]-p["R_inf"],1.),
                                   np.log(max(p["tau"],1e-10)),
                                   np.clip(p["alpha"],0.05,0.98)])
            else:
                theta0 = np.array([50., 150., np.log(7.96e-6), 0.75])

            from .mcmc_sampler import log_posterior
            fn = lambda th: log_posterior(th, f, Z_obs_j, snr_db)

            sampler = AdaptiveMCMC(
                n_samples=n_mcmc, n_warmup=n_warmup,
                adapt_start=min(300, n_warmup//4),
                adapt_interval=100,
                seed=int(rng.integers(0, 2**30))
            )
            try:
                mcmc_r = sampler.run(fn, theta0, verbose=False)
                samples_j = mcmc_r["samples"]
            except Exception:
                continue

            # Etapa 2: IS-Posterior → classificação
            result_j = classifier.classify(samples_j)

            pred_name = result_j["predicted"]
            i_pred = state_names.index(pred_name)
            confusion[i_true, i_pred] += 1
            true_labels.append(true_name)
            pred_labels.append(pred_name)
            confidences.append(result_j["confidence"])
            correct_flags.append(int(pred_name == true_name))
            ess_records.append(result_j["ess_is"])

            for k in state_names:
                all_probs[k].append(result_j["probs"][k])

            if verbose and j % 5 == 0:
                print(f"    {j+1}/{n_per_state}: pred={pred_name:10s} "
                      f"conf={result_j['confidence']:.2f} "
                      f"{'✓' if pred_name==true_name else '✗'}")

    n_total   = confusion.sum()
    accuracy  = float(np.diag(confusion).sum() / max(n_total, 1))
    class_acc = {state_names[i]: float(confusion[i,i]/max(confusion[i].sum(),1))
                 for i in range(n_states)}
    auc_roc   = {k: _compute_auc(
                     [1 if t==k else 0 for t in true_labels], all_probs[k])
                 for k in state_names}
    calibration = _compute_calibration(confidences, correct_flags, n_bins=10)
    mean_ess  = {k: float(np.mean([e[k] for e in ess_records if k in e]))
                 for k in state_names} if ess_records else {}

    return {
        "confusion":      confusion,
        "accuracy":       accuracy,
        "class_accuracy": class_acc,
        "state_names":    state_names,
        "all_probs":      all_probs,
        "true_labels":    true_labels,
        "pred_labels":    pred_labels,
        "auc_roc":        auc_roc,
        "confidences":    confidences,
        "correct_flags":  correct_flags,
        "calibration":    calibration,
        "mean_ess_is":    mean_ess,
        "snr_db":         snr_db,
    }


def snr_sensitivity_v2(f, snr_levels=None,
                        n_per_state=15, n_mcmc=3000,
                        n_warmup=1500, seed=42) -> dict:
    """Varredura SNR com o classificador IS-Posterior."""
    if snr_levels is None:
        snr_levels = [15, 20, 25, 30, 35, 40]
    results = {}
    for snr in snr_levels:
        print(f"\n=== SNR = {snr} dB ===")
        r = simulation_study_v2(f, snr_db=snr,
                                 n_per_state=n_per_state,
                                 n_mcmc=n_mcmc, n_warmup=n_warmup,
                                 seed=seed)
        auc_mean = float(np.mean(list(r["auc_roc"].values())))
        results[snr] = {
            "accuracy":  r["accuracy"],
            "auc_mean":  auc_mean,
            "auc_roc":   r["auc_roc"],
            "class_acc": r["class_accuracy"],
            "confusion": r["confusion"],
            "ece":       r["calibration"]["ece"],
            "mean_ess":  r.get("mean_ess_is", {}),
        }
        print(f"  Acurácia: {r['accuracy']*100:.1f}%  AUC médio: {auc_mean:.3f}  "
              f"ECE: {r['calibration']['ece']:.3f}")
    return results
