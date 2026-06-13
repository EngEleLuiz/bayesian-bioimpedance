"""
tissue_states.py
================
Parâmetros dos estados (Cole-Cole, tecido muscular):
    NORMAL   — Gabriel et al. (1996); IT'IS Foundation
    EDEMA    — Morimoto et al. (1993); ACS Measurement Sci. Au (2022)
               R_inf↓15-25%, tau↑40-60%, alpha↓ — retenção hídrica extracelular
    ISQUEMIA — Haemmerich et al. (2002); Zhao et al. (2007)
               R_inf↑20-35%, ΔR↑30-50%, tau↓30%, alpha↑ — morte celular/necrose

Referências:
    Gabriel, S. et al. (1996). Phys Med Biol, 41, 2271–2293.
    Morimoto, T. et al. (1993). Med Biol Eng Comput, 31, 9–16.
    Haemmerich, D. et al. (2002). Physiol Meas, 24, 251–260.
    Zhao, T.X. et al. (2007). Eur Heart J, 28, 167–174.
    ACS Measurement Science Au (2022). DOI: 10.1021/acsmeasuresciau.2c00024
"""

import numpy as np
from scipy.special import logsumexp
from dataclasses import dataclass
from .cole_model import cole_cole_impedance, characteristic_frequency
from .data_generation import snr_to_sigma


# ===========================================================================
# Definição dos estados teciduais
# ===========================================================================

@dataclass
class TissueState:
    """
    Estado tecidual com distribuição a priori sobre parâmetros Cole-Cole.

    Modelagem: parâmetros positivos via LogNormal (R_inf, ΔR, τ)
               e Beta para α ∈ (0,1].

    σ_log ≈ 0.30–0.40 reflete variabilidade inter-sujeito de ~30–40%,
    consistente com dados de Gabriel et al. (1996) para tecido muscular.
    """
    name:         str
    color:        str
    mu_log_Rinf:  float
    sig_log_Rinf: float
    mu_log_dR:    float
    sig_log_dR:   float
    mu_log_tau:   float
    sig_log_tau:  float
    alpha_beta_a: float
    alpha_beta_b: float
    reference:    str = ""

    def sample_params(self, n: int = 1000, seed: int = None) -> dict:
        rng = np.random.default_rng(seed)
        R_inf   = np.exp(rng.normal(self.mu_log_Rinf, self.sig_log_Rinf, n))
        delta_R = np.exp(rng.normal(self.mu_log_dR,   self.sig_log_dR,   n))
        tau     = np.exp(rng.normal(self.mu_log_tau,   self.sig_log_tau,  n))
        alpha   = rng.beta(self.alpha_beta_a, self.alpha_beta_b, n)
        return {
            "R_inf":   R_inf,
            "delta_R": delta_R,
            "tau":     tau,
            "alpha":   alpha,
            "R0":      R_inf + delta_R,
            "f_c":     np.array([characteristic_frequency(t, a)
                                 for t, a in zip(tau, alpha)]),
        }

    def nominal_params(self) -> dict:
        alpha_mode = (self.alpha_beta_a - 1) / (self.alpha_beta_a + self.alpha_beta_b - 2)
        R_inf = np.exp(self.mu_log_Rinf)
        dR    = np.exp(self.mu_log_dR)
        return {"R_inf": R_inf, "R0": R_inf + dR,
                "tau": np.exp(self.mu_log_tau), "alpha": alpha_mode}


# ---------------------------------------------------------------------------
# Parâmetros calibrados para sobreposição clínica realista
# Meta: Cohen d ≈ 0.7–1.3 (desafio real), conforme diagnóstico v2
# ---------------------------------------------------------------------------

STATE_NORMAL = TissueState(
    name="Normal", color="#1f77b4",
    # R_inf=50Ω, ΔR=150Ω, τ=8μs, α≈0.75 — Gabriel et al. (1996)
    mu_log_Rinf=np.log(50.0),   sig_log_Rinf=0.30,
    mu_log_dR  =np.log(150.0),  sig_log_dR  =0.32,
    mu_log_tau =np.log(7.96e-6),sig_log_tau =0.38,
    alpha_beta_a=10.0, alpha_beta_b=3.2,   # modo≈0.75, std≈0.10
    reference="Gabriel et al. (1996), Phys Med Biol 41:2271"
)

STATE_EDEMA = TissueState(
    name="Edema", color="#2ca02c",
    # R_inf↓25%=38Ω, ΔR↓30%=105Ω, τ↑60%=13μs, α↓=0.66
    # Edema: ↑ fluido extracelular → ↓ R_inf, membrana mais permeável → ↑ τ
    mu_log_Rinf=np.log(38.0),   sig_log_Rinf=0.35,
    mu_log_dR  =np.log(105.0),  sig_log_dR  =0.35,
    mu_log_tau =np.log(1.3e-5), sig_log_tau =0.42,
    alpha_beta_a=7.0, alpha_beta_b=3.2,    # modo≈0.66, std≈0.12
    reference="Morimoto et al. (1993); ACS Meas Sci Au (2022)"
)

STATE_ISCHEMIA = TissueState(
    name="Isquemia", color="#d62728",
    # R_inf↑30%=65Ω, ΔR↑45%=217Ω, τ↓35%=5μs, α↑=0.83
    # Isquemia: colapso membrana → ↑ R_inf, ↑ ΔR, τ cai (menos capacitância)
    mu_log_Rinf=np.log(65.0),   sig_log_Rinf=0.30,
    mu_log_dR  =np.log(217.0),  sig_log_dR  =0.33,
    mu_log_tau =np.log(5.0e-6), sig_log_tau =0.35,
    alpha_beta_a=13.0, alpha_beta_b=2.8,   # modo≈0.83, std≈0.08
    reference="Haemmerich et al. (2002); Zhao et al. (2007)"
)

ALL_STATES = {
    "Normal":   STATE_NORMAL,
    "Edema":    STATE_EDEMA,
    "Isquemia": STATE_ISCHEMIA,
}


# ===========================================================================
# Diagnóstico de separabilidade entre estados
# ===========================================================================

def diagnose_separability(n: int = 10000, seed: int = 0) -> dict:
    """
    Calcula Cohen d e overlap entre todos os pares de estados.
    Retorna dict com métricas de separabilidade.
    """
    results = {}
    state_names = list(ALL_STATES.keys())
    params_names = ["R_inf", "delta_R", "tau", "alpha"]

    samples = {k: ALL_STATES[k].sample_params(n=n, seed=seed+i)
               for i, k in enumerate(state_names)}

    for i, k1 in enumerate(state_names):
        for k2 in state_names[i+1:]:
            pair = f"{k1} vs {k2}"
            results[pair] = {}
            for p in params_names:
                a = samples[k1][p]
                b = samples[k2][p]
                d = (abs(a.mean() - b.mean()) /
                     np.sqrt((a.std()**2 + b.std()**2) / 2))
                lo = max(np.percentile(a, 5), np.percentile(b, 5))
                hi = min(np.percentile(a, 95), np.percentile(b, 95))
                overlap = max(0.0, hi - lo)
                results[pair][p] = {"cohen_d": float(d), "overlap_90": float(overlap)}
    return results


def print_separability_report(diag: dict):
    print("\n=== SEPARABILIDADE ENTRE ESTADOS (Cohen d) ===")
    print("  d < 0.5: pequeno | 0.5–0.8: médio | 0.8–1.5: grande | >2: trivial\n")
    for pair, params in diag.items():
        print(f"  {pair}:")
        for p, m in params.items():
            flag = ("⚠ trivial" if m["cohen_d"] > 2
                    else ("✓ desafiador" if m["cohen_d"] < 1.5 else "ok"))
            print(f"    {p:8s}: d={m['cohen_d']:.2f}  {flag}")


# ===========================================================================
# Verossimilhança marginal Monte Carlo (numericamente estável)
# ===========================================================================

def marginal_log_likelihood_mc(state: TissueState, f, Z_obs,
                                snr_db, n_mc=2000, seed=None) -> float:
    """
    Estima log p(Z_obs | estado) via Monte Carlo com estabilidade numérica.

    log p(Z|k) ≈ logsumexp(ll_i) - log(N_valid)

    Correção v2: usa apenas amostras com ll > percentil_5 para evitar
    que amostras outlier de regiões de baixa probabilidade do prior
    dominem a estimativa via logsumexp.
    """
    params = state.sample_params(n=n_mc, seed=seed)
    sigma_noise = snr_to_sigma(snr_db)
    log_likelihoods = []

    for i in range(n_mc):
        try:
            R_inf_i  = float(params["R_inf"][i])
            dR_i     = float(params["delta_R"][i])
            tau_i    = float(params["tau"][i])
            alpha_i  = float(params["alpha"][i])
            R0_i     = R_inf_i + dR_i

            if R_inf_i <= 0 or dR_i <= 0 or tau_i <= 0 or alpha_i <= 0 or alpha_i > 1:
                continue

            Z_pred = cole_cole_impedance(f, R_inf_i, R0_i, tau_i, alpha_i)
            if not np.all(np.isfinite(Z_pred)):
                continue

            sigma_f = np.maximum(sigma_noise * np.abs(Z_pred), 1e-12)
            res_R = Z_obs.real - Z_pred.real
            res_X = Z_obs.imag - Z_pred.imag
            ll = (- np.sum(np.log(sigma_f))
                  - 0.5 * np.sum((res_R / sigma_f) ** 2)
                  - np.sum(np.log(sigma_f))
                  - 0.5 * np.sum((res_X / sigma_f) ** 2))

            if np.isfinite(ll):
                log_likelihoods.append(ll)
        except Exception:
            continue

    if len(log_likelihoods) < 10:
        return -np.inf

    ll_arr = np.array(log_likelihoods)

    # Remover outliers extremos do prior (< percentil 5) para estabilidade
    threshold = np.percentile(ll_arr, 5)
    ll_arr = ll_arr[ll_arr >= threshold]

    return float(logsumexp(ll_arr) - np.log(len(ll_arr)))


# ===========================================================================
# Classificador Bayesiano
# ===========================================================================

class BayesianTissueClassifier:
    """
    Classificador Bayesiano via verossimilhança marginal Monte Carlo.
    Versão corrigida: parâmetros realistas + estimativa estável + calibração.
    """

    def __init__(self, states: dict = None, n_mc: int = 2000, seed: int = 42):
        self.states = states or ALL_STATES
        self.n_mc   = n_mc
        self.seed   = seed

    def classify(self, f, Z_obs, snr_db=30.0,
                 prior_probs: dict = None,
                 n_bootstrap: int = 50) -> dict:
        """
        Classifica Z_obs estimando p(estado | Z_obs) para cada estado.

        Retorna probabilidades posteriores com intervalo de credibilidade
        via bootstrap paramétrico do erro de estimação MC.
        """
        state_names = list(self.states.keys())
        if prior_probs is None:
            prior_probs = {k: 1.0 / len(state_names) for k in state_names}

        rng = np.random.default_rng(self.seed)

        # Marginal log-likelihood por estado
        log_marg = {}
        for name, state in self.states.items():
            seed_k = int(rng.integers(0, 2**30))
            log_marg[name] = marginal_log_likelihood_mc(
                state, f, Z_obs, snr_db, n_mc=self.n_mc, seed=seed_k)

        # Posterior normalizado
        log_post = {k: log_marg[k] + np.log(max(prior_probs[k], 1e-300))
                    for k in state_names}

        # Verificar se algum é finito
        finite_keys = [k for k in state_names if np.isfinite(log_post[k])]
        if not finite_keys:
            # Fallback: prior uniforme (não conseguiu discriminar)
            probs = {k: 1.0/len(state_names) for k in state_names}
        else:
            log_norm = logsumexp([log_post[k] for k in state_names
                                  if np.isfinite(log_post[k])])
            probs = {}
            for k in state_names:
                if np.isfinite(log_post[k]):
                    probs[k] = float(np.exp(log_post[k] - log_norm))
                else:
                    probs[k] = 0.0

        predicted  = max(probs, key=probs.get)
        confidence = probs[predicted]

        # Bootstrap do erro MC (σ ≈ 1/√n_mc por estimativa logsumexp)
        probs_boot = {k: [] for k in state_names}
        mc_err = 1.5 / np.sqrt(self.n_mc)   # erro conservador
        for _ in range(n_bootstrap):
            lm_b = {k: log_marg[k] + rng.normal(0, mc_err)
                    if np.isfinite(log_marg[k]) else -np.inf
                    for k in state_names}
            lp_b = {k: lm_b[k] + np.log(max(prior_probs[k], 1e-300))
                    for k in state_names}
            finite_b = [k for k in state_names if np.isfinite(lp_b[k])]
            if not finite_b:
                for k in state_names:
                    probs_boot[k].append(1.0/len(state_names))
                continue
            ln_b = logsumexp([lp_b[k] for k in finite_b])
            for k in state_names:
                p = float(np.exp(lp_b[k] - ln_b)) if np.isfinite(lp_b[k]) else 0.0
                probs_boot[k].append(p)

        probs_std = {k: float(np.std(probs_boot[k])) for k in state_names}

        return {
            "probs":      probs,
            "probs_std":  probs_std,
            "probs_boot": probs_boot,
            "log_marg":   log_marg,
            "predicted":  predicted,
            "confidence": confidence,
        }


# ===========================================================================
# Estudo de simulação com hold-out independente
# ===========================================================================

def simulation_study(f, snr_db: float = 30.0,
                     n_per_state: int = 40,
                     n_mc: int = 1000,
                     seed: int = 42) -> dict:
    """
    Avalia desempenho com hold-out separado do prior (sem data leakage).

    CORREÇÃO v2: os espectros de teste são gerados com parâmetros
    amostrados do prior, mas o classificador usa um prior independente
    (semente diferente). Isso garante que teste e 'treino' (prior) são
    conjuntos independentes.

    Retorna: confusion_matrix, accuracy, class_accuracy, AUC-ROC,
             calibration data (reliability diagram), confidence histogram
    """
    from .data_generation import generate_eis_data

    rng = np.random.default_rng(seed)
    state_names  = list(ALL_STATES.keys())
    n_states     = len(state_names)
    # Semente separada para classificador (garante independência)
    clf_seed = int(rng.integers(0, 2**30))
    classifier = BayesianTissueClassifier(n_mc=n_mc, seed=clf_seed)

    confusion   = np.zeros((n_states, n_states), dtype=int)
    all_probs   = {k: [] for k in state_names}
    true_labels = []
    pred_labels = []
    confidences = []
    correct_flags = []

    for i_true, true_name in enumerate(state_names):
        state = ALL_STATES[true_name]
        # Semente de teste DIFERENTE da semente do classificador
        test_seed = int(rng.integers(0, 2**30))
        params_samples = state.sample_params(n=n_per_state, seed=test_seed)

        print(f"\n  [{true_name}] {n_per_state} espectros de teste (SNR={snr_db} dB)...")

        for j in range(n_per_state):
            R_inf_j = float(params_samples["R_inf"][j])
            dR_j    = float(params_samples["delta_R"][j])
            tau_j   = float(params_samples["tau"][j])
            alpha_j = float(params_samples["alpha"][j])

            data_j = generate_eis_data(
                f, R_inf_j, R_inf_j + dR_j, tau_j, alpha_j,
                snr_db=snr_db,
                seed=int(rng.integers(0, 2**30))
            )
            Z_obs_j = data_j["Z_noisy"]

            result_j = classifier.classify(f, Z_obs_j, snr_db=snr_db,
                                           n_bootstrap=30)

            pred_name = result_j["predicted"]
            i_pred = state_names.index(pred_name)
            confusion[i_true, i_pred] += 1
            true_labels.append(true_name)
            pred_labels.append(pred_name)
            confidences.append(result_j["confidence"])
            correct_flags.append(int(pred_name == true_name))

            for k in state_names:
                all_probs[k].append(result_j["probs"][k])

    accuracy = float(np.diag(confusion).sum() / confusion.sum())
    class_accuracy = {state_names[i]: float(confusion[i,i] / max(confusion[i].sum(),1))
                      for i in range(n_states)}

    auc_roc = {}
    for k in state_names:
        true_bin = [1 if t == k else 0 for t in true_labels]
        auc_roc[k] = _compute_auc(true_bin, all_probs[k])

    # Calibração: reliability diagram
    calibration = _compute_calibration(confidences, correct_flags, n_bins=10)

    return {
        "confusion":      confusion,
        "accuracy":       accuracy,
        "class_accuracy": class_accuracy,
        "state_names":    state_names,
        "all_probs":      all_probs,
        "true_labels":    true_labels,
        "pred_labels":    pred_labels,
        "auc_roc":        auc_roc,
        "confidences":    confidences,
        "correct_flags":  correct_flags,
        "calibration":    calibration,
        "snr_db":         snr_db,
    }


def _compute_calibration(confidences, correct_flags, n_bins=10):
    """
    Reliability diagram: divide confidências em bins e compara com acurácia real.
    ECE = Σ (|bin|/N) * |acc_bin - conf_bin|  (Expected Calibration Error)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not idx:
            bin_acc.append(np.nan)
            bin_conf.append((lo+hi)/2)
            bin_count.append(0)
        else:
            bin_acc.append(np.mean([correct_flags[i] for i in idx]))
            bin_conf.append(np.mean([confidences[i] for i in idx]))
            bin_count.append(len(idx))

    # ECE
    total = len(confidences)
    ece = sum(c * abs(a - cf)
              for a, cf, c in zip(bin_acc, bin_conf, bin_count)
              if not np.isnan(a)) / max(total, 1)

    return {"bin_acc": bin_acc, "bin_conf": bin_conf,
            "bin_count": bin_count, "ece": ece}


def _compute_auc(y_true, scores) -> float:
    n_pos = sum(y_true); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pairs = sorted(zip(scores, y_true), key=lambda x: -x[0])
    tp = fp = 0
    tpr_list, fpr_list = [0.0], [0.0]
    for _, lb in pairs:
        if lb == 1: tp += 1
        else:       fp += 1
        tpr_list.append(tp/n_pos)
        fpr_list.append(fp/n_neg)
    return abs(float(np.trapezoid(tpr_list, fpr_list)))


def snr_sensitivity_classification(f, snr_levels=None,
                                   n_per_state=20, n_mc=600,
                                   seed=42) -> dict:
    if snr_levels is None:
        snr_levels = [15, 20, 25, 30, 35, 40]
    results_by_snr = {}
    for snr in snr_levels:
        print(f"\n  === SNR = {snr} dB ===")
        result = simulation_study(f, snr_db=snr,
                                  n_per_state=n_per_state,
                                  n_mc=n_mc, seed=seed)
        auc_mean = float(np.mean(list(r["auc_roc"].values())))
        ece      = r["calibration"]["ece"]          # inclui ECE
        results[snr] = {
            "accuracy":  r["accuracy"],
            "auc_mean":  auc_mean,
            "auc_roc":   r["auc_roc"],
            "class_acc": r["class_accuracy"],
            "confusion": r["confusion"],
            "ece":       ece,                        
            "n_test":    n_per_state * len(list(r["class_accuracy"].keys())),
        }
        print(f"  Acurácia: {r['accuracy']*100:.1f}%  "
              f"AUC médio: {auc_mean:.3f}  ECE: {ece:.3f}")
    return results_by_snr
