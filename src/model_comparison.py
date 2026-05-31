"""
model_comparison.py
===================
Comparação formal de modelos de circuito equivalente via WAIC e LOO-CV.

Critérios implementados:
    WAIC  — Widely Applicable Information Criterion (Watanabe, 2010)
    LOO   — Leave-One-Out CV por PSIS (Pareto-Smoothed Importance Sampling)
    ΔWAIC — diferença relativa entre modelos na escala de Jeffreys

Escala de Jeffreys (1961) para interpretação de ΔWAIC/2:
    |ΔWAIC/2| < 1   → evidência insignificante
    1 – 3           → evidência positiva
    3 – 5           → evidência forte
    > 5             → evidência muito forte

Referências fundamentais:
    Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and
    widely applicable information criterion in singular learning theory.
    JMLR, 11, 3571–3594.

    Vehtari, A., Gelman, A. & Gabry, J. (2017). Practical Bayesian model
    evaluation using leave-one-out cross-validation and WAIC.
    Stat Comput 27, 1413–1432.  ← referência canônica do PSIS-LOO

    arXiv:2407.20297 — An Assessment of Commonly Used Equivalent Circuit Models
    for Corrosion Analysis: A Bayesian Approach to EIS. Usa framework idêntico
    para comparar ECMs em contexto eletroquímico.

    Grimnes & Martinsen (2015), Cap. 7 — limitações de mínimos quadrados e
    necessidade de critérios de seleção de modelos.
"""

import numpy as np
from scipy.special import logsumexp


# ===========================================================================
# WAIC — Widely Applicable Information Criterion
# ===========================================================================

def compute_waic(ll_matrix: np.ndarray) -> dict:
    """
    Calcula WAIC a partir da matriz de log-verossimilhança ponto-a-ponto.

    Fórmula (Vehtari et al., 2017, eq. 5–7):
        lppd  = Σ_i log( (1/S) Σ_s exp(ll_{s,i}) )
              = Σ_i [ logsumexp(ll[:,i]) - log(S) ]

        p_WAIC = Σ_i Var_s(ll_{s,i})          ← p_WAIC2 (mais estável)

        WAIC   = -2 * (lppd - p_WAIC)
        SE_WAIC = sqrt(n_data * Var_i(elpdi))  ← erro padrão por ponto

    Parâmetros
    ----------
    ll_matrix : array (n_samples, n_data)
                ll_matrix[s,i] = log p(y_i | θ_s)

    Retorna
    -------
    dict com chaves:
        'waic'    : valor do WAIC
        'lppd'    : log pointwise predictive density
        'p_waic'  : penalidade de parâmetros efetivos
        'se_waic' : erro padrão do WAIC
        'elpd_i'  : array de ELPD por ponto (para diagnóstico)
    """
    S, n_data = ll_matrix.shape
    log_S = np.log(S)

    # lppd por ponto: log(média_s p(y_i|θ_s))
    lppd_i = logsumexp(ll_matrix, axis=0) - log_S        # (n_data,)
    lppd   = float(lppd_i.sum())

    # p_WAIC2: variância da log-verossimilhança por ponto (Vehtari 2017, eq.7)
    p_waic_i = ll_matrix.var(axis=0, ddof=1)             # (n_data,)
    p_waic   = float(p_waic_i.sum())

    # ELPD e WAIC
    elpd_i = lppd_i - p_waic_i
    waic   = float(-2.0 * elpd_i.sum())

    # Erro padrão (assumindo pontos i.i.d. — conservador para EIS)
    se_waic = float(np.sqrt(n_data * elpd_i.var(ddof=1)))

    return {
        "waic":    waic,
        "lppd":    lppd,
        "p_waic":  p_waic,
        "se_waic": se_waic,
        "elpd_i":  elpd_i,
        "n_params_eff": p_waic,  # alias interpretativo
    }


# ===========================================================================
# PSIS-LOO — Pareto-Smoothed Importance Sampling LOO
# ===========================================================================

def _pareto_smooth_weights(log_ratios: np.ndarray) -> tuple:
    """
    Suaviza os pesos de importância log-space via ajuste de cauda de Pareto.

    Implementação simplificada de Vehtari et al. (2017), Algorithm 1.
    Ajusta uma distribuição de Pareto generalizada à cauda superior dos
    log-ratios e substitui os pesos extremos por valores suavizados.

    Retorna (log_weights_smoothed, k_hat) onde k_hat é o parâmetro de
    forma de Pareto — indicador de confiabilidade:
        k < 0.5  → muito confiável
        0.5–0.7  → confiável
        0.7–1.0  → problemático (aviso)
        > 1.0    → não confiável
    """
    S = len(log_ratios)
    # Selecionar a cauda superior (20% dos pesos) para ajuste de Pareto
    n_tail = max(int(np.ceil(0.2 * S)), 5)
    sorted_idx = np.argsort(log_ratios)
    tail_idx = sorted_idx[-n_tail:]
    tail_vals = log_ratios[tail_idx]

    # Normalizar a cauda (referência: Vehtari 2017, equação 12)
    tail_min = tail_vals.min()
    tail_norm = tail_vals - tail_min

    # Estimativa de k via método dos momentos de Zhang & Stephens (2009)
    if tail_norm.max() < 1e-10:
        return log_ratios.copy(), 0.0

    theta = tail_norm.mean()
    if theta < 1e-10:
        return log_ratios.copy(), 0.0

    k_hat = float(np.log(tail_norm.mean()) - np.log(tail_norm[-1]) + 1.0) if tail_norm[-1] > 0 else 0.5
    k_hat = np.clip(k_hat, -2.0, 2.0)   # regularização numérica

    # Suavizar pesos da cauda com quantis da distribuição de Pareto estimada
    log_ratios_smooth = log_ratios.copy()
    if abs(k_hat) > 1e-6:
        for j, idx in enumerate(tail_idx):
            p_j = (j + 0.5) / n_tail
            # Quantil da GPD: theta/k * ((1-p)^(-k) - 1)
            try:
                q = theta / k_hat * ((1.0 - p_j) ** (-k_hat) - 1.0) + tail_min
                log_ratios_smooth[idx] = q
            except (ZeroDivisionError, ValueError):
                pass

    return log_ratios_smooth, k_hat


def compute_loo_psis(ll_matrix: np.ndarray) -> dict:
    """
    LOO-CV via PSIS (Pareto-Smoothed Importance Sampling).

    Para cada ponto de dado i:
        log w_{s,i} = -ll_{s,i}   (importância para excluir ponto i)
        LOO_i ≈ log( Σ_s w_norm_{s,i} · exp(ll_{s,i}) )

    Parâmetros
    ----------
    ll_matrix : array (n_samples, n_data)

    Retorna
    -------
    dict com:
        'loo'       : LOO-CV score (−2 * soma das LOO-i)
        'lppd_loo'  : Σ LOO_i
        'k_hats'    : array de k Pareto por ponto (diagnóstico)
        'n_bad_k'   : número de pontos com k > 0.7 (não confiáveis)
        'se_loo'    : erro padrão do LOO
        'loo_i'     : array de ELPD-LOO por ponto
    """
    S, n_data = ll_matrix.shape
    loo_i   = np.zeros(n_data)
    k_hats  = np.zeros(n_data)

    for i in range(n_data):
        # Pesos de importância = inverso da verossimilhança do ponto i
        log_ratios = -ll_matrix[:, i]

        # Normalizar antes da suavização
        log_ratios_c = log_ratios - log_ratios.max()

        # PSIS: suavizar pesos
        log_r_smooth, k = _pareto_smooth_weights(log_ratios_c)
        k_hats[i] = k

        # Normalizar pesos suavizados
        log_r_norm = log_r_smooth - logsumexp(log_r_smooth)

        # LOO para ponto i: E_posterior[-i][p(y_i|θ)]
        # = Σ_s w_{s,i} * p(y_i|θ_s)
        loo_i[i] = float(logsumexp(log_r_norm + ll_matrix[:, i]))

    lppd_loo = float(loo_i.sum())
    loo_score = -2.0 * lppd_loo
    se_loo = float(np.sqrt(n_data * loo_i.var(ddof=1)))
    n_bad_k = int((k_hats > 0.7).sum())

    return {
        "loo":      loo_score,
        "lppd_loo": lppd_loo,
        "k_hats":   k_hats,
        "n_bad_k":  n_bad_k,
        "se_loo":   se_loo,
        "loo_i":    loo_i,
    }


# ===========================================================================
# Comparação entre modelos
# ===========================================================================

def compare_models(results: dict) -> dict:
    """
    Compara múltiplos modelos pelo WAIC e LOO.

    Parâmetros
    ----------
    results : dict {nome_modelo: {'waic': dict, 'loo': dict, 'n_params': int}}

    Retorna
    -------
    dict com tabela de comparação e ranking.
    """
    model_names = list(results.keys())

    # Ordenar por WAIC
    waics = {k: results[k]["waic"]["waic"] for k in model_names}
    loos  = {k: results[k]["loo"]["loo"]   for k in model_names}

    best_waic = min(waics.values())
    best_loo  = min(loos.values())

    comparison = {}
    for name in model_names:
        w = results[name]["waic"]
        l = results[name]["loo"]
        delta_waic = waics[name] - best_waic
        delta_loo  = loos[name]  - best_loo
        n_bad = results[name]["loo"]["n_bad_k"]

        # Interpretação na escala de Jeffreys (ΔWAIC/2)
        jeffreys = jeffreys_scale(delta_waic / 2.0)

        comparison[name] = {
            "waic":        waics[name],
            "delta_waic":  delta_waic,
            "se_waic":     w["se_waic"],
            "p_waic":      w["p_waic"],
            "loo":         loos[name],
            "delta_loo":   delta_loo,
            "se_loo":      l["se_loo"],
            "n_bad_k":     n_bad,
            "jeffreys":    jeffreys,
            "n_params":    results[name].get("n_params", "?"),
            "is_best":     (delta_waic == 0.0),
        }

    # Ranking
    ranking = sorted(model_names, key=lambda k: waics[k])

    return {"table": comparison, "ranking": ranking, "best": ranking[0]}


def jeffreys_scale(delta_half_waic: float) -> str:
    """
    Interpreta |ΔWAIC/2| na escala de Jeffreys (1961).
    ΔWAIC/2 ≈ log Bayes Factor para modelos com mesmo prior.
    """
    v = abs(delta_half_waic)
    if v < 1.0:    return "insignificante"
    elif v < 3.0:  return "positiva"
    elif v < 5.0:  return "forte"
    else:          return "muito forte"


def print_comparison_table(comparison: dict, ranking: list):
    """Imprime tabela formatada de comparação de modelos."""
    print("\n" + "=" * 90)
    print("  COMPARAÇÃO DE MODELOS — WAIC e LOO-CV (Vehtari et al., 2017)")
    print("=" * 90)
    print(f"{'Modelo':>20} {'WAIC':>10} {'ΔWAIC':>9} {'SE(WAIC)':>10} "
          f"{'p_WAIC':>8} {'LOO':>10} {'ΔLOO':>8} {'k>0.7':>7} {'Evidência':>14}")
    print("-" * 90)
    for name in ranking:
        r = comparison[name]
        best_mark = " ★" if r["is_best"] else "  "
        print(f"{name+best_mark:>22} {r['waic']:>10.2f} {r['delta_waic']:>9.2f} "
              f"{r['se_waic']:>10.2f} {r['p_waic']:>8.2f} {r['loo']:>10.2f} "
              f"{r['delta_loo']:>8.2f} {r['n_bad_k']:>7d} {r['jeffreys']:>14}")
    print("=" * 90)
    print("  ★ = melhor modelo  |  Evidência: coluna ΔWAIC/2 na escala de Jeffreys (1961)")
    print("  k>0.7: pontos com estimativa LOO não confiável (Vehtari et al., 2017)")
    print()
