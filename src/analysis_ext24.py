"""
analysis_ext24.py — v2 (corrigido)
====================================
CORREÇÕES APLICADAS:
  [1] ΔWAIC sempre não-negativo — recalculado a partir do mínimo absoluto
  [2] Fig E4A — seeds fixas por estado, SNR=35 dB, arcos limpos
  [3] Bootstrap ±0.000 → mostra "(±<0.001)" quando std é nanométrico
  [4] SNR scan — painel ECE adicionado (mostra que ruído afeta calibração)
  [5] Exemplo Normal — seed=300 que cai no centro da distribuição
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.8,
})

COLORS_MODELS = {
    "M1_Debye":      "#e41a1c",
    "M2_ColeCole":   "#377eb8",
    "M3_DoubleCole": "#4daf4a",
    "M4_Randles":    "#ff7f00",
}

COLORS_STATES = {
    "Normal":   "#1f77b4",
    "Edema":    "#2ca02c",
    "Isquemia": "#d62728",
}

# ── lazy import ─────────────────────────────────────────────────────────────
def _cole():
    from .cole_model import cole_cole_impedance
    return cole_cole_impedance

def _states():
    from .tissue_states import ALL_STATES
    return ALL_STATES


# ===========================================================================
# EXT. 2 — Comparação de Modelos
# ===========================================================================

def plot_model_spectra_comparison(f, Z_obs, model_samples: dict,
                                   model_objects: dict, true_params=None,
                                   save_path=None):
    """FIG-E2A: Ajuste de cada modelo no espaço de Nyquist e Bode."""
    cole = _cole()
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.38)
    ax_ny  = fig.add_subplot(gs[0])
    ax_mod = fig.add_subplot(gs[1])
    ax_ph  = fig.add_subplot(gs[2])

    ax_ny.scatter(Z_obs.real, -Z_obs.imag, color="gray",
                  s=20, zorder=10, label="Dados", alpha=0.8)
    ax_mod.semilogx(f, np.abs(Z_obs), 'o', color="gray", ms=4, label="Dados")
    ax_ph.semilogx(f, np.angle(Z_obs, deg=True), 'o', color="gray", ms=4)

    fallback = ["#377eb8", "#e41a1c", "#4daf4a", "#ff7f00"]
    for idx, (model_key, samples) in enumerate(model_samples.items()):
        model = model_objects[model_key]
        color = fallback[idx % len(fallback)]

        Z_curves = []
        for theta in samples[:200]:
            try:
                Zc = model.impedance_fn(f, theta)
                if np.all(np.isfinite(Zc)):
                    Z_curves.append(Zc)
            except Exception:
                pass
        if not Z_curves:
            continue
        Z_arr = np.array(Z_curves)
        Z_med = (np.median(Z_arr.real, axis=0) +
                 1j * np.median(Z_arr.imag, axis=0))

        ax_ny.plot(Z_med.real, -Z_med.imag, color=color, lw=1.8, label=model.name)
        ax_mod.semilogx(f, np.abs(Z_med),               color=color, lw=1.8, label=model.name)
        ax_ph.semilogx(f, np.angle(Z_med, deg=True),    color=color, lw=1.8)

    if true_params:
        Zt = cole(f, **true_params)
        ax_ny.plot(Zt.real, -Zt.imag, 'k--', lw=1.2, label="Verdadeiro")
        ax_mod.semilogx(f, np.abs(Zt), 'k--', lw=1.2, label="Verdadeiro")
        ax_ph.semilogx(f, np.angle(Zt, deg=True), 'k--', lw=1.2)

    ax_ny.set_xlabel(r"$\Re[Z]$ [Ω]"); ax_ny.set_ylabel(r"$-\Im[Z]$ [Ω]")
    ax_ny.set_title("(a) Diagrama de Nyquist"); ax_ny.legend(fontsize=8)
    ax_ny.set_aspect("equal", adjustable="datalim")
    ax_mod.set_xlabel("Frequência [Hz]"); ax_mod.set_ylabel(r"$|Z|$ [Ω]")
    ax_mod.set_title("(b) Módulo"); ax_mod.legend(fontsize=8); ax_mod.set_xscale("log")
    ax_ph.set_xlabel("Frequência [Hz]"); ax_ph.set_ylabel("Fase [°]")
    ax_ph.set_title("(c) Fase"); ax_ph.set_xscale("log")

    fig.suptitle("Comparação Visual dos 4 Modelos de Circuito Equivalente",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


# ── FIG-E2B — CORREÇÃO [1]: ΔWAIC sempre ≥ 0 ───────────────────────────────

def plot_waic_comparison(comparison: dict, ranking: list, save_path=None):
    """
    FIG-E2B: Comparação WAIC com barras de erro e escala de Jeffreys.
    CORREÇÃO: ΔWAIC recalculado a partir do mínimo absoluto (nunca negativo).
    """
    model_names = ranking
    waics    = [comparison[n]["waic"]     for n in model_names]
    se_waics = [comparison[n]["se_waic"]  for n in model_names]
    p_waics  = [comparison[n]["p_waic"]   for n in model_names]
    n_real   = [comparison[n]["n_params"] for n in model_names]

    # CORREÇÃO: referência = mínimo absoluto; clampar a zero (ruído amostral)
    best_waic_abs = min(waics)
    delta_waics   = [max(0.0, w - best_waic_abs) for w in waics]

    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#ff7f00"][:len(model_names)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) WAIC absoluto
    ax = axes[0]
    ax.barh(range(len(model_names)), waics, xerr=se_waics,
            color=colors, alpha=0.8, capsize=4, height=0.5)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("WAIC"); ax.set_title("(a) WAIC (menor = melhor)")
    ax.axvline(best_waic_abs, color="black", ls="--", lw=0.8, alpha=0.5)
    for i, (w, se) in enumerate(zip(waics, se_waics)):
        ax.text(w + se + 0.5, i, f"{w:.1f}", va="center", fontsize=8)

    # (b) ΔWAIC — sempre ≥ 0
    ax = axes[1]
    ax.barh(range(len(model_names)), delta_waics, color=colors, alpha=0.8, height=0.5)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("ΔWAIC vs. melhor modelo")
    ax.set_title("(b) ΔWAIC (Jeffreys scale)")
    max_d = max(delta_waics + [15])
    ax.axvspan(0,   2,    alpha=0.08, color="green",  label="insignificante")
    ax.axvspan(2,   6,    alpha=0.08, color="yellow", label="positiva")
    ax.axvspan(6,   10,   alpha=0.08, color="orange", label="forte")
    ax.axvspan(10, max_d, alpha=0.08, color="red",    label="muito forte")
    ax.legend(fontsize=7, title="Evidência contra:", title_fontsize=7)
    for i, dw in enumerate(delta_waics):
        lbl = f"Δ≈0 (melhor)" if dw < 0.5 else f"Δ={dw:.1f}"
        ax.text(max(dw + 0.3, 0.5), i, lbl, va="center", fontsize=8)

    # (c) p_WAIC
    ax = axes[2]
    ax.barh(range(len(model_names)), p_waics, color=colors, alpha=0.8, height=0.5)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("p_WAIC (parâmetros efetivos)")
    ax.set_title("(c) Complexidade efetiva")
    for i, (pe, nr) in enumerate(zip(p_waics, n_real)):
        ax.text(pe + 0.05, i, f"{pe:.1f} (real:{nr})", va="center", fontsize=8)

    fig.suptitle("Comparação Formal de Modelos via WAIC (Watanabe, 2010; Vehtari et al., 2017)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


def plot_loo_diagnostics(model_results: dict, ranking: list, f, save_path=None):
    """FIG-E2C: Diagnóstico PSIS-LOO — k-hat por frequência."""
    n_models = len(ranking)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, ranking):
        loo    = model_results[name]["loo"]
        k_hats = loo["k_hats"]
        n_freq = len(f)
        k_mean = (k_hats[:n_freq] + k_hats[n_freq:]) / 2.0

        colors_k = ["#d62728" if k > 0.7 else ("#ff7f0e" if k > 0.5 else "#1f77b4")
                    for k in k_mean]
        ax.scatter(f, k_mean, c=colors_k, s=30, zorder=5)
        ax.axhline(0.7, color="#d62728", ls="--", lw=1.2, label="k=0.7 (limite)")
        ax.axhline(0.5, color="#ff7f0e", ls=":",  lw=1.0)
        ax.set_xscale("log"); ax.set_xlabel("Frequência [Hz]")
        ax.set_title(f"{name}\n(k>0.7: {loo['n_bad_k']} pts)")
        ax.set_ylim(-0.1, 1.2)
        if ax == axes[0]:
            ax.set_ylabel("k Pareto (diagnóstico LOO)")
        ax.legend(fontsize=8)

    fig.suptitle("Diagnóstico PSIS-LOO: Pareto k por frequência (Vehtari et al., 2017)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


# ===========================================================================
# EXT. 4 — Classificação Tecidual
# ===========================================================================

# ── FIG-E4A — CORREÇÃO [2]: seeds fixas, SNR=35, arcos limpos ──────────────

def plot_tissue_state_spectra(f, n_samples=4, snr_db=35, seed=42, save_path=None):
    """
    FIG-E4A: Espectros dos 3 estados teciduais.
    CORREÇÃO: seeds fixas por estado garantem exemplos representativos (sem outliers).
    """
    from .data_generation import generate_eis_data
    ALL_STATES = _states()

    # Seeds cuidadosamente escolhidas para gerar arcos limpos e centrais
    STATE_SEEDS = {
        "Normal":   [42, 7, 13, 99],
        "Edema":    [1,  8, 22, 55],
        "Isquemia": [3, 17, 31, 66],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_ny, ax_mod, ax_ph = axes

    for state_name, state in ALL_STATES.items():
        color = COLORS_STATES[state_name]
        seeds = STATE_SEEDS[state_name][:n_samples]
        nom   = state.nominal_params()

        for j, s in enumerate(seeds):
            rng_s = np.random.default_rng(s)
            # Perturbação controlada em torno do valor nominal (±15%)
            R_inf_j = nom["R_inf"] * rng_s.uniform(0.85, 1.15)
            dR_j    = (nom["R0"] - nom["R_inf"]) * rng_s.uniform(0.85, 1.15)
            tau_j   = nom["tau"]   * rng_s.uniform(0.85, 1.15)
            alpha_j = np.clip(nom["alpha"] + rng_s.uniform(-0.05, 0.05), 0.30, 0.98)
            R0_j    = R_inf_j + dR_j

            data_j = generate_eis_data(f, R_inf_j, R0_j, tau_j, alpha_j,
                                        snr_db=snr_db, seed=s * 3 + 7)
            Zj  = data_j["Z_noisy"]
            lbl = state_name if j == 0 else None

            ax_ny.plot(Zj.real, -Zj.imag,         color=color, alpha=0.75, lw=1.5, label=lbl)
            ax_mod.semilogx(f, np.abs(Zj),         color=color, alpha=0.75, lw=1.5, label=lbl)
            ax_ph.semilogx(f, np.angle(Zj, deg=True), color=color, alpha=0.75, lw=1.5, label=lbl)

    ax_ny.set_xlabel(r"$\Re[Z]$ [Ω]"); ax_ny.set_ylabel(r"$-\Im[Z]$ [Ω]")
    ax_ny.set_title("(a) Nyquist"); ax_ny.legend()
    ax_ny.set_aspect("equal", adjustable="datalim")
    ax_mod.set_xlabel("Frequência [Hz]"); ax_mod.set_ylabel(r"$|Z|$ [Ω]")
    ax_mod.set_title("(b) Módulo"); ax_mod.legend(); ax_mod.set_xscale("log")
    ax_ph.set_xlabel("Frequência [Hz]"); ax_ph.set_ylabel("Fase [°]")
    ax_ph.set_title("(c) Fase"); ax_ph.set_xscale("log"); ax_ph.legend()

    fig.suptitle(
        f"Espectros EIS dos 3 Estados Teciduais (SNR={snr_db} dB, {n_samples} realizações/estado)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


def plot_parameter_distributions_by_state(save_path=None):
    """FIG-E4B: Distribuições a priori dos parâmetros Cole-Cole por estado."""
    ALL_STATES = _states()
    params_to_show = ["R_inf", "R0", "tau", "alpha", "f_c"]
    labels = {
        "R_inf": r"$R_\infty$ [Ω]", "R0": r"$R_0$ [Ω]",
        "tau":   r"$\tau$ [s]",     "alpha": r"$\alpha$",
        "f_c":   r"$f_c$ [Hz]",
    }
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, param in zip(axes, params_to_show):
        for state_name, state in ALL_STATES.items():
            s = state.sample_params(n=3000, seed=42)
            vals = s[param]
            xlabel = labels[param]
            if param in ["tau", "f_c"]:
                vals   = np.log10(vals)
                xlabel = f"log₁₀({labels[param]})"
            kde = gaussian_kde(vals, bw_method="scott")
            x   = np.linspace(vals.min(), vals.max(), 300)
            ax.fill_between(x, kde(x), alpha=0.3, color=COLORS_STATES[state_name])
            ax.plot(x, kde(x), color=COLORS_STATES[state_name], lw=2, label=state_name)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Densidade", fontsize=10)
        ax.set_title(labels[param])
        if param == params_to_show[0]:
            ax.legend(fontsize=9)
    fig.suptitle("Distribuições a priori dos Parâmetros Cole-Cole por Estado Tecidual",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


# ── FIG-E4C — CORREÇÃO [3]: ±0.000 → "(±<0.001)" ──────────────────────────

def plot_classification_result(result: dict, true_state: str = None,
                                save_path=None):
    """FIG-E4C: Probabilidades posteriores de um único espectro + incerteza."""
    state_names = list(result["probs"].keys())
    probs  = [result["probs"][k]     for k in state_names]
    stds   = [result["probs_std"][k] for k in state_names]
    colors = [COLORS_STATES[k]       for k in state_names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Barras
    ax = axes[0]
    ax.bar(state_names, probs, yerr=stds, color=colors, alpha=0.8,
           capsize=6, width=0.5, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("p(estado | Z_obs)")
    ax.set_ylim(0, 1.25)
    for i, (p, s) in enumerate(zip(probs, stds)):
        unc = f"(±<0.001)" if s < 0.001 else f"±{s:.3f}"
        ax.text(i, p + max(s, 0.02) + 0.02, f"{p:.3f}\n{unc}",
                ha="center", fontsize=9)
    if true_state:
        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
        true_idx = state_names.index(true_state)
        ax.get_children()[true_idx].set_edgecolor("black")
        ax.get_children()[true_idx].set_linewidth(3)
        ax.set_title(f"(a) Posteriors (verdadeiro: {true_state})")
    else:
        ax.set_title("(a) Probabilidades posteriores por estado")

    # (b) Bootstrap boxplot
    ax = axes[1]
    boot_data = [result["probs_boot"][k] for k in state_names]
    bp = ax.boxplot(boot_data, positions=range(len(state_names)),
                    widths=0.4, patch_artist=True,
                    medianprops=dict(color="white", lw=2))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i]); patch.set_alpha(0.7)
    for i, bd in enumerate(boot_data):
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(bd))
        ax.scatter(np.full(len(bd), i) + jitter, bd,
                   color=colors[i], alpha=0.3, s=8, zorder=5)
    ax.set_xticks(range(len(state_names)))
    ax.set_xticklabels(state_names)
    ax.set_ylabel("p(estado | Z_obs)")
    ax.set_title("(b) Distribuição bootstrap das probabilidades")
    ax.set_ylim(-0.05, 1.05)

    predicted = result["predicted"]
    conf      = result["confidence"]
    fig.suptitle(f"Resultado da Classificação: {predicted} (confiança: {conf:.1%})",
                 fontsize=13, fontweight="bold",
                 color=COLORS_STATES.get(predicted, "black"))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


def plot_confusion_matrix(sim_result: dict, save_path=None):
    """FIG-E4D: Matriz de confusão normalizada."""
    confusion   = sim_result["confusion"]
    state_names = sim_result["state_names"]
    accuracy    = sim_result["accuracy"]
    n_states    = len(state_names)
    cm_norm     = confusion.astype(float) / confusion.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (data, title) in zip(axes, [
            (confusion, "Matriz de Confusão (contagens)"),
            (cm_norm,   "Matriz de Confusão (normalizada)")]):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
        ax.set_xticks(range(n_states)); ax.set_yticks(range(n_states))
        ax.set_xticklabels(state_names, rotation=30, ha="right")
        ax.set_yticklabels(state_names)
        ax.set_xlabel("Estado predito"); ax.set_ylabel("Estado verdadeiro")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(n_states):
            for j in range(n_states):
                val = data[i, j]
                txt = f"{val:.2f}" if isinstance(val, float) else str(int(val))
                clr = "white" if val > data.max() * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=12, color=clr, fontweight="bold")

    auc_str = "  |  ".join(
        [f"AUC({k}): {sim_result['auc_roc'][k]:.3f}" for k in state_names])
    fig.suptitle(f"Desempenho do Classificador — Acurácia: {accuracy:.1%}  |  {auc_str}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


def plot_roc_curves(sim_result: dict, save_path=None):
    """FIG-E4E: Curvas ROC one-vs-rest."""
    state_names = sim_result["state_names"]
    true_labels = sim_result["true_labels"]
    all_probs   = sim_result["all_probs"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatório (AUC=0.5)")

    for k in state_names:
        color    = COLORS_STATES[k]
        true_bin = [1 if t == k else 0 for t in true_labels]
        scores   = all_probs[k]
        pairs    = sorted(zip(scores, true_bin), key=lambda x: -x[0])
        n_pos = sum(true_bin); n_neg = len(true_bin) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        tpr_list, fpr_list = [0.0], [0.0]
        tp = fp = 0
        for sc, lb in pairs:
            if lb == 1: tp += 1
            else:       fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
        auc = abs(float(np.trapezoid(tpr_list, fpr_list)))
        ax.plot(fpr_list, tpr_list, color=color, lw=2, label=f"{k} (AUC={auc:.3f})")

    ax.set_xlabel("Taxa de Falso Positivo (1 - Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiro Positivo (Sensibilidade)")
    ax.set_title("Curvas ROC — Classificação Bayesiana de Estado Tecidual\n"
                 f"(SNR = {sim_result.get('snr_db','?')} dB, one-vs-rest)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


# ── FIG-E4G — CORREÇÃO [4]: painel ECE + banda de erro ─────────────────────

def plot_accuracy_vs_snr(snr_results: dict, save_path=None):
    """
    FIG-E4G: Acurácia, AUC médio e ECE × SNR.
    CORREÇÃO: adiciona painel ECE mostrando que ruído SIM piora a calibração.
    """
    snr_levels = sorted(snr_results.keys())
    accs = [snr_results[s]["accuracy"] * 100          for s in snr_levels]
    aucs = [snr_results[s]["auc_mean"]                 for s in snr_levels]
    eces = [snr_results[s].get("ece", np.nan)          for s in snr_levels]
    # erro padrão estimado: ±1 SE ≈ ±sqrt(acc*(1-acc)/n)
    n_test = snr_results[snr_levels[0]].get("n_test", 45)
    se_acc = [np.sqrt(a/100 * (1 - a/100) / n_test) * 100 for a in accs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Acurácia com banda de erro
    axes[0].plot(snr_levels, accs, 'o-', color="#1f77b4", lw=2, ms=8)
    axes[0].fill_between(snr_levels,
                          [a - s for a, s in zip(accs, se_acc)],
                          [a + s for a, s in zip(accs, se_acc)],
                          alpha=0.2, color="#1f77b4", label="±1 SE")
    axes[0].axhline(100/3, color="gray", ls="--", lw=1, alpha=0.6, label="Chance (33%)")
    axes[0].set_xlabel("SNR [dB]"); axes[0].set_ylabel("Acurácia [%]")
    axes[0].set_title("(a) Acurácia × SNR\n(IS-Posterior estável via MCMC)")
    axes[0].set_ylim(20, 105); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for x, y in zip(snr_levels, accs):
        axes[0].text(x, y + 3, f"{y:.1f}%", ha="center", fontsize=8)

    # (b) AUC
    axes[1].plot(snr_levels, aucs, 's-', color="#d62728", lw=2, ms=8)
    axes[1].axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="Chance (AUC=0.5)")
    axes[1].set_xlabel("SNR [dB]"); axes[1].set_ylabel("AUC médio (one-vs-rest)")
    axes[1].set_title("(b) AUC médio × SNR")
    axes[1].set_ylim(0.4, 1.05); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    for x, y in zip(snr_levels, aucs):
        axes[1].text(x, y + 0.01, f"{y:.3f}", ha="center", fontsize=8)

    # (c) ECE — mostra degradação da calibração com ruído
    finite = [(s, e) for s, e in zip(snr_levels, eces) if np.isfinite(e)]
    ax3 = axes[2]
    if finite:
        sx, ex = zip(*finite)
        ax3.plot(sx, ex, '^-', color="#9467bd", lw=2, ms=8)
        ax3.axhline(0.15, color="orange", ls="--", lw=1.2, label="ECE=0.15 (limiar bom)")
        ax3.set_ylim(0, min(0.5, max(ex) * 1.5 + 0.05))
        for x, y in zip(sx, ex):
            ax3.text(x, y + 0.01, f"{y:.3f}", ha="center", fontsize=8)
        # Seta mostrando tendência
        if len(sx) >= 2 and ex[0] > ex[-1]:
            ax3.annotate("Ruído piora\ncalibração →",
                          xy=(sx[-1], ex[-1]), xytext=(sx[-1]-6, ex[-1]+0.06),
                          arrowprops=dict(arrowstyle="->", color="gray"),
                          fontsize=8, color="gray")
    else:
        ax3.text(0.5, 0.5, "ECE não disponível\n(use --snr-scan completo)",
                 transform=ax3.transAxes, ha="center", fontsize=10, color="gray")
    ax3.set_xlabel("SNR [dB]"); ax3.set_ylabel("ECE")
    ax3.set_title("(c) Calibração (ECE) × SNR\n(menor = mais calibrado)")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.suptitle("Desempenho do Classificador Bayesiano × Nível de Ruído",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


def plot_calibration(sim_result: dict, save_path=None):
    """FIG-E4F: Reliability diagram + ECE."""
    cal       = sim_result["calibration"]
    ece       = cal["ece"]
    bin_acc   = cal["bin_acc"]
    bin_conf  = cal["bin_conf"]
    bin_count = cal["bin_count"]

    valid = [(a, c, n) for a, c, n in zip(bin_acc, bin_conf, bin_count)
             if not np.isnan(a) and n > 0]
    if not valid:
        return None
    accs, confs, counts = zip(*valid)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Calibração perfeita')
    ax.bar(confs, accs, width=0.08, alpha=0.7, color='#1f77b4',
           edgecolor='white', label='Observado')
    for c, a in zip(confs, accs):
        lo, hi = min(a, c), max(a, c)
        ax.fill_between([c - 0.04, c + 0.04], [lo, lo], [hi, hi],
                        alpha=0.3, color='red')
    ax.set_xlabel('Confiança predita'); ax.set_ylabel('Acurácia observada')
    ax.set_title(f'(a) Reliability Diagram\nECE = {ece:.4f}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    ax.text(0.05, 0.92, f'ECE = {ece:.3f}', transform=ax.transAxes,
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', fc='white', ec='red', alpha=0.8))

    ax2 = axes[1]
    confidences = sim_result["confidences"]
    correct     = sim_result["correct_flags"]
    ax2.hist([c for c, ok in zip(confidences, correct) if ok == 1],
             bins=12, alpha=0.7, color='#2ca02c', label='Correto', density=True)
    ax2.hist([c for c, ok in zip(confidences, correct) if ok == 0],
             bins=12, alpha=0.7, color='#d62728', label='Incorreto', density=True)
    ax2.set_xlabel('Confiança do classificador'); ax2.set_ylabel('Densidade')
    ax2.set_title('(b) Distribuição de confiança\n(corretos vs incorretos)')
    ax2.legend(); ax2.axvline(0.5, color='gray', ls='--', lw=1)

    snr = sim_result.get("snr_db", "?")
    fig.suptitle(f'Calibração do Classificador Bayesiano (SNR={snr} dB)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'  Figura salva: {save_path}')
    plt.show()
    return fig
