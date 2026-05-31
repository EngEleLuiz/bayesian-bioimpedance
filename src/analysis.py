"""
analysis.py
===========
Diagnósticos bayesianos, estatísticas posteriores e visualizações.

Implementa:
    - HDI (Highest Density Interval) — intervalo de credibilidade
    - Resumo posterior completo (média, mediana, std, HDI)
    - Propagação de incerteza para grandezas derivadas
    - Todos os gráficos do pipeline

Referências:
    Kruschke, J.K. (2015). Doing Bayesian Data Analysis (2nd ed.).
    Academic Press. — HDI e visualizações.

    Gelman, A. et al. (2013). Bayesian Data Analysis (3rd ed.). CRC.

    McElreath, R. (2020). Statistical Rethinking (2nd ed.). CRC Press.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from .cole_model import (cole_cole_impedance, characteristic_frequency,
                          delta_R as compute_deltaR)
from .mcmc_sampler import compute_ess


# ---------------------------------------------------------------------------
# Estilo global dos gráficos
# ---------------------------------------------------------------------------

COLORS = {
    "bayes":   "#1f77b4",   # azul — posterior Bayesiana
    "ls":      "#d62728",   # vermelho — NLLS
    "true":    "#2ca02c",   # verde — valor verdadeiro
    "data":    "#7f7f7f",   # cinza — dados observados
    "hdi":     "#aec7e8",   # azul claro — região HDI
    "fit":     "#ff7f0e",   # laranja — ajuste
}

PARAM_LABELS = {
    "R_inf":  r"$R_\infty$ [Ω]",
    "delta_R": r"$\Delta R = R_0 - R_\infty$ [Ω]",
    "log_tau": r"$\ln(\tau)$",
    "alpha":   r"$\alpha$",
}

PARAM_LABELS_DERIVED = {
    "R0":     r"$R_0$ [Ω]",
    "tau":    r"$\tau$ [s]",
    "f_c":    r"$f_c$ [Hz]",
    "delta_R": r"$\Delta R$ [Ω]",
}

plt.rcParams.update({
    "figure.dpi":       120,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "lines.linewidth":  1.8,
})


# ---------------------------------------------------------------------------
# HDI (Highest Density Interval)
# ---------------------------------------------------------------------------

def compute_hdi(samples: np.ndarray, credibility: float = 0.95) -> tuple:
    """
    Calcula o Highest Density Interval (HDI) de uma amostra unidimensional.

    O HDI é o intervalo mais estreito que contém `credibility` da massa
    posterior. Para distribuições simétricas coincide com o percentil;
    para assimétricas é mais informativo (Kruschke, 2015, Cap.3).

    Algoritmo: ordena as amostras e desliza uma janela de tamanho
    proporcional à credibilidade, escolhendo a janela mais estreita.

    Parâmetros
    ----------
    samples     : array 1D de amostras
    credibility : fração de credibilidade (0.95 por padrão)

    Retorna
    -------
    (hdi_low, hdi_high)
    """
    s = np.sort(samples)
    n = len(s)
    n_included = int(np.floor(credibility * n))
    widths = s[n_included:] - s[:n - n_included]
    idx_min = np.argmin(widths)
    return float(s[idx_min]), float(s[idx_min + n_included])


def posterior_summary(samples: np.ndarray,
                      param_names: list,
                      true_params: dict = None,
                      credibility: float = 0.95) -> dict:
    """
    Resumo estatístico completo da distribuição posterior.

    Retorna um dicionário com: média, mediana, std, HDI, ESS para cada parâmetro.
    """
    summary = {}
    for i, name in enumerate(param_names):
        s = samples[:, i]
        hdi_l, hdi_h = compute_hdi(s, credibility)
        ess = compute_ess(s)
        entry = {
            "mean":   float(s.mean()),
            "median": float(np.median(s)),
            "std":    float(s.std()),
            "hdi_low":  hdi_l,
            "hdi_high": hdi_h,
            "hdi_width": hdi_h - hdi_l,
            "ess":    float(ess),
        }
        if true_params and name in true_params:
            true_val = true_params[name]
            entry["true"] = float(true_val)
            entry["bias_pct"] = float((s.mean() - true_val) / abs(true_val) * 100)
            entry["in_hdi"] = bool(hdi_l <= true_val <= hdi_h)
        summary[name] = entry
    return summary


def print_summary_table(summary: dict, title: str = "Resumo Posterior"):
    """Imprime tabela formatada do resumo posterior."""
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    header = f"{'Parâmetro':>12} {'Média':>10} {'Mediana':>10} {'Std':>9} "
    header += f"{'HDI 95% low':>12} {'HDI 95% high':>13} {'ESS':>7}"
    if any("true" in v for v in summary.values()):
        header += f"  {'Viés%':>8}  {'∈HDI':>5}"
    print(header)
    print("-" * 72)
    for name, s in summary.items():
        row = (f"{name:>12} {s['mean']:>10.4g} {s['median']:>10.4g} "
               f"{s['std']:>9.4g} {s['hdi_low']:>12.4g} {s['hdi_high']:>13.4g} "
               f"{s['ess']:>7.0f}")
        if "true" in s:
            row += f"  {s['bias_pct']:>7.2f}%  {'✓' if s['in_hdi'] else '✗':>5}"
        print(row)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Conversão de amostras: espaço MCMC → espaço físico
# ---------------------------------------------------------------------------

def samples_to_physical(samples: np.ndarray) -> dict:
    """
    Converte amostras MCMC do espaço de parametrização para espaço físico.

    Espaço MCMC:  [R_inf, delta_R, log_tau, alpha]
    Espaço físico:[R_inf, R0, tau, alpha, f_c, delta_R]

    Propaga incerteza automaticamente para grandezas derivadas.
    """
    R_inf  = samples[:, 0]
    dR     = samples[:, 1]
    log_tau = samples[:, 2]
    alpha  = samples[:, 3]

    R0  = R_inf + dR
    tau = np.exp(log_tau)
    f_c = characteristic_frequency(tau, alpha)

    return {
        "R_inf":   R_inf,
        "R0":      R0,
        "tau":     tau,
        "alpha":   alpha,
        "delta_R": dR,
        "f_c":     f_c,
        "log_tau": log_tau,
    }


# ---------------------------------------------------------------------------
# FIGURA 1 — Espectro EIS: dados, ajuste e incerteza posterior
# ---------------------------------------------------------------------------

def plot_eis_spectrum(f, Z_obs, samples, true_params=None,
                      ls_result=None, title="Espectro EIS — Posterior Bayesiana",
                      save_path=None):
    """
    Plota o espectro EIS com banda de credibilidade 95% da posterior.

    Painéis: (a) Nyquist, (b) Módulo × frequência, (c) Fase × frequência.
    """
    phys = samples_to_physical(samples)

    # Computar envelope do posterior: amostragem de N_env curvas
    N_env = min(500, len(samples))
    idx = np.random.choice(len(samples), N_env, replace=False)
    Z_curves = []
    for i in idx:
        Ri = phys["R_inf"][i]
        R0i = phys["R0"][i]
        ti  = phys["tau"][i]
        ai  = phys["alpha"][i]
        try:
            Zc = cole_cole_impedance(f, Ri, R0i, ti, ai)
            Z_curves.append(Zc)
        except Exception:
            pass
    Z_curves = np.array(Z_curves)

    # Percentis do envelope
    Z_mod_lo = np.percentile(np.abs(Z_curves), 2.5, axis=0)
    Z_mod_hi = np.percentile(np.abs(Z_curves), 97.5, axis=0)
    Z_phase_lo = np.percentile(np.angle(Z_curves, deg=True), 2.5, axis=0)
    Z_phase_hi = np.percentile(np.angle(Z_curves, deg=True), 97.5, axis=0)

    # Curva da mediana posterior
    R_med  = np.median(phys["R_inf"])
    R0_med = np.median(phys["R0"])
    t_med  = np.median(phys["tau"])
    a_med  = np.median(phys["alpha"])
    Z_med  = cole_cole_impedance(f, R_med, R0_med, t_med, a_med)

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ----- (a) Nyquist -----
    ax1.scatter(Z_obs.real, -Z_obs.imag, color=COLORS["data"],
                s=25, zorder=5, label="Dados observados", alpha=0.8)
    ax1.plot(Z_med.real, -Z_med.imag,
             color=COLORS["bayes"], label="Mediana posterior")
    # Envelope Nyquist via curvas individuais
    for Zc in Z_curves[::5]:
        ax1.plot(Zc.real, -Zc.imag, color=COLORS["bayes"], alpha=0.04, lw=0.8)

    if true_params:
        Zt = cole_cole_impedance(f, **true_params)
        ax1.plot(Zt.real, -Zt.imag, '--', color=COLORS["true"],
                 label="Valor verdadeiro", zorder=6)
    if ls_result and ls_result.get("converged"):
        Zls = ls_result["Z_fitted"]
        ax1.plot(Zls.real, -Zls.imag, ':', color=COLORS["ls"],
                 label="NLLS", zorder=7)
    ax1.set_xlabel(r"$\Re[Z]$ [Ω]")
    ax1.set_ylabel(r"$-\Im[Z]$ [Ω]")
    ax1.set_title("(a) Diagrama de Nyquist")
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal", adjustable="datalim")

    # ----- (b) Módulo × f -----
    ax2.fill_between(f, Z_mod_lo, Z_mod_hi,
                     alpha=0.3, color=COLORS["hdi"], label="HDI 95%")
    ax2.semilogx(f, np.abs(Z_obs), 'o', color=COLORS["data"],
                 ms=4, label="Dados")
    ax2.semilogx(f, np.abs(Z_med), color=COLORS["bayes"], label="Mediana")
    if true_params:
        ax2.semilogx(f, np.abs(cole_cole_impedance(f, **true_params)),
                     '--', color=COLORS["true"], label="Verdadeiro")
    ax2.set_xlabel("Frequência [Hz]")
    ax2.set_ylabel(r"$|Z|$ [Ω]")
    ax2.set_title("(b) Módulo da Impedância")
    ax2.legend(fontsize=8)
    ax2.set_xscale("log")

    # ----- (c) Fase × f -----
    ax3.fill_between(f, Z_phase_lo, Z_phase_hi,
                     alpha=0.3, color=COLORS["hdi"], label="HDI 95%")
    ax3.semilogx(f, np.angle(Z_obs, deg=True), 'o',
                 color=COLORS["data"], ms=4, label="Dados")
    ax3.semilogx(f, np.angle(Z_med, deg=True),
                 color=COLORS["bayes"], label="Mediana")
    if true_params:
        ax3.semilogx(f, np.angle(cole_cole_impedance(f, **true_params), deg=True),
                     '--', color=COLORS["true"], label="Verdadeiro")
    ax3.set_xlabel("Frequência [Hz]")
    ax3.set_ylabel("Fase [°]")
    ax3.set_title("(c) Fase")
    ax3.legend(fontsize=8)
    ax3.set_xscale("log")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# FIGURA 2 — Distribuições posteriores dos parâmetros
# ---------------------------------------------------------------------------

def plot_posterior_distributions(samples, true_params=None,
                                  ls_result=None, title="Distribuições Posteriores",
                                  save_path=None):
    """
    Plota as distribuições posteriores marginais (KDE) de cada parâmetro.
    Inclui HDI 95%, valor verdadeiro e estimativa NLLS para comparação.
    """
    phys = samples_to_physical(samples)
    params_to_plot = ["R_inf", "R0", "tau", "alpha", "f_c", "delta_R"]
    labels = {
        "R_inf":   r"$R_\infty$ [Ω]",
        "R0":      r"$R_0$ [Ω]",
        "tau":     r"$\tau$ [s]",
        "alpha":   r"$\alpha$",
        "f_c":     r"$f_c$ [Hz]",
        "delta_R": r"$\Delta R$ [Ω]",
    }
    true_map = {}
    if true_params:
        true_map = {
            "R_inf":   true_params["R_inf"],
            "R0":      true_params["R0"],
            "tau":     true_params["tau"],
            "alpha":   true_params["alpha"],
            "f_c":     characteristic_frequency(true_params["tau"], true_params["alpha"]),
            "delta_R": true_params["R0"] - true_params["R_inf"],
        }

    ls_map = {}
    if ls_result and ls_result.get("converged") and ls_result.get("params"):
        p = ls_result["params"]
        s = ls_result["std"]
        ls_map = {
            "R_inf":   (p["R_inf"],   s["R_inf"]),
            "R0":      (p["R0"],      s["R0"]),
            "tau":     (p["tau"],     s["tau"]),
            "alpha":   (p["alpha"],   s["alpha"]),
            "f_c":     (ls_result["derived"]["f_c"], np.nan),
            "delta_R": (ls_result["derived"]["delta_R"], np.nan),
        }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for ax, name in zip(axes, params_to_plot):
        s = phys[name]
        hdi_l, hdi_h = compute_hdi(s)

        # KDE suavizada
        kde = gaussian_kde(s, bw_method="scott")
        x_range = np.linspace(s.min(), s.max(), 400)
        y_kde = kde(x_range)

        # Pintar a região HDI
        mask = (x_range >= hdi_l) & (x_range <= hdi_h)
        ax.fill_between(x_range, y_kde, where=mask,
                        color=COLORS["hdi"], alpha=0.7, label="HDI 95%")
        ax.plot(x_range, y_kde, color=COLORS["bayes"], lw=2, label="Posterior")

        # Mediana
        med = np.median(s)
        ax.axvline(med, color=COLORS["bayes"], ls="--", lw=1.5, alpha=0.8)

        # Valor verdadeiro
        if name in true_map:
            ax.axvline(true_map[name], color=COLORS["true"],
                       ls="-", lw=2, label=f"Verdadeiro: {true_map[name]:.4g}")

        # NLLS
        if name in ls_map:
            val_ls, std_ls = ls_map[name]
            ax.axvline(val_ls, color=COLORS["ls"], ls=":",
                       lw=2, label=f"NLLS: {val_ls:.4g}")
            if np.isfinite(std_ls):
                ax.axvspan(val_ls - std_ls, val_ls + std_ls,
                           alpha=0.15, color=COLORS["ls"])

        ax.set_xlabel(labels.get(name, name))
        ax.set_ylabel("Densidade")
        ax.set_title(f"{labels.get(name, name)}")
        ax.legend(fontsize=8)

        # Nota do HDI
        ax.text(0.98, 0.95,
                f"HDI: [{hdi_l:.4g}, {hdi_h:.4g}]",
                transform=ax.transAxes, fontsize=8,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# FIGURA 3 — Trace plots e diagnóstico de convergência
# ---------------------------------------------------------------------------

def plot_trace(samples, warmup=None, param_names=None, save_path=None):
    """
    Trace plots (cadeia MCMC) + autocorrelação para diagnóstico visual.
    """
    if param_names is None:
        param_names = [r"$R_\infty$", r"$\Delta R$", r"$\ln\tau$", r"$\alpha$"]

    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 3 * n_params))

    for i in range(n_params):
        ax_trace = axes[i, 0]
        ax_acf   = axes[i, 1]

        # --- Trace ---
        if warmup is not None:
            ax_trace.plot(np.arange(len(warmup)),
                          warmup[:, i], color="lightgray", lw=0.5, alpha=0.7)
            offset = len(warmup)
        else:
            offset = 0
        ax_trace.plot(np.arange(offset, offset + len(samples)),
                      samples[:, i], color=COLORS["bayes"], lw=0.6)
        ax_trace.set_ylabel(param_names[i])
        ax_trace.set_xlabel("Iteração")
        if i == 0:
            ax_trace.set_title("Trace (cinza=warm-up, azul=pós-warm-up)")

        # --- Autocorrelação ---
        s = samples[:, i] - samples[:, i].mean()
        n = len(s)
        max_lag = min(100, n // 2)
        acf_vals = np.correlate(s, s, mode="full")[n - 1:]
        acf_vals = acf_vals[:max_lag] / acf_vals[0]
        lags = np.arange(max_lag)
        ax_acf.bar(lags, acf_vals, color=COLORS["bayes"], width=0.8, alpha=0.7)
        ax_acf.axhline(0, color="black", lw=0.8)
        ax_acf.axhline(1.96 / np.sqrt(n), color="red", ls="--", lw=0.8)
        ax_acf.axhline(-1.96 / np.sqrt(n), color="red", ls="--", lw=0.8)
        ax_acf.set_ylim(-0.3, 1.1)
        ax_acf.set_xlabel("Lag")
        if i == 0:
            ax_acf.set_title("Autocorrelação (linhas vermelhas: IC 95%)")
        ess = compute_ess(samples[:, i])
        ax_acf.text(0.97, 0.93, f"ESS={ess:.0f}",
                    transform=ax_acf.transAxes, ha="right", va="top",
                    fontsize=9, color=COLORS["bayes"])

    fig.suptitle("Diagnóstico MCMC — Trace e Autocorrelação",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# FIGURA 4 — Estudo de sensibilidade: HDI width × SNR
# ---------------------------------------------------------------------------

def plot_snr_sensitivity(snr_levels, hdi_widths_by_param,
                          param_names_display, title="Sensibilidade ao SNR",
                          save_path=None):
    """
    Plota como a largura do HDI 95% de cada parâmetro varia com o SNR.

    hdi_widths_by_param : dict {nome_param: array de larguras por SNR}
    """
    n_params = len(param_names_display)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4), sharey=False)
    if n_params == 1:
        axes = [axes]

    cmap = plt.cm.viridis(np.linspace(0.2, 0.85, n_params))

    for ax, (name, label), color in zip(axes, param_names_display.items(), cmap):
        widths = np.array(hdi_widths_by_param[name])
        ax.plot(snr_levels, widths, 'o-', color=color, lw=2, ms=7)
        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("Largura do HDI 95%")
        ax.set_title(label)
        ax.grid(alpha=0.3)

        # Linha de referência: 10% da mediana dos valores
        ax.annotate("↓ SNR melhora", xy=(max(snr_levels)*0.8, widths[-1]),
                    fontsize=8, color="gray")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# FIGURA 5 — Comparação Bayes vs NLLS
# ---------------------------------------------------------------------------

def plot_bayes_vs_nlls(samples, ls_result, true_params,
                        save_path=None):
    """
    Comparação lado-a-lado de intervalos de incerteza: Bayes vs NLLS.
    """
    phys = samples_to_physical(samples)
    params = ["R_inf", "R0", "tau", "alpha"]
    labels = [r"$R_\infty$ [Ω]", r"$R_0$ [Ω]", r"$\tau$ [s]", r"$\alpha$"]

    true_vals = {
        "R_inf": true_params["R_inf"],
        "R0":    true_params["R0"],
        "tau":   true_params["tau"],
        "alpha": true_params["alpha"],
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for ax, name, label in zip(axes, params, labels):
        s = phys[name]
        hdi_l, hdi_h = compute_hdi(s)
        med = np.median(s)
        true_val = true_vals[name]

        # Normalizar para o valor verdadeiro (desvio relativo)
        rel = lambda x: (x - true_val) / abs(true_val) * 100

        ax.barh(0.7, rel(hdi_h) - rel(hdi_l),
                left=rel(hdi_l), height=0.3,
                color=COLORS["hdi"], edgecolor=COLORS["bayes"], lw=1.5,
                label="Bayes HDI 95%")
        ax.plot(rel(med), 0.7, 'D', color=COLORS["bayes"], ms=8, zorder=5)

        if ls_result and ls_result.get("converged"):
            p = ls_result["params"]
            s_ls = ls_result["std"]
            val_ls = p[name]
            std_ls = s_ls[name]
            ax.barh(0.3, 2 * rel(val_ls + std_ls) - 2 * rel(val_ls),
                    left=rel(val_ls - std_ls), height=0.3,
                    color="#ffb3b3", edgecolor=COLORS["ls"], lw=1.5,
                    label="NLLS ± 1σ")
            ax.plot(rel(val_ls), 0.3, 's', color=COLORS["ls"], ms=8, zorder=5)

        ax.axvline(0, color=COLORS["true"], lw=2, ls="--", label="Verdadeiro")
        ax.set_xlabel("Desvio relativo [%]")
        ax.set_title(label)
        ax.set_yticks([0.3, 0.7])
        ax.set_yticklabels(["NLLS", "Bayes"])
        ax.legend(fontsize=8)

    fig.suptitle("Comparação de Incerteza: Bayes vs NLLS",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# FIGURA 6 — Matriz de correlação posterior (corner-plot simplificado)
# ---------------------------------------------------------------------------

def plot_corner(samples, param_names=None, true_params=None,
                save_path=None):
    """
    Corner-plot: distribuições marginais e correlações 2D (sem dependência externa).
    """
    if param_names is None:
        param_names = [r"$R_\infty$", r"$\Delta R$", r"$\ln\tau$", r"$\alpha$"]

    n = samples.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(10, 10))

    true_list = None
    if true_params:
        dR_true = true_params["R0"] - true_params["R_inf"]
        log_tau_true = np.log(true_params["tau"])
        true_list = [true_params["R_inf"], dR_true, log_tau_true, true_params["alpha"]]

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histograma marginal
                ax.hist(samples[:, i], bins=40, density=True,
                        color=COLORS["bayes"], alpha=0.7, edgecolor="white", lw=0.3)
                if true_list:
                    ax.axvline(true_list[i], color=COLORS["true"], lw=2)
                ax.set_xlabel(param_names[i] if i == n-1 else "")
                ax.set_yticks([])
            elif i > j:
                # Abaixo da diagonal: scatter 2D com densidade
                ax.scatter(samples[::5, j], samples[::5, i],
                           s=0.8, alpha=0.2, color=COLORS["bayes"])
                if true_list:
                    ax.plot(true_list[j], true_list[i], '*',
                            color=COLORS["true"], ms=12, zorder=10)
                if i == n-1:
                    ax.set_xlabel(param_names[j])
                if j == 0:
                    ax.set_ylabel(param_names[i])
            else:
                ax.set_visible(False)

    fig.suptitle("Corner Plot — Posterior Conjunta", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig
