"""
main.py
=======
Pipeline completo — Estimação Bayesiana de Parâmetros de Bioimpedância
com Quantificação de Incerteza.

Executa sequencialmente:
    Etapa 1 — Geração de dados sintéticos (modelo Cole-Cole + ruído)
    Etapa 2 — Ajuste NLLS (referência frequentista)
    Etapa 3 — Inferência Bayesiana via AM-MCMC
    Etapa 4 — Diagnósticos e resumo posterior
    Etapa 5 — Estudo de sensibilidade: HDI × SNR
    Etapa 6 — Todas as figuras do artigo

Uso:
    python main.py                    # SNR padrão = 30 dB
    python main.py --snr 25           # SNR customizado
    python main.py --tecido gordura   # outro tecido
    python main.py --rapido           # menos amostras (para teste rápido)
    python main.py --sem-snr          # pula o estudo de sensibilidade (lento)

Referências:
    Grimnes, S. & Martinsen, Ø.G. (2015). Bioimpedance and Bioelectricity Basics.
    Barsoukov, E. & Macdonald, J.R. (2018). Impedance Spectroscopy. Wiley.
    Haario, H. et al. (2001). An adaptive Metropolis algorithm. Bernoulli, 7(2).
    Gelman, A. et al. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.
    Huang, J. et al. (2021). Hierarchical Bayesian EIS inversion. Electrochim. Acta.
"""

import argparse
import time
import warnings
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sem janela (salva direto em arquivo)
import matplotlib.pyplot as plt

# --- Módulos do projeto ---
from src.cole_model       import TISSUE_PARAMS, characteristic_frequency
from src.data_generation  import frequency_grid, generate_eis_data, generate_multiple_snr
from src.ls_fitting       import nlls_fit
from src.mcmc_sampler     import (AdaptiveMCMC, log_posterior,
                                   initial_theta_from_nlls, compute_rhat, compute_ess)
from src.analysis         import (samples_to_physical, posterior_summary,
                                   print_summary_table, compute_hdi,
                                   plot_eis_spectrum, plot_posterior_distributions,
                                   plot_trace, plot_snr_sensitivity,
                                   plot_bayes_vs_nlls, plot_corner)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===========================================================================
# Configuração da linha de comando
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimação Bayesiana de Bioimpedância (Cole-Cole)")
    parser.add_argument("--snr",      type=float, default=30.0,
                        help="SNR em dB (padrão: 30)")
    parser.add_argument("--tecido",   type=str,   default="musculo",
                        choices=list(TISSUE_PARAMS.keys()),
                        help="Tecido de referência")
    parser.add_argument("--n-freq",   type=int,   default=30,
                        help="Número de pontos de frequência (padrão: 30)")
    parser.add_argument("--n-amostras", type=int, default=5000,
                        help="Amostras MCMC pós warm-up (padrão: 5000)")
    parser.add_argument("--n-warmup",   type=int, default=2000,
                        help="Amostras de warm-up (padrão: 2000)")
    parser.add_argument("--seed",     type=int,   default=42,
                        help="Semente aleatória")
    parser.add_argument("--rapido",   action="store_true",
                        help="Modo rápido: menos amostras (para teste)")
    parser.add_argument("--sem-snr",  action="store_true",
                        help="Pula o estudo de sensibilidade ao SNR")
    parser.add_argument("--saida",    type=str,   default="resultados",
                        help="Pasta de saída das figuras")
    return parser.parse_args()


# ===========================================================================
# Etapa 1 — Geração de dados sintéticos
# ===========================================================================

def etapa1_dados(tecido, snr_db, n_freq, seed):
    print("\n" + "="*60)
    print(" ETAPA 1 — Geração de Dados Sintéticos")
    print("="*60)

    params = TISSUE_PARAMS[tecido]
    true_params = {k: v for k, v in params.items()
                   if k in ("R_inf", "R0", "tau", "alpha")}

    print(f"  Tecido       : {tecido}")
    print(f"  Fonte        : {params['fonte']}")
    print(f"  R_inf        : {true_params['R_inf']:.1f} Ω")
    print(f"  R0           : {true_params['R0']:.1f} Ω")
    print(f"  tau          : {true_params['tau']:.2e} s")
    print(f"  alpha        : {true_params['alpha']:.3f}")
    print(f"  f_c          : {characteristic_frequency(true_params['tau'], true_params['alpha']):.1f} Hz")
    print(f"  SNR          : {snr_db:.1f} dB")
    print(f"  Pontos freq. : {n_freq}")

    f = frequency_grid(100.0, 1e6, n_freq)
    data = generate_eis_data(f, seed=seed, snr_db=snr_db, **true_params)

    print(f"\n  |Z| range  : [{np.abs(data['Z_noisy']).min():.1f}, "
          f"{np.abs(data['Z_noisy']).max():.1f}] Ω")
    print(f"  σ_noise    : {data['sigma']*100:.2f}%")
    return f, data, true_params


# ===========================================================================
# Etapa 2 — Ajuste NLLS
# ===========================================================================

def etapa2_nlls(f, Z_obs):
    print("\n" + "="*60)
    print(" ETAPA 2 — Ajuste por Mínimos Quadrados (NLLS)")
    print("="*60)

    t0 = time.time()
    result = nlls_fit(f, Z_obs, method="complex", n_restarts=5)
    dt = time.time() - t0

    if result.get("converged"):
        p = result["params"]
        s = result["std"]
        print(f"  Convergiu em {dt:.2f}s")
        print(f"  R_inf  = {p['R_inf']:.4g} ± {s['R_inf']:.4g} Ω")
        print(f"  R0     = {p['R0']:.4g} ± {s['R0']:.4g} Ω")
        print(f"  tau    = {p['tau']:.4g} ± {s['tau']:.4g} s")
        print(f"  alpha  = {p['alpha']:.4f} ± {s['alpha']:.4f}")
        print(f"  f_c    = {result['derived']['f_c']:.2f} Hz")
        print(f"  chi2   = {result['chi2']:.4g}")
    else:
        print("  [AVISO] NLLS não convergiu. Usando apenas Bayesiana.")

    return result


# ===========================================================================
# Etapa 3 — Inferência Bayesiana (AM-MCMC)
# ===========================================================================

def etapa3_mcmc(f, Z_obs, snr_db, n_amostras, n_warmup, seed, ls_result=None):
    print("\n" + "="*60)
    print(" ETAPA 3 — Inferência Bayesiana (AM-MCMC)")
    print("="*60)
    print(f"  Amostras pós warm-up : {n_amostras}")
    print(f"  Warm-up              : {n_warmup}")

    # Ponto inicial via NLLS (MAP aproximado) — acelera convergência
    if ls_result and ls_result.get("converged"):
        p = ls_result["params"]
        theta0 = np.array([
            p["R_inf"],
            p["R0"] - p["R_inf"],
            np.log(p["tau"]),
            p["alpha"]
        ])
        print(f"  Inicialização        : via NLLS (MAP)")
    else:
        theta0 = initial_theta_from_nlls(f, Z_obs, snr_db)
        print(f"  Inicialização        : fallback fisiológico")

    print(f"  θ0 = [R_inf={theta0[0]:.2f}, ΔR={theta0[1]:.2f}, "
          f"ln(τ)={theta0[2]:.3f}, α={theta0[3]:.4f}]")

    # Verificação do log-posterior no ponto inicial
    lp0 = log_posterior(theta0, f, Z_obs, snr_db)
    if not np.isfinite(lp0):
        print(f"  [AVISO] log-posterior inválido em θ0 (lp={lp0}). Ajustando...")
        theta0 = np.array([50.0, 150.0, np.log(7.96e-6), 0.75])

    # Executar duas cadeias independentes para R-hat
    cadeias = []
    for chain_id in range(2):
        print(f"\n  --- Cadeia {chain_id + 1}/2 ---")
        perturbacao = np.array([
            theta0[0] * (1.0 + 0.1 * chain_id),
            theta0[1] * (1.0 - 0.05 * chain_id),
            theta0[2] + 0.1 * chain_id,
            theta0[3] - 0.02 * chain_id
        ])
        sampler = AdaptiveMCMC(
            n_samples=n_amostras,
            n_warmup=n_warmup,
            adapt_start=500,
            adapt_interval=100,
            seed=seed + chain_id * 100
        )
        fn = lambda th: log_posterior(th, f, Z_obs, snr_db)
        chain_result = sampler.run(fn, perturbacao, verbose=True)
        cadeias.append(chain_result)

    # Combinar cadeias
    samples = np.vstack([c["samples"] for c in cadeias])
    warmup  = np.vstack([c["warmup"]  for c in cadeias])

    # Diagnósticos
    rhat = compute_rhat([c["samples"] for c in cadeias])
    ess_per_param = [compute_ess(samples[:, i]) for i in range(4)]

    print(f"\n  Taxa aceitação (cadeia 1): {cadeias[0]['accept_rate']*100:.1f}%")
    print(f"  Taxa aceitação (cadeia 2): {cadeias[1]['accept_rate']*100:.1f}%")
    print(f"  R-hat: {rhat}")
    print(f"  ESS:   {[f'{e:.0f}' for e in ess_per_param]}")

    for i, (rh, name) in enumerate(zip(rhat, ["R_inf", "ΔR", "ln_tau", "alpha"])):
        status = "✓" if rh < 1.05 else "⚠ CHECAR"
        print(f"    {name:>8}: R-hat = {rh:.4f} {status}")

    return samples, warmup, cadeias


# ===========================================================================
# Etapa 4 — Resumo posterior e derivadas
# ===========================================================================

def etapa4_resumo(samples, true_params):
    print("\n" + "="*60)
    print(" ETAPA 4 — Resumo Posterior e Grandezas Derivadas")
    print("="*60)

    phys = samples_to_physical(samples)

    # Resumo das 4 grandezas diretamente amostradas (espaço MCMC)
    param_names_mcmc = ["R_inf", "delta_R", "log_tau", "alpha"]
    true_mcmc = {
        "R_inf":    true_params["R_inf"],
        "delta_R":  true_params["R0"] - true_params["R_inf"],
        "log_tau":  np.log(true_params["tau"]),
        "alpha":    true_params["alpha"],
    }
    summ_mcmc = posterior_summary(samples, param_names_mcmc, true_mcmc)
    print_summary_table(summ_mcmc, "Parâmetros MCMC (espaço de amostragem)")

    # Resumo das grandezas físicas derivadas
    phys_names = ["R_inf", "R0", "tau", "alpha", "f_c", "delta_R"]
    phys_samples_arr = np.column_stack([phys[k] for k in phys_names])
    true_phys = {
        "R_inf":   true_params["R_inf"],
        "R0":      true_params["R0"],
        "tau":     true_params["tau"],
        "alpha":   true_params["alpha"],
        "f_c":     characteristic_frequency(true_params["tau"], true_params["alpha"]),
        "delta_R": true_params["R0"] - true_params["R_inf"],
    }
    summ_phys = posterior_summary(phys_samples_arr, phys_names, true_phys)
    print_summary_table(summ_phys, "Grandezas Físicas (pós-transformação)")

    return summ_phys, phys


# ===========================================================================
# Etapa 5 — Estudo de sensibilidade: HDI × SNR
# ===========================================================================

def etapa5_sensibilidade(true_params, n_freq, n_amostras, n_warmup, seed, pasta):
    print("\n" + "="*60)
    print(" ETAPA 5 — Sensibilidade: Largura do HDI × SNR")
    print("="*60)

    snr_levels = [15, 20, 25, 30, 35, 40]
    n_rep = 5   # realizações por SNR (aumentar para artigo final)
    params_interesse = ["R_inf", "R0", "tau", "alpha", "f_c"]
    hdi_widths = {p: [] for p in params_interesse}

    f = frequency_grid(100.0, 1e6, n_freq)

    for snr in snr_levels:
        print(f"\n  SNR = {snr} dB ({n_rep} realizações)...")
        widths_rep = {p: [] for p in params_interesse}

        for rep in range(n_rep):
            seed_r = seed + snr * 100 + rep
            data = generate_eis_data(f, seed=seed_r, snr_db=snr, **true_params)
            Z_obs = data["Z_noisy"]

            # NLLS como inicialização
            ls_r = nlls_fit(f, Z_obs, method="complex", n_restarts=3)
            theta0 = _build_theta0(ls_r, true_params)

            sampler = AdaptiveMCMC(
                n_samples=n_amostras, n_warmup=n_warmup,
                adapt_start=300, adapt_interval=100,
                seed=seed_r + 7
            )
            fn = lambda th: log_posterior(th, f, Z_obs, snr)
            try:
                res = sampler.run(fn, theta0, verbose=False)
                phys = samples_to_physical(res["samples"])
                for p in params_interesse:
                    l, h = compute_hdi(phys[p])
                    widths_rep[p].append(h - l)
            except Exception as e:
                print(f"    [rep {rep}] Erro: {e}")

        for p in params_interesse:
            ws = widths_rep[p]
            hdi_widths[p].append(np.mean(ws) if ws else np.nan)
        print(f"    HDI médio |R_inf|: {hdi_widths['R_inf'][-1]:.2f} Ω  |  "
              f"alpha: {hdi_widths['alpha'][-1]:.4f}")

    # Salvar figura
    labels_snr = {
        "R_inf": r"$R_\infty$ [Ω]",
        "R0":    r"$R_0$ [Ω]",
        "tau":   r"$\tau$ [s]",
        "alpha": r"$\alpha$",
        "f_c":   r"$f_c$ [Hz]",
    }
    path_snr = os.path.join(pasta, "fig5_sensibilidade_snr.png")
    plot_snr_sensitivity(snr_levels, hdi_widths, labels_snr,
                         save_path=path_snr)
    return snr_levels, hdi_widths


def _build_theta0(ls_result, true_params):
    """Constrói θ0 com fallback para valores fisiológicos."""
    if ls_result and ls_result.get("converged") and ls_result.get("params"):
        p = ls_result["params"]
        return np.array([
            max(p["R_inf"], 1.0),
            max(p["R0"] - p["R_inf"], 1.0),
            np.log(max(p["tau"], 1e-10)),
            np.clip(p["alpha"], 0.05, 0.98)
        ])
    return np.array([
        true_params["R_inf"],
        true_params["R0"] - true_params["R_inf"],
        np.log(true_params["tau"]),
        true_params["alpha"]
    ])


# ===========================================================================
# Etapa 6 — Gerar todas as figuras do artigo
# ===========================================================================

def etapa6_figuras(f, data, samples, warmup, ls_result, true_params, pasta):
    print("\n" + "="*60)
    print(" ETAPA 6 — Gerando Figuras do Artigo")
    print("="*60)
    os.makedirs(pasta, exist_ok=True)

    Z_obs = data["Z_noisy"]

    print("  [1/5] Espectro EIS com banda posterior...")
    plot_eis_spectrum(
        f, Z_obs, samples,
        true_params=true_params,
        ls_result=ls_result,
        title=f"Espectro EIS — SNR={data['snr_db']:.0f} dB",
        save_path=os.path.join(pasta, "fig1_espectro_eis.png")
    )

    print("  [2/5] Distribuições posteriores...")
    plot_posterior_distributions(
        samples,
        true_params=true_params,
        ls_result=ls_result,
        title="Distribuições Posteriores dos Parâmetros Cole-Cole",
        save_path=os.path.join(pasta, "fig2_posteriors.png")
    )

    print("  [3/5] Trace plots e diagnóstico MCMC...")
    plot_trace(
        samples, warmup=warmup,
        param_names=[r"$R_\infty$ [Ω]", r"$\Delta R$ [Ω]",
                     r"$\ln(\tau)$", r"$\alpha$"],
        save_path=os.path.join(pasta, "fig3_trace.png")
    )

    print("  [4/5] Comparação Bayes vs NLLS...")
    plot_bayes_vs_nlls(
        samples, ls_result, true_params,
        save_path=os.path.join(pasta, "fig4_bayes_vs_nlls.png")
    )

    print("  [5/5] Corner plot (correlações posteriores)...")
    plot_corner(
        samples,
        param_names=[r"$R_\infty$", r"$\Delta R$", r"$\ln\tau$", r"$\alpha$"],
        true_params=true_params,
        save_path=os.path.join(pasta, "fig6_corner.png")
    )

    print(f"\n  Todas as figuras salvas em: {os.path.abspath(pasta)}/")


# ===========================================================================
# Pipeline principal
# ===========================================================================

def main():
    args = parse_args()

    # Modo rápido (para testes)
    if args.rapido:
        n_amostras = 1000
        n_warmup   = 500
        print("[MODO RÁPIDO] n_amostras=1000, n_warmup=500")
    else:
        n_amostras = args.n_amostras
        n_warmup   = args.n_warmup

    print("\n" + "█"*60)
    print("  ESTIMAÇÃO BAYESIANA DE BIOIMPEDÂNCIA (Cole-Cole)")
    print("  EEL410279 — Fundamentos de Bioimpedância — UFSC")
    print("█"*60)

    t_inicio = time.time()

    # Etapa 1 — Dados
    f, data, true_params = etapa1_dados(
        args.tecido, args.snr, args.n_freq, args.seed)
    Z_obs = data["Z_noisy"]

    # Etapa 2 — NLLS
    ls_result = etapa2_nlls(f, Z_obs)

    # Etapa 3 — MCMC
    samples, warmup, cadeias = etapa3_mcmc(
        f, Z_obs, args.snr, n_amostras, n_warmup,
        args.seed, ls_result)

    # Etapa 4 — Resumo
    summ, phys = etapa4_resumo(samples, true_params)

    # Etapa 5 — Sensibilidade (opcional, é lenta)
    if not args.sem_snr:
        etapa5_sensibilidade(
            true_params, args.n_freq,
            n_amostras=min(n_amostras, 2000),
            n_warmup=min(n_warmup, 1000),
            seed=args.seed,
            pasta=args.saida
        )

    # Etapa 6 — Figuras
    etapa6_figuras(f, data, samples, warmup, ls_result, true_params, args.saida)

    t_total = time.time() - t_inicio
    print(f"\n{'='*60}")
    print(f"  Pipeline concluído em {t_total/60:.1f} minutos.")
    print(f"  Resultados em: {os.path.abspath(args.saida)}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
