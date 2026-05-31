"""
main_real_data.py
=================
Pipeline de validação com dados reais públicos.

Executa o framework Bayesiano completo em dados experimentais reais de
tecidos biológicos, sem dados sintéticos.

Uso:
    python main_real_data.py                         # todos os datasets
    python main_real_data.py --dataset gabriel_muscle_longitudinal
    python main_real_data.py --csv minha_medicao.csv
    python main_real_data.py --lista                 # mostra datasets
    python main_real_data.py --rapido                # apenas Gabriel muscle
    python main_real_data.py --exemplo-csv           # gera CSV de template
"""

import argparse
import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.real_data     import DatasetRegistry
from src.ls_fitting    import nlls_fit
from src.mcmc_sampler  import AdaptiveMCMC, log_posterior, compute_ess
from src.analysis      import (samples_to_physical, posterior_summary,
                                print_summary_table, plot_eis_spectrum,
                                plot_posterior_distributions, plot_trace,
                                plot_corner)
from src.models        import ALL_MODELS, pointwise_log_likelihood
from src.model_comparison import (compute_waic, compute_loo_psis,
                                   compare_models, print_comparison_table)

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                      "axes.spines.right": False, "font.size": 11})


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Validação com dados reais — Bioimpedância Bayesiana")
    p.add_argument("--dataset",    type=str, default=None,
                   help="Chave do dataset (use --lista para ver opções)")
    p.add_argument("--csv",        type=str, default=None,
                   help="Caminho para CSV do usuário (freq_hz, Re_Z, Im_Z)")
    p.add_argument("--snr",        type=float, default=30.0,
                   help="SNR estimado da medição em dB (padrão: 30)")
    p.add_argument("--n-amostras", type=int, default=4000)
    p.add_argument("--n-warmup",   type=int, default=2000)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--lista",      action="store_true",
                   help="Lista todos os datasets disponíveis e sai")
    p.add_argument("--rapido",     action="store_true",
                   help="Roda apenas Gabriel muscle (teste rápido)")
    p.add_argument("--exemplo-csv", dest="exemplo_csv", action="store_true",
                   help="Gera CSV de exemplo e sai")
    p.add_argument("--saida",      type=str, default="resultados_real")
    p.add_argument("--comparar-modelos", dest="comparar_modelos",
                   action="store_true",
                   help="Adiciona comparação WAIC/LOO de 4 modelos ECM")
    return p.parse_args()


# ===========================================================================
# Pipeline principal para um dataset
# ===========================================================================

def run_single_dataset(key: str, data: dict, snr_db: float,
                        n_amostras: int, n_warmup: int, seed: int,
                        pasta: str, comparar_modelos: bool = False):
    """
    Executa o pipeline Bayesiano completo para um espectro EIS.

    1. NLLS para inicialização
    2. AM-MCMC → posterior completo
    3. Resumo HDI, ESS, R-hat
    4. Comparação de modelos (opcional)
    5. Geração de figuras
    """
    os.makedirs(pasta, exist_ok=True)

    f    = data["f"]
    Z    = data["Z_noisy"]
    meta = data.get("meta", {})

    print(f"\n{'='*60}")
    print(f"  Dataset : {key}")
    print(f"  Fonte   : {meta.get('source','?')[:60]}")
    print(f"  Pontos  : {len(f)}  |  f: [{f.min():.0f}, {f.max():.0f}] Hz")
    print(f"  |Z| range: [{np.abs(Z).min():.1f}, {np.abs(Z).max():.1f}] Ω")
    if "temp_C" in meta:
        print(f"  Temp    : {meta['temp_C']} °C")
    print(f"{'='*60}")

    # ── 1. NLLS (referência) ──────────────────────────────────────
    print("\n[1/4] Ajuste NLLS...")
    ls_result = nlls_fit(f, Z, method="complex", n_restarts=5)
    if ls_result.get("converged"):
        p = ls_result["params"]
        print(f"  R_inf={p['R_inf']:.2f} Ω  R0={p['R0']:.2f} Ω  "
              f"tau={p['tau']*1e6:.2f} µs  alpha={p['alpha']:.4f}")
        print(f"  f_c={ls_result['derived']['f_c']:.1f} Hz  "
              f"chi2={ls_result['chi2']:.4g}")
        R_inf_fit = max(p["R_inf"], 1.)
        dR_fit    = max(p["R0"] - p["R_inf"], 1.)
        tau_fit   = max(p["tau"], 1e-10)
        alpha_fit = np.clip(p["alpha"], 0.05, 0.98)
        theta0 = np.array([R_inf_fit, dR_fit,
                           np.log(tau_fit), alpha_fit])
    else:
        # Fallback: estimar da parte real em frequências extremas
        R_inf_est = float(np.abs(Z[-3:]).mean())
        R0_est    = float(np.abs(Z[:3]).mean())
        R_inf_est = max(R_inf_est, 1.)
        dR_est    = max(R0_est - R_inf_est, 1.)
        print(f"  NLLS não convergiu — usando R_inf≈{R_inf_est:.1f}, ΔR≈{dR_est:.1f}")
        theta0 = np.array([R_inf_est, dR_est, np.log(7.96e-6), 0.75])

    # ── 2. AM-MCMC ────────────────────────────────────────────────
    print(f"\n[2/4] AM-MCMC ({n_amostras} amostras, {n_warmup} warm-up)...")
    t0 = time.time()
    cadeias = []
    from src.mcmc_sampler import _empirical_bayes_log_posterior
    fn_post = lambda th: _empirical_bayes_log_posterior(th, f, Z, snr_db, theta0)
    for chain_id in range(2):
        perturb = theta0 * np.array([1. + 0.08*chain_id,
                                     1. - 0.04*chain_id,
                                     1.,
                                     1. - 0.02*chain_id])
        sampler = AdaptiveMCMC(
            n_samples=n_amostras, n_warmup=n_warmup,
            adapt_start=min(400, n_warmup//4), adapt_interval=100,
            seed=seed + chain_id * 100)
        cadeias.append(sampler.run(fn_post, perturb, verbose=(chain_id == 0)))
    print(f"  Tempo: {time.time()-t0:.1f}s")

    samples = np.vstack([c["samples"] for c in cadeias])
    warmup  = np.vstack([c["warmup"]  for c in cadeias])

    # Diagnósticos
    ess_vals = [compute_ess(samples[:, i]) for i in range(4)]
    print(f"\n  Taxa aceit.: {np.mean([c['accept_rate'] for c in cadeias])*100:.1f}%")
    print(f"  ESS        : {[f'{e:.0f}' for e in ess_vals]}")

    from src.mcmc_sampler import compute_rhat
    rhat = compute_rhat([c["samples"] for c in cadeias])
    for i, (rh, name) in enumerate(zip(rhat, ["R_inf","ΔR","ln_tau","alpha"])):
        status = "✓" if rh < 1.05 else "⚠ CHECAR"
        print(f"    {name:>8}: R-hat={rh:.4f} {status}")

    # ── 3. Resumo posterior ────────────────────────────────────────
    print(f"\n[3/4] Resumo posterior...")
    true_params = None
    if all(k in data for k in ("R_inf", "R0", "tau", "alpha")):
        true_params = {
            "R_inf": data["R_inf"], "R0": data["R0"],
            "tau": data["tau"],    "alpha": data["alpha"]
        }
    summ = posterior_summary(
        samples,
        param_names=["R_inf", "delta_R", "log_tau", "alpha"],
        true_params={
            "R_inf":   true_params["R_inf"] if true_params else None,
            "delta_R": (true_params["R0"] - true_params["R_inf"]) if true_params else None,
            "log_tau": np.log(true_params["tau"]) if true_params else None,
            "alpha":   true_params["alpha"] if true_params else None,
        } if true_params else None
    )
    print_summary_table(summ, title=f"Posterior — {key}")

    # ── 4. Figuras ─────────────────────────────────────────────────
    print(f"\n[4/4] Gerando figuras → {pasta}/")

    # Parâmetros verdadeiros para plots (se disponíveis)
    tp_plot = true_params  # pode ser None para CSVs do usuário

    plot_eis_spectrum(
        f, Z, samples, true_params=tp_plot, ls_result=ls_result,
        title=f"EIS — {key} | {meta.get('source','')[:40]}",
        save_path=os.path.join(pasta, f"{key}_espectro.png"))

    plot_posterior_distributions(
        samples, true_params=tp_plot, ls_result=ls_result,
        title=f"Posteriors — {key}",
        save_path=os.path.join(pasta, f"{key}_posteriors.png"))

    plot_trace(
        samples, warmup=warmup,
        save_path=os.path.join(pasta, f"{key}_trace.png"))

    plot_corner(
        samples,
        true_params=tp_plot,
        save_path=os.path.join(pasta, f"{key}_corner.png"))

    # ── 5. Comparação de modelos (opcional) ───────────────────────
    if comparar_modelos:
        print(f"\n[Extra] Comparação WAIC/LOO de 4 modelos ECM...")
        waic_results = {}
        from main_ext24 import run_mcmc_for_model
        for mk, model in ALL_MODELS.items():
            print(f"  Modelo {mk}: {model.name}...")
            r = run_mcmc_for_model(model, f, Z, snr_db,
                                    n_amostras, n_warmup, seed)
            ll_mat = pointwise_log_likelihood(model, r["samples"], f, Z, snr_db)
            waic_r = compute_waic(ll_mat)
            loo_r  = compute_loo_psis(ll_mat)
            waic_results[mk] = {"waic": waic_r, "loo": loo_r,
                                  "n_params": model.n_params}
        comp = compare_models(waic_results)
        print_comparison_table(comp["table"], comp["ranking"])

    return {"samples": samples, "ls_result": ls_result, "summary": summ}


# ===========================================================================
# Comparação multi-dataset (R_inf × tipo de tecido)
# ===========================================================================

def plot_multi_dataset_comparison(all_results: dict, pasta: str):
    """
    Plota comparação de R_inf posterior entre múltiplos datasets reais.
    Útil para validação cross-dataset.
    """
    from src.analysis import samples_to_physical, compute_hdi

    fig, ax = plt.subplots(figsize=(max(8, len(all_results)*1.5), 5))
    keys   = sorted(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(keys)))

    for i, (key, res) in enumerate(sorted(all_results.items())):
        phys = samples_to_physical(res["samples"])
        r_inf_samples = phys["R_inf"]
        med = np.median(r_inf_samples)
        lo, hi = compute_hdi(r_inf_samples, 0.95)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]],
                    fmt="o", color=colors[i], ms=8, capsize=5,
                    label=key.replace("gabriel_","G:").replace("dipa_","D:").replace("fruit_","F:"))
        ax.text(i, hi + 2, f"{med:.0f}Ω", ha="center", fontsize=8)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.split("_")[-1] for k in sorted(all_results.keys())],
                        rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"$R_\infty$ [Ω] — Mediana posterior + HDI 95%")
    ax.set_title("Comparação Cross-Dataset: $R_\\infty$ (Posterior Bayesiana)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(pasta, "comparacao_datasets.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Figura salva: {path}")
    plt.show()
    return fig


# ===========================================================================
# Main
# ===========================================================================

def main():
    args  = parse_args()
    reg   = DatasetRegistry()

    if args.lista:
        reg.list_datasets()
        return

    if args.exemplo_csv:
        reg.save_example_csv("exemplo_medicao.csv", snr_db=25.0)
        print("\nUse: python main_real_data.py --csv exemplo_medicao.csv")
        return

    if args.rapido:
        datasets_to_run = ["gabriel_muscle_longitudinal",
                           "dipa_chicken_muscle_37C"]
    elif args.csv:
        datasets_to_run = ["__csv__"]
    elif args.dataset:
        datasets_to_run = [args.dataset]
    else:
        # Rodar todos os datasets principais
        datasets_to_run = [
            "gabriel_muscle_longitudinal",
            "gabriel_muscle_transverse",
            "gabriel_liver",
            "gabriel_fat",
            "gabriel_blood",
            "dipa_chicken_muscle_37C",
            "dipa_chicken_muscle_25C",
            "dipa_chicken_muscle_15C",
            "fruit_potato_day1",
            "fruit_potato_day9",
        ]

    if args.rapido:
        n_amostras = 2000
        n_warmup   = 1000
    else:
        n_amostras = args.n_amostras
        n_warmup   = args.n_warmup

    os.makedirs(args.saida, exist_ok=True)
    all_results = {}
    t_total = time.time()

    print("\n" + "█"*60)
    print("  VALIDAÇÃO COM DADOS REAIS — Bioimpedância Bayesiana")
    print("  Datasets: Gabriel (1996), Dipa (2024), Frutas/Vegetais")
    print("█"*60)

    for key in datasets_to_run:
        try:
            if key == "__csv__":
                data = reg.load_csv(args.csv)
                key_label = Path(args.csv).stem
            else:
                data = reg.load(key, snr_db=args.snr, seed=args.seed)
                key_label = key

            pasta_k = os.path.join(args.saida, key_label)
            result  = run_single_dataset(
                key_label, data, args.snr,
                n_amostras, n_warmup, args.seed,
                pasta_k, comparar_modelos=args.comparar_modelos)
            all_results[key_label] = result

        except Exception as e:
            print(f"\n  [ERRO] {key}: {e}")
            import traceback; traceback.print_exc()
            continue

    # Comparação cross-dataset (apenas para múltiplos datasets)
    if len(all_results) > 1:
        print(f"\n  Gerando comparação cross-dataset...")
        try:
            plot_multi_dataset_comparison(all_results, args.saida)
        except Exception as e:
            print(f"  [AVISO] Comparação falhou: {e}")

    print(f"\n{'='*60}")
    print(f"  Concluído em {(time.time()-t_total)/60:.1f} min.")
    print(f"  {len(all_results)} datasets processados.")
    print(f"  Resultados: {os.path.abspath(args.saida)}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
