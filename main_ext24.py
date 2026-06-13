"""
main_ext24.py  v2 — Extensões 2 e 4 com IS-Posterior Classifier
================================================================
EXT.2: Comparação WAIC/LOO de 4 modelos de circuito equivalente
EXT.4: Classificação de estados teciduais via IS-Posterior (2 estágios)

Uso:
    python main_ext24.py                      # completo (lento ~30-60 min)
    python main_ext24.py --rapido             # teste rápido (~5 min)
    python main_ext24.py --so-modelos         # só Ext.2
    python main_ext24.py --so-classificacao   # só Ext.4
    python main_ext24.py --snr-scan           # + varredura SNR
"""

import argparse, os, time, warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.cole_model       import TISSUE_PARAMS, characteristic_frequency
from src.data_generation  import frequency_grid, generate_eis_data
from src.ls_fitting       import nlls_fit
from src.mcmc_sampler     import AdaptiveMCMC, log_posterior, compute_ess
from src.models           import ALL_MODELS, pointwise_log_likelihood
from src.model_comparison import (compute_waic, compute_loo_psis,
                                   compare_models, print_comparison_table)
from src.tissue_states    import ALL_STATES
from src.classifier_v2    import (ISPosteriorClassifier,
                                   simulation_study_v2, snr_sensitivity_v2)
from src.analysis_ext24   import (plot_model_spectra_comparison,
                                   plot_waic_comparison, plot_loo_diagnostics,
                                   plot_tissue_state_spectra,
                                   plot_parameter_distributions_by_state,
                                   plot_classification_result,
                                   plot_confusion_matrix, plot_roc_curves,
                                   plot_accuracy_vs_snr, plot_calibration)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snr",        type=float, default=30.0)
    p.add_argument("--tecido",     type=str,   default="musculo",
                   choices=list(TISSUE_PARAMS.keys()))
    p.add_argument("--n-freq",     type=int,   default=30)
    p.add_argument("--n-amostras", type=int,   default=4000)
    p.add_argument("--n-warmup",   type=int,   default=2000)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--rapido",     action="store_true")
    p.add_argument("--so-modelos", dest="so_modelos",  action="store_true")
    p.add_argument("--so-classificacao", dest="so_clf",action="store_true")
    p.add_argument("--snr-scan",   dest="snr_scan",    action="store_true")
    p.add_argument("--saida",      type=str,   default="resultados_v3")
    return p.parse_args()


# ===========================================================================
# MCMC genérico para qualquer modelo
# ===========================================================================

def _loglike_model(model, theta, f, Z_obs, snr_db):
    from src.data_generation import snr_to_sigma
    try:
        Z_pred = model.impedance_fn(f, theta)
        if not np.all(np.isfinite(Z_pred)):
            return -np.inf
        sn = snr_to_sigma(snr_db)
        sf = np.maximum(sn * np.abs(Z_pred), 1e-12)
        rR = Z_obs.real - Z_pred.real
        rX = Z_obs.imag - Z_pred.imag
        ll = (-np.sum(np.log(sf)) - 0.5*np.sum((rR/sf)**2)
              -np.sum(np.log(sf)) - 0.5*np.sum((rX/sf)**2))
        return float(ll) if np.isfinite(ll) else -np.inf
    except Exception:
        return -np.inf


def run_mcmc_for_model(model, f, Z_obs, snr_db, n_amostras, n_warmup, seed):
    theta0 = model.theta0_fn(f, Z_obs, snr_db)
    theta0 = np.clip(theta0, [b+1e-6 for b in model.bounds_lower],
                              [b-1e-6 for b in model.bounds_upper])
    fn = lambda th: model.log_prior_fn(th) + _loglike_model(model, th, f, Z_obs, snr_db)
    lp0 = fn(theta0)
    if not np.isfinite(lp0):
        theta0 = np.array([(lo+hi)/2 for lo,hi in
                           zip(model.bounds_lower, model.bounds_upper)])
    sampler = AdaptiveMCMC(n_samples=n_amostras, n_warmup=n_warmup,
                            adapt_start=min(400,n_warmup//4),
                            adapt_interval=100, seed=seed)
    return sampler.run(fn, theta0, verbose=True)


# ===========================================================================
# EXTENSÃO 2
# ===========================================================================

def run_extension2(f, Z_obs, snr_db, true_params, n_amostras, n_warmup, seed, pasta):
    print("\n"+"█"*60)
    print("  EXTENSÃO 2 — COMPARAÇÃO FORMAL DE MODELOS (WAIC/LOO)")
    print("█"*60)
    os.makedirs(pasta, exist_ok=True)

    model_samples = {}
    model_results = {}

    for key, model in ALL_MODELS.items():
        print(f"\n  ── {key}: {model.name} ({model.n_params} params) ──")
        t0 = time.time()
        r  = run_mcmc_for_model(model, f, Z_obs, snr_db,
                                  n_amostras, n_warmup, seed)
        samples = r["samples"]
        model_samples[key] = samples
        print(f"     {time.time()-t0:.1f}s | aceitação {r['accept_rate']*100:.1f}%")
        print(f"     ESS: {[f'{compute_ess(samples[:,i]):.0f}' for i in range(model.n_params)]}")

        ll_mat = pointwise_log_likelihood(model, samples, f, Z_obs, snr_db)
        waic_r = compute_waic(ll_mat)
        loo_r  = compute_loo_psis(ll_mat)
        print(f"     WAIC={waic_r['waic']:.2f}  p_WAIC={waic_r['p_waic']:.2f}  "
              f"LOO={loo_r['loo']:.2f}  k>0.7:{loo_r['n_bad_k']}")

        model_results[key] = {"waic": waic_r, "loo": loo_r,
                               "n_params": model.n_params, "samples": samples}

    comp = compare_models(model_results)
    print_comparison_table(comp["table"], comp["ranking"])

    plot_model_spectra_comparison(f, Z_obs, model_samples, dict(ALL_MODELS),
        true_params=true_params,
        save_path=os.path.join(pasta, "fig_E2a_espectros.png"))
    plot_waic_comparison(comp["table"], comp["ranking"],
        save_path=os.path.join(pasta, "fig_E2b_waic.png"))
    plot_loo_diagnostics(model_results, comp["ranking"], f,
        save_path=os.path.join(pasta, "fig_E2c_loo.png"))

    print(f"\n  Melhor modelo: {comp['best']} ({ALL_MODELS[comp['best']].name})")
    return comp, model_results, model_samples


# ===========================================================================
# EXTENSÃO 4 — IS-Posterior
# ===========================================================================

def run_extension4(f, snr_db, n_mcmc, n_warmup, n_per_state, seed, pasta, snr_scan):
    """Pipeline Ext.4 com correções de seed e ECE."""
    import os, numpy as np
    from src.tissue_states import ALL_STATES
    from src.data_generation import generate_eis_data
    from src.ls_fitting import nlls_fit
    from src.mcmc_sampler import AdaptiveMCMC, log_posterior
    from src.classifier_v2 import ISPosteriorClassifier, simulation_study_v2, snr_sensitivity_v2
    from src.analysis_ext24 import (plot_tissue_state_spectra,
                                     plot_parameter_distributions_by_state,
                                     plot_classification_result,
                                     plot_confusion_matrix, plot_roc_curves,
                                     plot_calibration, plot_accuracy_vs_snr)
    os.makedirs(pasta, exist_ok=True)

    # Figuras descritivas
    print("\n  [1/6] Espectros de referência dos estados...")
    plot_tissue_state_spectra(f, n_samples=4, snr_db=35, seed=42,
        save_path=os.path.join(pasta, "fig_E4a_espectros.png"))

    print("  [2/6] Distribuições a priori dos parâmetros...")
    plot_parameter_distributions_by_state(
        save_path=os.path.join(pasta, "fig_E4b_priors.png"))

    # ── CORREÇÃO [5]: seeds fixas por estado para exemplos representativos ──
    print("\n  [3/6] Classificando exemplos individuais (IS-Posterior)...")

    # Seeds escolhidas para garantir amostras CENTRAIS (não outliers)
    EXAMPLE_SEEDS = {
        "Normal":   300,   # R_inf ≈ 50Ω, alpha ≈ 0.76 — exemplo central
        "Edema":    1,     # R_inf ≈ 38Ω, tau maior — exemplo típico
        "Isquemia": 3,     # R_inf ≈ 65Ω, tau menor — exemplo típico
    }

    clf = ISPosteriorClassifier(n_bootstrap=100)

    for true_name, state in ALL_STATES.items():
        ex_seed = EXAMPLE_SEEDS[true_name]
        print(f"\n    Exemplo: tecido {true_name} (seed={ex_seed})")

        # Usar parâmetros nominais + pequena perturbação controlada
        nom  = state.nominal_params()
        rng  = np.random.default_rng(ex_seed)
        R_inf_ex = nom["R_inf"] * rng.uniform(0.90, 1.10)
        dR_ex    = (nom["R0"] - nom["R_inf"]) * rng.uniform(0.90, 1.10)
        tau_ex   = nom["tau"]   * rng.uniform(0.90, 1.10)
        alpha_ex = np.clip(nom["alpha"] + rng.uniform(-0.03, 0.03), 0.30, 0.98)

        data_ex = generate_eis_data(f, R_inf_ex, R_inf_ex + dR_ex,
                                     tau_ex, alpha_ex,
                                     snr_db=snr_db,
                                     seed=ex_seed * 7 + 3)
        Z_ex = data_ex["Z_noisy"]

        # NLLS → theta0
        ls = nlls_fit(f, Z_ex, n_restarts=3)
        if ls.get("converged"):
            p   = ls["params"]
            th0 = np.array([max(p["R_inf"], 1.),
                            max(p["R0"] - p["R_inf"], 1.),
                            np.log(max(p["tau"], 1e-10)),
                            np.clip(p["alpha"], 0.05, 0.98)])
        else:
            th0 = np.array([R_inf_ex, dR_ex, np.log(tau_ex), alpha_ex])

        # MCMC
        sampler = AdaptiveMCMC(n_samples=n_mcmc, n_warmup=n_warmup,
                                adapt_start=300, adapt_interval=100,
                                seed=ex_seed * 13)
        fn      = lambda th: log_posterior(th, f, Z_ex, snr_db)
        mcmc_r  = sampler.run(fn, th0, verbose=False)

        # IS-Posterior
        r = clf.classify(mcmc_r["samples"])
        ok = "✓" if r["predicted"] == true_name else "✗"
        print(f"    Predito: {r['predicted']:10s} conf={r['confidence']:.4f} {ok}")
        for k, pv in r["probs"].items():
            print(f"      p({k:10s}|Z) = {pv:.4f}  ESS_IS={r['ess_is'][k]:.0f}")

        plot_classification_result(r, true_state=true_name,
            save_path=os.path.join(pasta, f"fig_E4c_{true_name.lower()}.png"))

    # Estudo de simulação completo
    print(f"\n  [4/6] Simulação completa (IS-Posterior, SNR={snr_db} dB)...")
    sim = simulation_study_v2(f, snr_db=snr_db,
                               n_per_state=n_per_state,
                               n_mcmc=n_mcmc, n_warmup=n_warmup,
                               seed=seed, verbose=True)

    print(f"\n  Acurácia: {sim['accuracy']*100:.1f}%")
    for k in sim["state_names"]:
        print(f"    {k:10s}: {sim['class_accuracy'][k]*100:.1f}%  "
              f"AUC={sim['auc_roc'][k]:.3f}")
    print(f"  ECE: {sim['calibration']['ece']:.4f}")

    plot_confusion_matrix(sim,
        save_path=os.path.join(pasta, "fig_E4d_confusion.png"))
    plot_roc_curves(sim,
        save_path=os.path.join(pasta, "fig_E4e_roc.png"))
    plot_calibration(sim,
        save_path=os.path.join(pasta, "fig_E4f_calibration.png"))

    # SNR scan com ECE — CORREÇÃO [4b]
    if snr_scan:
        print("\n  [5/6] Varredura SNR (IS-Posterior + ECE)...")
        snr_results = snr_sensitivity_v2(
            f, snr_levels=[15, 20, 25, 30, 35, 40],
            n_per_state=max(n_per_state // 2, 8),
            n_mcmc=n_mcmc, n_warmup=n_warmup, seed=seed)

        # Adicionar n_test para cálculo de SE
        for snr_val in snr_results:
            snr_results[snr_val]["n_test"] = max(n_per_state // 2, 8) * 3

        plot_accuracy_vs_snr(snr_results,
            save_path=os.path.join(pasta, "fig_E4g_snr_scan.png"))

    print("\n  [6/6] Figuras concluídas.")
    return sim


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    if args.rapido:
        n_amostras = 1500; n_warmup = 750; n_per_state = 8
        print("[RÁPIDO] n_amostras=1500, n_per_state=8")
    else:
        n_amostras = args.n_amostras; n_warmup = args.n_warmup
        n_per_state = 30

    os.makedirs(args.saida, exist_ok=True)
    t0 = time.time()

    print("\n"+"█"*60)
    print("  BIOIMPEDÂNCIA BAYESIANA — EXT.2 + EXT.4 (IS-Posterior v2)")
    print("  EEL410279 — PPGEEL-UFSC")
    print("█"*60)

    params_ref  = TISSUE_PARAMS[args.tecido]
    true_params = {k: v for k, v in params_ref.items()
                   if k in ("R_inf","R0","tau","alpha")}
    f    = frequency_grid(100., 1e6, args.n_freq)
    data = generate_eis_data(f, seed=args.seed, snr_db=args.snr, **true_params)
    Z_obs = data["Z_noisy"]
    print(f"\n  Tecido={args.tecido}  SNR={args.snr} dB  n_freq={args.n_freq}")

    if not args.so_clf:
        run_extension2(f, Z_obs, args.snr, true_params,
                       n_amostras, n_warmup, args.seed,
                       pasta=os.path.join(args.saida, "ext2_modelos"))

    if not args.so_modelos:
        run_extension4(f, args.snr,
                       n_mcmc=n_amostras, n_warmup=n_warmup,
                       n_per_state=n_per_state, seed=args.seed,
                       pasta=os.path.join(args.saida, "ext4_classificacao"),
                       snr_scan=args.snr_scan)

    print(f"\n{'='*60}")
    print(f"  Concluído em {(time.time()-t0)/60:.1f} min.")
    print(f"  Resultados: {os.path.abspath(args.saida)}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
