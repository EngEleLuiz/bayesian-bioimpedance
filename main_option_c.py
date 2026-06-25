"""
main_option_c.py
================
Pipeline da Opção C: validação honesta com modelo dielétrico completo
de Gabriel (4-Cole-Cole) + ajuste de banda limitada (sub-banda β).

ANTES DE RODAR — PASSO OBRIGATÓRIO
====================================
Os parâmetros completos de Δε_n, τ_n, α_n para cada tecido em
`src/gabriel_4cc_model.py` (MUSCLE_4CC_PARTIAL) estão marcados como
NÃO VERIFICADOS. Antes de usar este pipeline para o artigo final:

  1. Acesse: https://www.fcc.gov/general/body-tissue-dielectric-parameters
  2. Selecione o tecido "Muscle"
  3. Para cada frequência da sua grade (ex.: 100 Hz, 215 Hz, 464 Hz, ...,
     1 MHz — 30 pontos log-espaçados), anote ε' (Permittivity) e
     σ (Conductivity, S/m)
  4. Edite o arquivo `dados_fcc_consultados.csv` (gerado por
     `--gerar-template`) com os valores reais consultados
  5. Rode o pipeline com `--csv-fcc dados_fcc_consultados.csv`

Isso elimina qualquer risco de erro de transcrição de parâmetros e
torna a validação 100% rastreável à fonte oficial.

Uso:
    # Gerar template CSV para você preencher com a calculadora oficial
    python main_option_c.py --gerar-template

    # Rodar com dados reais consultados na FCC
    python main_option_c.py --csv-fcc dados_fcc_consultados.csv

    # Modo demonstração (usa parâmetros NÃO verificados — só para teste de código)
    python main_option_c.py --demo --rapido
"""

import argparse
import os
import csv
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.gabriel_4cc_model import (
    FourColeColeParams, MUSCLE_4CC_PARTIAL,
    complex_permittivity_4cc, conductivity_and_permittivity,
    impedance_from_4cc, impedance_from_4cc_calibrated,
    calibrar_geometria_medicao,
    impedance_from_eps_sigma,
    select_beta_band, fit_single_cole_cole_subband
)
from src.itis_database_loader import (
    preencher_parametros_tecido, gerar_template_csv_itis,
    carregar_de_csv_itis, validar_parametros_fisicos
)
from src.data_generation import frequency_grid
from src.cole_model import cole_cole_impedance

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                      "axes.spines.right": False, "font.size": 11})


# ===========================================================================
# Geração do template CSV para consulta manual à calculadora oficial
# ===========================================================================

def gerar_template_csv(filepath="dados_fcc_consultados.csv",
                        n_freq=30, f_min=100.0, f_max=1e6):
    """
    Gera um CSV-template com a grade de frequências que você deve
    consultar na calculadora oficial da FCC ou IFAC-CNR, com colunas
    vazias para preencher manualmente.
    """
    f = frequency_grid(f_min, f_max, n_freq)
    with open(filepath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["freq_hz", "eps_prime", "sigma_S_per_m", "fonte"])
        for fi in f:
            writer.writerow([f"{fi:.4f}", "", "", ""])

    print(f"\nTemplate salvo em: {filepath}")
    print(f"\n{'='*70}")
    print("  PRÓXIMO PASSO (manual, ~5 minutos):")
    print(f"{'='*70}")
    print("  1. Acesse: https://www.fcc.gov/general/body-tissue-dielectric-parameters")
    print("  2. Selecione o tecido (ex.: Muscle)")
    print(f"  3. Para cada uma das {n_freq} frequências no CSV, consulte")
    print("     ε' (permittivity) e σ (conductivity, S/m)")
    print("  4. Preencha as colunas 'eps_prime' e 'sigma_S_per_m'")
    print("  5. Rode: python main_option_c.py --csv-fcc "
          f"{filepath}")
    print(f"{'='*70}\n")
    return filepath


def carregar_csv_fcc(filepath):
    """Carrega o CSV preenchido com valores consultados na FCC/IFAC-CNR."""
    freqs, eps_vals, sigma_vals = [], [], []
    with open(filepath, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                f_val = float(row["freq_hz"])
                eps_val = float(row["eps_prime"])
                sig_val = float(row["sigma_S_per_m"])
                freqs.append(f_val)
                eps_vals.append(eps_val)
                sigma_vals.append(sig_val)
            except (ValueError, KeyError):
                continue

    if len(freqs) < 5:
        raise ValueError(
            f"CSV '{filepath}' tem apenas {len(freqs)} linhas preenchidas. "
            "Preencha as colunas eps_prime e sigma_S_per_m com valores "
            "consultados na calculadora oficial antes de rodar."
        )

    idx = np.argsort(freqs)
    return (np.array(freqs)[idx], np.array(eps_vals)[idx],
            np.array(sigma_vals)[idx])


# ===========================================================================
# Figuras
# ===========================================================================

def plot_full_spectrum_with_band(f_full, Z_full, f_band, Z_band,
                                  band_range, save_path=None):
    """
    Mostra o espectro completo (modelo Gabriel) com a sub-banda β
    destacada — visualização honesta do que é "tudo" vs "identificável".
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_ny, ax_mod, ax_ph = axes

    ax_ny.plot(Z_full.real, -Z_full.imag, '--', color="gray",
               alpha=0.6, label="Espectro completo (Gabriel 4-CC)")
    ax_ny.plot(Z_band.real, -Z_band.imag, 'o-', color="#1f77b4",
               ms=4, label=f"Sub-banda β [{band_range[0]:.0f}, {band_range[1]:.0f}] Hz")
    ax_ny.set_xlabel(r"$\Re[Z]$ [Ω]"); ax_ny.set_ylabel(r"$-\Im[Z]$ [Ω]")
    ax_ny.set_title("(a) Nyquist"); ax_ny.legend(fontsize=8)

    ax_mod.loglog(f_full, np.abs(Z_full), '--', color="gray", alpha=0.6,
                  label="Completo")
    ax_mod.loglog(f_band, np.abs(Z_band), 'o-', color="#1f77b4", ms=4,
                  label="Sub-banda β")
    ax_mod.axvspan(band_range[0], band_range[1], alpha=0.1, color="#1f77b4")
    ax_mod.set_xlabel("Frequência [Hz]"); ax_mod.set_ylabel(r"$|Z|$ [Ω]")
    ax_mod.set_title("(b) Módulo"); ax_mod.legend(fontsize=8)

    ax_ph.semilogx(f_full, np.angle(Z_full, deg=True), '--', color="gray", alpha=0.6)
    ax_ph.semilogx(f_band, np.angle(Z_band, deg=True), 'o-', color="#1f77b4", ms=4)
    ax_ph.axvspan(band_range[0], band_range[1], alpha=0.1, color="#1f77b4")
    ax_ph.set_xlabel("Frequência [Hz]"); ax_ph.set_ylabel("Fase [°]")
    ax_ph.set_title("(c) Fase")

    fig.suptitle("Espectro Dielétrico Completo (Gabriel 4-CC) vs. Sub-banda Identificável",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


def plot_subband_fit_result(result, true_params_subband=None, save_path=None):
    """Mostra o ajuste Cole-Cole de 1 dispersão na sub-banda β."""
    from src.analysis import samples_to_physical, compute_hdi

    f_band = result["f_band"]
    Z_band = result["Z_band"]
    samples = result["mcmc_samples"]
    phys = samples_to_physical(samples)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_ny, ax_mod, ax_ph = axes

    R_med = np.median(phys["R_inf"]); R0_med = np.median(phys["R0"])
    t_med = np.median(phys["tau"]);   a_med = np.median(phys["alpha"])
    Z_fit = cole_cole_impedance(f_band, R_med, R0_med, t_med, a_med)

    ax_ny.scatter(Z_band.real, -Z_band.imag, color="gray", s=25,
                  label="Dados (sub-banda β)", zorder=5)
    ax_ny.plot(Z_fit.real, -Z_fit.imag, color="#1f77b4", lw=2,
               label="Ajuste Cole-Cole (1 dispersão)")
    ax_ny.set_xlabel(r"$\Re[Z]$ [Ω]"); ax_ny.set_ylabel(r"$-\Im[Z]$ [Ω]")
    ax_ny.set_title(f"(a) Nyquist — R²={result['r_squared_subband']:.4f}")
    ax_ny.legend(fontsize=8)
    ax_ny.set_aspect("equal", adjustable="datalim")

    ax_mod.semilogx(f_band, np.abs(Z_band), 'o', color="gray", ms=5)
    ax_mod.semilogx(f_band, np.abs(Z_fit), color="#1f77b4", lw=2)
    ax_mod.set_xlabel("Frequência [Hz]"); ax_mod.set_ylabel(r"$|Z|$ [Ω]")
    ax_mod.set_title("(b) Módulo")

    ax_ph.semilogx(f_band, np.angle(Z_band, deg=True), 'o', color="gray", ms=5)
    ax_ph.semilogx(f_band, np.angle(Z_fit, deg=True), color="#1f77b4", lw=2)
    ax_ph.set_xlabel("Frequência [Hz]"); ax_ph.set_ylabel("Fase [°]")
    ax_ph.set_title("(c) Fase")

    cov_pct = result["coverage_fraction"] * 100
    fig.suptitle(
        f"Ajuste Cole-Cole — Sub-banda β honesta "
        f"({cov_pct:.0f}% do espectro original)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figura salva: {save_path}")
    plt.show()
    return fig


# ===========================================================================
# Pipelines
# ===========================================================================

def run_demo_mode(n_amostras, n_warmup, seed, pasta):
    """
    Modo demonstração: usa MUSCLE_4CC_PARTIAL (parâmetros NÃO verificados)
    apenas para validar que o código funciona. NÃO usar para o artigo.
    """
    print("\n" + "⚠️ "*20)
    print("  MODO DEMONSTRAÇÃO — parâmetros NÃO verificados")
    print("  NÃO use estes resultados no artigo final.")
    print("  Rode com --csv-fcc após consultar a calculadora oficial.")
    print("⚠️ "*20 + "\n")

    os.makedirs(pasta, exist_ok=True)
    f_full = frequency_grid(10.0, 1e8, 200)   # banda larga para ver as 4 dispersões
    Z_full = impedance_from_4cc(f_full, MUSCLE_4CC_PARTIAL)

    f_grid = frequency_grid(100.0, 1e6, 30)   # nossa grade experimental
    Z_grid = impedance_from_4cc(f_grid, MUSCLE_4CC_PARTIAL)

    f_band, Z_band_clean = select_beta_band(f_grid, Z_grid, 1e3, 1e6)

    plot_full_spectrum_with_band(
        f_full, Z_full, f_band, Z_band_clean, (1e3, 1e6),
        save_path=os.path.join(pasta, "demo_espectro_completo_vs_banda.png")
    )

    result = fit_single_cole_cole_subband(
        f_grid, Z_grid, f_low=1e3, f_high=1e6,
        snr_db=30.0, n_amostras=n_amostras, n_warmup=n_warmup, seed=seed
    )

    print(f"\n  R² (sub-banda): {result['r_squared_subband']:.4f}")
    print(f"  Cobertura: {result['coverage_fraction']*100:.0f}% do espectro original")
    print(f"  R_inf={result['params_median']['R_inf']:.2f} Ω  "
          f"R0={result['params_median']['R0']:.2f} Ω  "
          f"tau={result['params_median']['tau']*1e6:.2f} µs  "
          f"alpha={result['params_median']['alpha']:.3f}")

    plot_subband_fit_result(
        result,
        save_path=os.path.join(pasta, "demo_ajuste_subbanda.png")
    )
    return result


def run_itis_csv_mode(csv_path, n_amostras, n_warmup, seed, pasta,
                       snr_db=30.0, R_inf_calib=50.0):
    """
    Pipeline com parâmetros brutos de Gabriel (4-CC) lidos da planilha
    oficial da IT'IS Foundation. Esta é a forma RECOMENDADA de validação
    — um único arquivo, todos os tecidos, sem consultas pontuais.
    """
    print("\n" + "="*60)
    print("  OPÇÃO C (v2) — Validação com banco de dados IT'IS Foundation")
    print("  Fonte primária: Gabriel et al. (1996)")
    print("="*60)

    os.makedirs(pasta, exist_ok=True)
    tecidos = carregar_de_csv_itis(csv_path)

    resultados = {}
    for nome, params in tecidos.items():
        print(f"\n  ── Tecido: {nome} ──")

        # Validação física automática (detecta erros de transcrição)
        ok = validar_parametros_fisicos(params)
        if not ok:
            print(f"  ⚠️  Pulando '{nome}' — corrija os parâmetros e rode novamente.")
            continue

        f_grid = frequency_grid(100.0, 1e6, 30)
        Z_grid = impedance_from_4cc_calibrated(
            f_grid, params, R_inf_desejado=R_inf_calib, f_ref=1e6)

        f_band, Z_band_clean = select_beta_band(f_grid, Z_grid, 1e3, 1e6)
        print(f"  Sub-banda β: {len(f_band)}/{len(f_grid)} pontos "
              f"({len(f_band)/len(f_grid)*100:.0f}%)")

        plot_full_spectrum_with_band(
            f_grid, Z_grid, f_band, Z_band_clean, (1e3, 1e6),
            save_path=os.path.join(pasta, f"{nome}_espectro_completo_vs_banda.png")
        )

        result = fit_single_cole_cole_subband(
            f_grid, Z_grid, f_low=1e3, f_high=1e6,
            snr_db=snr_db, n_amostras=n_amostras, n_warmup=n_warmup, seed=seed
        )

        print(f"  R² (sub-banda): {result['r_squared_subband']:.4f}")
        p = result["params_median"]
        print(f"  R_inf={p['R_inf']:.2f} Ω  R0={p['R0']:.2f} Ω  "
              f"tau={p['tau']*1e6:.2f} µs  alpha={p['alpha']:.3f}")

        plot_subband_fit_result(
            result,
            save_path=os.path.join(pasta, f"{nome}_ajuste_subbanda.png")
        )
        resultados[nome] = result

    print(f"\n  Resultados salvos em: {os.path.abspath(pasta)}/")
    return resultados



    """
    Pipeline com dados reais consultados na calculadora oficial FCC/IFAC-CNR.
    Esta é a validação cientificamente honesta para o artigo.
    """
    print("\n" + "="*60)
    print("  OPÇÃO C — Validação com dados oficiais FCC/IFAC-CNR")
    print("="*60)

    os.makedirs(pasta, exist_ok=True)
    f_fcc, eps_fcc, sigma_fcc = carregar_csv_fcc(csv_path)
    print(f"\n  Carregados {len(f_fcc)} pontos de '{csv_path}'")
    print(f"  Faixa: [{f_fcc.min():.0f}, {f_fcc.max():.0f}] Hz")

    Z_full = impedance_from_eps_sigma(f_fcc, eps_fcc, sigma_fcc)
    print(f"  |Z| range: [{np.abs(Z_full).min():.2f}, {np.abs(Z_full).max():.2f}] Ω")

    f_band, Z_band_clean = select_beta_band(f_fcc, Z_full, 1e3, 1e6)
    print(f"  Sub-banda β: {len(f_band)}/{len(f_fcc)} pontos")

    plot_full_spectrum_with_band(
        f_fcc, Z_full, f_band, Z_band_clean, (1e3, 1e6),
        save_path=os.path.join(pasta, "fcc_espectro_completo_vs_banda.png")
    )

    result = fit_single_cole_cole_subband(
        f_fcc, Z_full, f_low=1e3, f_high=1e6,
        snr_db=snr_db, n_amostras=n_amostras, n_warmup=n_warmup, seed=seed
    )

    print(f"\n  R² (sub-banda): {result['r_squared_subband']:.4f}")
    print(f"  Cobertura: {result['coverage_fraction']*100:.0f}% do espectro original")
    print(f"  R_inf={result['params_median']['R_inf']:.2f} Ω  "
          f"R0={result['params_median']['R0']:.2f} Ω  "
          f"tau={result['params_median']['tau']*1e6:.2f} µs  "
          f"alpha={result['params_median']['alpha']:.3f}")

    plot_subband_fit_result(
        result,
        save_path=os.path.join(pasta, "fcc_ajuste_subbanda.png")
    )

    print(f"\n  Resultados salvos em: {os.path.abspath(pasta)}/")
    return result


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Opção C — Validação honesta de banda limitada")
    p.add_argument("--gerar-template", dest="gerar_template", action="store_true",
                   help="[Método antigo] Gera CSV-template para consulta manual na FCC")
    p.add_argument("--gerar-template-itis", dest="gerar_template_itis", action="store_true",
                   help="[RECOMENDADO] Gera CSV-template para preencher com a "
                        "planilha oficial da IT'IS Foundation (14 parâmetros/tecido)")
    p.add_argument("--csv-fcc", type=str, default=None,
                   help="[Método antigo] CSV com (freq, eps', sigma) consultados na FCC")
    p.add_argument("--csv-itis", type=str, default=None,
                   help="[RECOMENDADO] CSV com os 14 parâmetros de 4-CC lidos da "
                        "planilha oficial da IT'IS Foundation")
    p.add_argument("--r-inf-calib", type=float, default=50.0,
                   help="Resistência de referência [Ω] para calibrar a geometria "
                        "de medição (padrão: 50 Ω, típico de tecido mole)")
    p.add_argument("--demo", action="store_true",
                   help="Modo demonstração (parâmetros NÃO verificados, só teste de código)")
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--snr", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--saida", type=str, default="resultados_opcaoC")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gerar_template:
        gerar_template_csv()
        return

    if args.gerar_template_itis:
        gerar_template_csv_itis()
        return

    n_amostras = 1500 if args.rapido else 4000
    n_warmup   = 750  if args.rapido else 2000

    if args.csv_itis:
        run_itis_csv_mode(args.csv_itis, n_amostras, n_warmup,
                           args.seed, args.saida, args.snr, args.r_inf_calib)
    elif args.csv_fcc:
        run_fcc_csv_mode(args.csv_fcc, n_amostras, n_warmup,
                          args.seed, args.saida, args.snr)
    elif args.demo:
        run_demo_mode(n_amostras, n_warmup, args.seed, args.saida)
    else:
        print("Especifique uma opção:")
        print("  --gerar-template-itis  (RECOMENDADO — download único)")
        print("  --csv-itis <arquivo>   (RECOMENDADO — rodar com dados reais)")
        print("  --gerar-template / --csv-fcc <arquivo>  (método antigo, manual)")
        print("  --demo                 (teste de código apenas)")
        print("Use --help para mais informações")


if __name__ == "__main__":
    main()
