import argparse
import os
import csv
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.gabriel_4cc_model import (
    MUSCLE_4CC_PARTIAL, impedance_from_4cc, impedance_from_4cc_calibrated,
    impedance_from_eps_sigma, select_beta_band, fit_single_cole_cole_subband
)
from src.itis_database_loader import (
    gerar_template_csv_itis, carregar_de_csv_itis, validar_parametros_fisicos
)
from src.data_generation import frequency_grid
from src.cole_model import cole_cole_impedance

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                      "axes.spines.right": False, "font.size": 11})


def gerar_template_csv(filepath="dados_fcc_consultados.csv",
                        n_freq=30, f_min=100.0, f_max=1e6):
    f = frequency_grid(f_min, f_max, n_freq)
    with open(filepath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["freq_hz", "eps_prime", "sigma_S_per_m", "fonte"])
        for fi in f:
            writer.writerow([f"{fi:.4f}", "", "", ""])
    print(f"\nTemplate salvo em: {filepath}")
    return filepath


def carregar_csv_fcc(filepath):
    freqs, eps_vals, sigma_vals = [], [], []
    with open(filepath, "r") as fh:
        for row in csv.DictReader(fh):
            try:
                freqs.append(float(row["freq_hz"]))
                eps_vals.append(float(row["eps_prime"]))
                sigma_vals.append(float(row["sigma_S_per_m"]))
            except (ValueError, KeyError):
                continue
    if len(freqs) < 5:
        raise ValueError(f"CSV '{filepath}' tem poucas linhas preenchidas.")
    idx = np.argsort(freqs)
    return (np.array(freqs)[idx], np.array(eps_vals)[idx], np.array(sigma_vals)[idx])


def plot_full_spectrum_with_band(f_full, Z_full, f_band, Z_band, band_range, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_ny, ax_mod, ax_ph = axes
    ax_ny.plot(Z_full.real, -Z_full.imag, '--', color="gray", alpha=0.6, label="Espectro completo")
    ax_ny.plot(Z_band.real, -Z_band.imag, 'o-', color="#1f77b4", ms=4, label=f"Sub-banda [{band_range[0]:.0f}, {band_range[1]:.0f}] Hz")
    ax_ny.set_xlabel(r"$\Re[Z]$"); ax_ny.set_ylabel(r"$-\Im[Z]$"); ax_ny.set_title("(a) Nyquist"); ax_ny.legend(fontsize=8)
    ax_mod.loglog(f_full, np.abs(Z_full), '--', color="gray", alpha=0.6, label="Completo")
    ax_mod.loglog(f_band, np.abs(Z_band), 'o-', color="#1f77b4", ms=4, label="Sub-banda")
    ax_mod.axvspan(band_range[0], band_range[1], alpha=0.1, color="#1f77b4")
    ax_mod.set_xlabel("Frequência [Hz]"); ax_mod.set_ylabel(r"$|Z|$"); ax_mod.set_title("(b) Módulo"); ax_mod.legend(fontsize=8)
    ax_ph.semilogx(f_full, np.angle(Z_full, deg=True), '--', color="gray", alpha=0.6)
    ax_ph.semilogx(f_band, np.angle(Z_band, deg=True), 'o-', color="#1f77b4", ms=4)
    ax_ph.axvspan(band_range[0], band_range[1], alpha=0.1, color="#1f77b4")
    ax_ph.set_xlabel("Frequência [Hz]"); ax_ph.set_ylabel("Fase [°]"); ax_ph.set_title("(c) Fase")
    fig.suptitle("Espectro Completo vs. Sub-banda Identificável", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return fig


def plot_subband_fit_result(result, save_path=None):
    from src.analysis import samples_to_physical
    f_band = result["f_band"]; Z_band = result["Z_band"]
    phys = samples_to_physical(result["mcmc_samples"])
    R_med = np.median(phys["R_inf"]); R0_med = np.median(phys["R0"])
    t_med = np.median(phys["tau"]); a_med = np.median(phys["alpha"])
    Z_fit = cole_cole_impedance(f_band, R_med, R0_med, t_med, a_med)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_ny, ax_mod, ax_ph = axes
    ax_ny.scatter(Z_band.real, -Z_band.imag, color="gray", s=25, label="Dados", zorder=5)
    ax_ny.plot(Z_fit.real, -Z_fit.imag, color="#1f77b4", lw=2, label="Ajuste")
    ax_ny.set_title(f"(a) Nyquist — R²={result['r_squared_subband']:.4f}"); ax_ny.legend(fontsize=8)
    ax_ny.set_aspect("equal", adjustable="datalim")
    ax_mod.semilogx(f_band, np.abs(Z_band), 'o', color="gray", ms=5)
    ax_mod.semilogx(f_band, np.abs(Z_fit), color="#1f77b4", lw=2)
    ax_mod.set_title("(b) Módulo")
    ax_ph.semilogx(f_band, np.angle(Z_band, deg=True), 'o', color="gray", ms=5)
    ax_ph.semilogx(f_band, np.angle(Z_fit, deg=True), color="#1f77b4", lw=2)
    ax_ph.set_title("(c) Fase")
    cov_pct = result["coverage_fraction"] * 100
    fig.suptitle(f"Ajuste Cole-Cole — Sub-banda ({cov_pct:.0f}%)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return fig


def run_demo_mode(n_amostras, n_warmup, seed, pasta):
    os.makedirs(pasta, exist_ok=True)
    f_full = frequency_grid(10.0, 1e8, 200)
    Z_full = impedance_from_4cc(f_full, MUSCLE_4CC_PARTIAL)
    f_grid = frequency_grid(100.0, 1e6, 30)
    Z_grid = impedance_from_4cc(f_grid, MUSCLE_4CC_PARTIAL)
    f_band, Z_band_clean = select_beta_band(f_grid, Z_grid, 1e3, 1e6)
    plot_full_spectrum_with_band(f_full, Z_full, f_band, Z_band_clean, (1e3, 1e6),
                                  save_path=os.path.join(pasta, "demo_espectro.png"))
    result = fit_single_cole_cole_subband(f_grid, Z_grid, f_low=1e3, f_high=1e6,
                                           snr_db=30.0, n_amostras=n_amostras, n_warmup=n_warmup, seed=seed)
    plot_subband_fit_result(result, save_path=os.path.join(pasta, "demo_ajuste.png"))
    return result


def run_itis_csv_mode(csv_path, n_amostras, n_warmup, seed, pasta, snr_db=30.0, R_inf_calib=50.0):
    os.makedirs(pasta, exist_ok=True)
    tecidos = carregar_de_csv_itis(csv_path)
    resultados = {}
    for nome, params in tecidos.items():
        if not validar_parametros_fisicos(params):
            continue
        f_grid = frequency_grid(100.0, 1e6, 30)
        Z_grid = impedance_from_4cc_calibrated(f_grid, params, R_inf_desejado=R_inf_calib, f_ref=1e6)
        f_band, Z_band_clean = select_beta_band(f_grid, Z_grid, 1e3, 1e6)
        plot_full_spectrum_with_band(f_grid, Z_grid, f_band, Z_band_clean, (1e3, 1e6),
                                      save_path=os.path.join(pasta, f"{nome}_espectro.png"))
        result = fit_single_cole_cole_subband(f_grid, Z_grid, f_low=1e3, f_high=1e6,
                                               snr_db=snr_db, n_amostras=n_amostras, n_warmup=n_warmup, seed=seed)
        plot_subband_fit_result(result, save_path=os.path.join(pasta, f"{nome}_ajuste.png"))
        resultados[nome] = result
    return resultados


def run_fcc_csv_mode(csv_path, n_amostras, n_warmup, seed, pasta, snr_db=30.0):
    os.makedirs(pasta, exist_ok=True)
    f_fcc, eps_fcc, sigma_fcc = carregar_csv_fcc(csv_path)
    Z_full = impedance_from_eps_sigma(f_fcc, eps_fcc, sigma_fcc)
    f_band, Z_band_clean = select_beta_band(f_fcc, Z_full, 1e3, 1e6)
    plot_full_spectrum_with_band(f_fcc, Z_full, f_band, Z_band_clean, (1e3, 1e6),
                                  save_path=os.path.join(pasta, "fcc_espectro.png"))
    result = fit_single_cole_cole_subband(f_fcc, Z_full, f_low=1e3, f_high=1e6,
                                           snr_db=snr_db, n_amostras=n_amostras, n_warmup=n_warmup, seed=seed)
    plot_subband_fit_result(result, save_path=os.path.join(pasta, "fcc_ajuste.png"))
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Opção C")
    p.add_argument("--gerar-template", dest="gerar_template", action="store_true")
    p.add_argument("--gerar-template-itis", dest="gerar_template_itis", action="store_true")
    p.add_argument("--csv-fcc", type=str, default=None)
    p.add_argument("--csv-itis", type=str, default=None)
    p.add_argument("--r-inf-calib", type=float, default=50.0)
    p.add_argument("--demo", action="store_true")
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
    n_warmup = 750 if args.rapido else 2000
    if args.csv_itis:
        run_itis_csv_mode(args.csv_itis, n_amostras, n_warmup, args.seed, args.saida, args.snr, args.r_inf_calib)
    elif args.csv_fcc:
        run_fcc_csv_mode(args.csv_fcc, n_amostras, n_warmup, args.seed, args.saida, args.snr)
    elif args.demo:
        run_demo_mode(n_amostras, n_warmup, args.seed, args.saida)
    else:
        print("Especifique uma opção. Use --help.")


if __name__ == "__main__":
    main()
