"""
itis_database_loader.py
========================
Carregador dos parâmetros de 4-Cole-Cole diretamente do banco de dados
oficial da IT'IS Foundation (mesma fonte primária da calculadora FCC:
Gabriel et al., 1996), via download local do ZIP oficial.

POR QUE ESTA VERSÃO É MELHOR QUE A CONSULTA MANUAL (main_option_c.py v1)
==========================================================================
A versão anterior exigia consultar a calculadora FCC frequência por
frequência (30 consultas manuais). Esta versão usa o banco de dados
bruto da IT'IS Foundation, que contém os **14 parâmetros originais**
(ε∞, σ_s, Δε₁₋₄, τ₁₋₄, α₁₋₄) para cada tecido — a mesma fonte primária
usada pela calculadora da FCC, mas em formato tabular, baixável uma
única vez.

Com os 14 parâmetros, geramos o espectro completo Z(f) em qualquer
frequência automaticamente (sem necessidade de consultas pontuais).

PASSO A PASSO PARA OBTER O ARQUIVO (5 minutos, only uma vez)
===============================================================
1. Baixe o arquivo ZIP oficial (testado e acessível em 24/06/2026):
   https://itis.swiss/assets/Downloads/TissueDb/Database-V4-0.zip

   (Página de referência: https://itis.swiss/virtual-population/tissue-properties/downloads)

   Se este link não funcionar, tente a versão mais recente:
   https://itis.swiss/virtual-population/tissue-properties/downloads/database-v5-0

2. Extraia o ZIP. Dentro dele há uma pasta com arquivos em 3 formatos:
   - .db   (para Sim4Life/SEMCAD X — não precisamos)
   - .xlsx (Excel — RECOMENDADO, mais fácil de abrir)
   - .txt/.csv (ASCII)

3. Abra o arquivo Excel e procure a aba/planilha de
   **"Dielectric Properties"** ou **"4-Cole-Cole"**.

4. Localize a linha do tecido **"Muscle"** (ou "Skeletal Muscle").
   As colunas terão nomes como:

   | Coluna | Significado |
   |---|---|
   | ε∞ (eps_inf) | Permissividade em alta frequência |
   | σ_s (sigma_s) | Condutividade iônica estática [S/m] |
   | Δε1, τ1, α1 | Dispersão 1 (α — sub-Hz a Hz) |
   | Δε2, τ2, α2 | Dispersão 2 (β — kHz, a que nos interessa) |
   | Δε3, τ3, α3 | Dispersão 3 (δ — MHz) |
   | Δε4, τ4, α4 | Dispersão 4 (γ — GHz, água livre) |

5. Preencha esses 14 valores em `preencher_parametros_muscle()` abaixo,
   ou use `carregar_de_csv_itis()` se exportar a linha para CSV.

Referências:
    Gabriel, C. (1996). Compilation of the Dielectric Properties of Body
    Tissues at RF and Microwave Frequencies. Report AL/OE-TR-1996-0037,
    Brooks Air Force Base, Texas.

    IT'IS Foundation (2018). Tissue Properties Database V4.0.
    DOI: 10.13099/VIP21000-04-0
    https://itis.swiss/virtual-population/tissue-properties/downloads
"""

import numpy as np
import csv
from pathlib import Path
from .gabriel_4cc_model import FourColeColeParams


# ===========================================================================
# OPÇÃO A — Preencher manualmente após consultar o Excel/ASCII baixado
# ===========================================================================

def preencher_parametros_tecido(
    tissue_name: str,
    eps_inf: float,
    sigma_s: float,
    delta_eps_1: float, tau_1: float, alpha_1: float,
    delta_eps_2: float, tau_2: float, alpha_2: float,
    delta_eps_3: float, tau_3: float, alpha_3: float,
    delta_eps_4: float, tau_4: float, alpha_4: float,
    tau_unit: str = "s"
) -> FourColeColeParams:
    """
    Constrói os parâmetros de 4-Cole-Cole a partir dos 14 valores que
    você leu diretamente da planilha oficial da IT'IS Foundation.

    IMPORTANTE sobre unidades de τ:
    A planilha da IT'IS geralmente reporta τ em **picossegundos (ps)**
    para as dispersões mais rápidas. Verifique o cabeçalho da coluna!
    Se a planilha disser "tau (psec)", use tau_unit="ps".

    Exemplo de uso (você preenche com os valores REAIS do Excel):

        params = preencher_parametros_tecido(
            tissue_name="muscle",
            eps_inf=4.0,
            sigma_s=0.2,
            delta_eps_1=..., tau_1=..., alpha_1=0.10,   # ler do Excel
            delta_eps_2=..., tau_2=..., alpha_2=0.10,   # ler do Excel
            delta_eps_3=..., tau_3=..., alpha_3=0.22,   # ler do Excel
            delta_eps_4=..., tau_4=..., alpha_4=0.00,   # ler do Excel
            tau_unit="ps"   # ajustar conforme a planilha
        )
    """
    unit_factor = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
    if tau_unit not in unit_factor:
        raise ValueError(f"tau_unit deve ser um de {list(unit_factor)}")
    f = unit_factor[tau_unit]

    return FourColeColeParams(
        tissue_name=tissue_name,
        eps_inf=eps_inf,
        sigma_s=sigma_s,
        delta_eps=[delta_eps_1, delta_eps_2, delta_eps_3, delta_eps_4],
        tau=[tau_1 * f, tau_2 * f, tau_3 * f, tau_4 * f],
        alpha=[alpha_1, alpha_2, alpha_3, alpha_4],
        source=f"IT'IS Foundation Tissue Properties Database V4.0/V5.0 "
                f"(Gabriel et al., 1996) — valores lidos manualmente da "
                f"planilha oficial pelo usuário"
    )


# ===========================================================================
# OPÇÃO B — Carregar de um CSV exportado da planilha (mais robusto)
# ===========================================================================

def gerar_template_csv_itis(filepath="parametros_itis_template.csv"):
    """
    Gera um CSV-template com as 14 colunas exatas que você deve copiar
    da planilha oficial da IT'IS Foundation para o tecido de interesse.
    """
    header = ["tissue_name", "eps_inf", "sigma_s",
              "delta_eps_1", "tau_1", "alpha_1",
              "delta_eps_2", "tau_2", "alpha_2",
              "delta_eps_3", "tau_3", "alpha_3",
              "delta_eps_4", "tau_4", "alpha_4",
              "tau_unit"]
    with open(filepath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow(["muscle", "", "", "", "", "", "", "", "",
                         "", "", "", "", "", "", "ps"])
        writer.writerow(["liver", "", "", "", "", "", "", "", "",
                         "", "", "", "", "", "", "ps"])
        writer.writerow(["fat", "", "", "", "", "", "", "", "",
                         "", "", "", "", "", "", "ps"])

    print(f"\nTemplate salvo em: {filepath}")
    print("\nPreencha cada linha com os 14 parâmetros lidos da planilha")
    print("oficial da IT'IS Foundation (Database-V4-0.zip ou V5.0).")
    print("Verifique a unidade de tau na planilha (geralmente ps) e")
    print("ajuste a coluna 'tau_unit' se necessário.")
    return filepath


def carregar_de_csv_itis(filepath: str) -> dict:
    """
    Carrega múltiplos tecidos de um CSV preenchido com os parâmetros
    lidos da planilha oficial da IT'IS Foundation.

    Retorna dict {tissue_name: FourColeColeParams}
    """
    resultado = {}
    with open(filepath, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                p = preencher_parametros_tecido(
                    tissue_name=row["tissue_name"],
                    eps_inf=float(row["eps_inf"]),
                    sigma_s=float(row["sigma_s"]),
                    delta_eps_1=float(row["delta_eps_1"]),
                    tau_1=float(row["tau_1"]),
                    alpha_1=float(row["alpha_1"]),
                    delta_eps_2=float(row["delta_eps_2"]),
                    tau_2=float(row["tau_2"]),
                    alpha_2=float(row["alpha_2"]),
                    delta_eps_3=float(row["delta_eps_3"]),
                    tau_3=float(row["tau_3"]),
                    alpha_3=float(row["alpha_3"]),
                    delta_eps_4=float(row["delta_eps_4"]),
                    tau_4=float(row["tau_4"]),
                    alpha_4=float(row["alpha_4"]),
                    tau_unit=row.get("tau_unit", "s") or "s"
                )
                resultado[row["tissue_name"]] = p
            except (ValueError, KeyError) as e:
                print(f"  [aviso] linha '{row.get('tissue_name','?')}' "
                      f"incompleta ou inválida — pulando ({e})")
                continue

    if not resultado:
        raise ValueError(
            f"Nenhum tecido válido carregado de '{filepath}'. "
            "Verifique se todas as 14 colunas numéricas estão preenchidas."
        )

    print(f"\n  {len(resultado)} tecido(s) carregado(s) de '{filepath}':")
    for name in resultado:
        print(f"    - {name}")
    return resultado


# ===========================================================================
# Validação física básica dos parâmetros carregados
# ===========================================================================

def validar_parametros_fisicos(p: FourColeColeParams, verbose=True) -> bool:
    """
    Verifica se os parâmetros carregados produzem um espectro
    fisicamente válido (ε' monotonicamente decrescente, fase sempre
    negativa). Detecta o tipo de erro identificado na rodada anterior
    (fase positiva por erro de transcrição).
    """
    from .gabriel_4cc_model import conductivity_and_permittivity, impedance_from_4cc

    f_test = np.logspace(1, 9, 200)
    sigma, eps_real = conductivity_and_permittivity(f_test, p)
    Z = impedance_from_4cc(f_test, p)
    phase = np.angle(Z, deg=True)

    problemas = []

    # ε' deve ser monotonicamente decrescente (ou no mínimo, não subir)
    d_eps = np.diff(eps_real)
    n_subidas = np.sum(d_eps > eps_real[:-1] * 0.01)  # tolerância de 1%
    if n_subidas > 2:  # pequenas flutuações numéricas são ok
        problemas.append(
            f"ε'(f) sobe em {n_subidas} pontos — deveria ser "
            "monotonicamente decrescente. Possível erro de transcrição."
        )

    # Fase deve ser sempre ≤ 0 (sistema passivo RC)
    n_fase_positiva = np.sum(phase > 0.5)  # tolerância numérica
    if n_fase_positiva > 0:
        idx_problema = np.where(phase > 0.5)[0]
        f_problema = f_test[idx_problema]
        problemas.append(
            f"Fase positiva detectada em {n_fase_positiva} pontos "
            f"(f≈{f_problema.min():.0f}–{f_problema.max():.0f} Hz). "
            "Isso é fisicamente impossível para este modelo — "
            "verifique sinais e unidades dos parâmetros transcritos."
        )

    # σ deve ser positivo e finito
    if np.any(sigma <= 0) or np.any(~np.isfinite(sigma)):
        problemas.append("σ(f) negativo, zero ou não-finito em algum ponto.")

    if verbose:
        if problemas:
            print(f"\n  ⚠️  PROBLEMAS DETECTADOS em '{p.tissue_name}':")
            for prob in problemas:
                print(f"    - {prob}")
        else:
            print(f"\n  ✓ Parâmetros de '{p.tissue_name}' passaram na "
                  "validação física básica.")

    return len(problemas) == 0
