"""
diagnosticar_csv.py
====================
Ferramenta de diagnóstico que lê DIRETAMENTE do seu CSV já preenchido
(parametros_itis_template.csv ou similar) e identifica automaticamente
se a fase positiva do Liver é:

  (A) Erro de transcrição (ex.: unidade de tau errada, coluna trocada)
  (B) Efeito real de interferência entre dispersões com τ próximos
  (C) Outro problema no pipeline

USO:
    python diagnosticar_csv.py parametros_itis_template.csv liver
    python diagnosticar_csv.py parametros_itis_template.csv muscle  # controle
"""

import sys
import csv
import numpy as np

EPS0 = 8.8541878128e-12


def carregar_linha_csv(filepath, tissue_name):
    """Carrega os 14 parâmetros de uma linha específica do CSV."""
    with open(filepath, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["tissue_name"].strip().lower() == tissue_name.lower():
                tau_unit = row.get("tau_unit", "s") or "s"
                unit_factor = {"s": 1.0, "ms": 1e-3, "us": 1e-6,
                               "ns": 1e-9, "ps": 1e-12}
                f = unit_factor.get(tau_unit, 1.0)

                eps_inf = float(row["eps_inf"])
                sigma_s = float(row["sigma_s"])
                delta_eps = [float(row[f"delta_eps_{i}"]) for i in range(1, 5)]
                tau_s = [float(row[f"tau_{i}"]) * f for i in range(1, 5)]
                alpha = [float(row[f"alpha_{i}"]) for i in range(1, 5)]

                return {
                    "eps_inf": eps_inf, "sigma_s": sigma_s,
                    "delta_eps": delta_eps, "tau_s": tau_s, "alpha": alpha,
                    "tau_raw": [float(row[f"tau_{i}"]) for i in range(1, 5)],
                    "tau_unit": tau_unit
                }
    raise ValueError(f"Tecido '{tissue_name}' não encontrado em '{filepath}'")


def calc_eps_star(omega, eps_inf, sigma_s, delta_eps, tau_s, alpha):
    eps = np.full_like(omega, eps_inf, dtype=complex)
    for d, t, a in zip(delta_eps, tau_s, alpha):
        eps += d / (1 + (1j * omega * t) ** (1 - a))
    eps += sigma_s / (1j * omega * EPS0)
    return eps


def calc_fase(omega, eps_inf, sigma_s, delta_eps, tau_s, alpha):
    eps_star = calc_eps_star(omega, eps_inf, sigma_s, delta_eps, tau_s, alpha)
    Y = 1j * omega * EPS0 * eps_star
    Z = 1.0 / Y
    return np.angle(Z, deg=True)


def diagnosticar(filepath, tissue_name):
    print("=" * 65)
    print(f"  DIAGNÓSTICO: {tissue_name.upper()}")
    print(f"  Fonte: {filepath}")
    print("=" * 65)

    p = carregar_linha_csv(filepath, tissue_name)
    print(f"\n  eps_inf = {p['eps_inf']}")
    print(f"  sigma_s = {p['sigma_s']} S/m")
    print(f"  tau_unit declarada = '{p['tau_unit']}'")
    print()
    for i in range(4):
        print(f"  Dispersão {i+1}: Δε={p['delta_eps'][i]:.4g}  "
              f"τ={p['tau_raw'][i]:.4g}{p['tau_unit']} "
              f"(={p['tau_s'][i]:.4e}s)  α={p['alpha'][i]:.3f}")

    # ── Teste 1: validade física básica dos parâmetros ─────────────
    print("\n[Teste 1] Validade física básica (pré-requisito matemático):")
    print("  Teorema: soma de dispersões Cole-Cole com Δε>0, τ>0, 0≤α<1,")
    print("  σ_s≥0 SEMPRE tem fase ≤0 — não importa quão próximos os τ.")
    print("  Logo, fase positiva só pode vir de parâmetro fisicamente inválido")
    print("  ou bug de código — não de 'interferência' entre dispersões.\n")

    problemas_validade = []
    for i in range(4):
        if p["delta_eps"][i] <= 0:
            problemas_validade.append(f"  ⚠️  Δε{i+1} = {p['delta_eps'][i]:.4g} ≤ 0 — INVÁLIDO")
        if p["tau_s"][i] <= 0:
            problemas_validade.append(f"  ⚠️  τ{i+1} = {p['tau_s'][i]:.4g} ≤ 0 — INVÁLIDO")
        if not (0 <= p["alpha"][i] < 1):
            problemas_validade.append(f"  ⚠️  α{i+1} = {p['alpha'][i]:.4g} fora de [0,1) — INVÁLIDO")
    if p["sigma_s"] < 0:
        problemas_validade.append(f"  ⚠️  σ_s = {p['sigma_s']:.4g} < 0 — INVÁLIDO")

    if problemas_validade:
        print("  PARÂMETROS INVÁLIDOS ENCONTRADOS (causa raiz provável):")
        for prob in problemas_validade:
            print(prob)
    else:
        print("  ✓ Todos os 14 parâmetros satisfazem os critérios de validade.")
        print("    (Δε>0, τ>0, 0≤α<1, σ_s≥0 — não há violação óbvia)")

    # ── Teste 2: ordenação física dos tau ──────────────────────────
    print("\n[Teste 2] Os τ devem decrescer: τ1 > τ2 > τ3 > τ4")
    print("          (dispersão 1=α mais lenta ... 4=γ mais rápida)")
    tau_s = p["tau_s"]
    ordem_correta = all(tau_s[i] > tau_s[i+1] for i in range(3))
    if ordem_correta:
        print("  ✓ Ordem correta (τ1 > τ2 > τ3 > τ4)")
    else:
        print("  ⚠️  ORDEM INCORRETA — possível troca de colunas/linhas na transcrição!")
        for i in range(3):
            rel = "✓" if tau_s[i] > tau_s[i+1] else "✗ FORA DE ORDEM"
            print(f"     τ{i+1}={tau_s[i]:.3e}s vs τ{i+2}={tau_s[i+1]:.3e}s  {rel}")

    # ── Teste 3: fase na banda crítica 1-50kHz ─────────────────────
    print("\n[Teste 3] Fase calculada na banda 1-50 kHz:")
    f_band = np.logspace(3, np.log10(50e3), 100)
    omega_band = 2 * np.pi * f_band
    fase = calc_fase(omega_band, p["eps_inf"], p["sigma_s"],
                      p["delta_eps"], p["tau_s"], p["alpha"])
    print(f"  Fase min={fase.min():.4f}°  max={fase.max():.4f}°")

    if fase.max() > 0.01:
        idx_max = np.argmax(fase)
        print(f"  ⚠️  FASE POSITIVA CONFIRMADA: {fase.max():.4f}° em f={f_band[idx_max]:.0f}Hz")
        positivo = True
    else:
        print(f"  ✓ Fase sempre negativa com estes parâmetros.")
        positivo = False

    # ── Teste 4: isolar qual dispersão causa o problema ────────────
    if positivo:
        print("\n[Teste 4] Removendo cada dispersão para isolar a causa:")
        for remove_idx in range(4):
            d2 = [d for i, d in enumerate(p["delta_eps"]) if i != remove_idx]
            t2 = [t for i, t in enumerate(p["tau_s"]) if i != remove_idx]
            a2 = [a for i, a in enumerate(p["alpha"]) if i != remove_idx]
            fase_sem = calc_fase(omega_band, p["eps_inf"], p["sigma_s"], d2, t2, a2)
            diff = fase.max() - fase_sem.max()
            marca = "🎯 CULPADO PRINCIPAL" if abs(diff) > 0.05 else ""
            print(f"  Sem dispersão {remove_idx+1}: fase max muda em "
                  f"{diff:+.4f}°  {marca}")

        print("\n[Teste 5] Testando se trocar a unidade de tau resolve:")
        for unit_test, factor in [("ps", 1e-12), ("ns", 1e-9),
                                    ("us", 1e-6), ("ms", 1e-3)]:
            if unit_test == p["tau_unit"]:
                continue
            tau_s_test = [t * factor for t in p["tau_raw"]]
            fase_test = calc_fase(omega_band, p["eps_inf"], p["sigma_s"],
                                  p["delta_eps"], tau_s_test, p["alpha"])
            status = "✓ RESOLVERIA o problema!" if fase_test.max() <= 0.01 else f"ainda positiva ({fase_test.max():.3f}°)"
            print(f"  Se tau_unit fosse '{unit_test}' em vez de '{p['tau_unit']}': {status}")

    # ── Conclusão ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CONCLUSÃO")
    print("=" * 65)

    if problemas_validade:
        print("  → AÇÃO NECESSÁRIA: parâmetro(s) fisicamente inválido(s)")
        print("    detectado(s) no Teste 1, independentemente de já estarem")
        print("    causando fase positiva nesta banda específica ou não.")
        print("    Um Δε negativo, τ negativo, α≥1 ou σ_s<0 é SEMPRE um erro")
        print("    de transcrição ou de leitura da planilha — corrija antes")
        print("    de prosseguir, mesmo que o Teste 3 mostre fase OK aqui.")
        print("    (parâmetros errados podem ainda gerar problemas em outras")
        print("    bandas ou degradar a qualidade do ajuste sutilmente)")
    elif not ordem_correta:
        print("  → CAUSA MAIS PROVÁVEL: τ fora de ordem decrescente —")
        print("    indica troca de linha/coluna na transcrição da planilha.")
        print("    Mesmo sem causar fase positiva nesta banda, a ordem errada")
        print("    associa cada Δε/α ao τ errado, distorcendo o espectro")
        print("    completo. Revise a leitura da planilha original.")
    elif not positivo:
        print("  ✓ Todos os parâmetros são válidos, bem ordenados, e a fase")
        print("    calculada é sempre negativa — nenhum problema detectado")
        print("    nestes 14 parâmetros.")
        print("    Se a FIGURA gerada anteriormente mostrou fase positiva,")
        print("    o problema está em OUTRA parte do pipeline: confirme que")
        print("    o CSV usado para gerar aquela figura é este mesmo arquivo,")
        print("    e/ou que a versão do código usada já tem a correção do")
        print("    sinal em impedance_from_4cc() (bug corrigido anteriormente).")
    else:
        print("  → INCOMUM: todos os 14 parâmetros parecem válidos e bem")
        print("    ordenados, mas a fase ainda é positiva. Isso seria uma")
        print("    violação do teorema de funções de Herglotz — verifique")
        print("    o Teste 4/5 acima para isolar a dispersão responsável,")
        print("    e confira ESSE valor específico contra a planilha mais")
        print("    uma vez (provável dígito ou unidade incorreta).")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    diagnosticar(sys.argv[1], sys.argv[2])
