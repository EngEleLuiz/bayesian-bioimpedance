"""
itis_database_loader.py
========================
Loads 4-Cole-Cole parameters directly from the official IT'IS Foundation
tissue properties database (the same primary source used by the FCC
calculator: Gabriel et al., 1996), via a locally downloaded ZIP file.

WHY THIS IS BETTER THAN THE MANUAL FCC LOOKUP (main_option_c.py v1)
=====================================================================
The previous approach required querying the FCC calculator frequency by
frequency (30 manual lookups). This version uses the raw IT'IS Foundation
database, which contains the 14 original parameters
(eps_inf, sigma_s, delta_eps[1-4], tau[1-4], alpha[1-4]) for each tissue --
the same primary source used by the FCC calculator, but in tabular form,
downloaded once.

With the 14 parameters, the full spectrum Z(f) can be generated at any
frequency automatically (no need for point-by-point lookups).

HOW TO GET THE FILE (5 minutes, one time only)
=================================================
1. Download the official ZIP (verified reachable on 2026-06-24):
   https://itis.swiss/assets/Downloads/TissueDb/Database-V4-0.zip
   (Reference page: https://itis.swiss/virtual-population/tissue-properties/downloads)
   If that link doesn't work, try the newer version:
   https://itis.swiss/virtual-population/tissue-properties/downloads/database-v5-0

2. Extract the ZIP. It contains files in 3 formats:
   - .db   (Sim4Life/SEMCAD X -- not needed here)
   - .xlsx (Excel -- RECOMMENDED, easiest to open)
   - .txt/.csv (ASCII)

3. Open the Excel file and find the "Dielectric Properties" or
   "4-Cole-Cole" sheet.

4. Find the row for the tissue of interest, e.g. "Muscle"
   (or "Skeletal Muscle"). Columns look like:
   eps_inf, sigma_s, delta_eps1/tau1/alpha1 (alpha dispersion),
   delta_eps2/tau2/alpha2 (beta), delta_eps3/tau3/alpha3 (delta),
   delta_eps4/tau4/alpha4 (gamma).

5. Fill in these 14 values with build_tissue_params() below, or use
   load_itis_csv() if you export the row to a CSV.

References:
    Gabriel, C. (1996). Compilation of the Dielectric Properties of Body
    Tissues at RF and Microwave Frequencies. Report AL/OE-TR-1996-0037,
    Brooks Air Force Base, Texas.

    IT'IS Foundation (2018). Tissue Properties Database V4.0.
    DOI: 10.13099/VIP21000-04-0
    https://itis.swiss/virtual-population/tissue-properties/downloads
"""

import csv
import numpy as np
from .gabriel_4cc_model import FourColeColeParams

_TAU_UNIT_FACTOR = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}


def build_tissue_params(
    tissue_name: str, eps_inf: float, sigma_s: float,
    delta_eps_1: float, tau_1: float, alpha_1: float,
    delta_eps_2: float, tau_2: float, alpha_2: float,
    delta_eps_3: float, tau_3: float, alpha_3: float,
    delta_eps_4: float, tau_4: float, alpha_4: float,
    tau_unit: str = "s",
) -> FourColeColeParams:
    """
    Build 4-Cole-Cole parameters from the 14 values read directly from
    the official IT'IS Foundation spreadsheet.

    IMPORTANT about tau units: the IT'IS spreadsheet usually reports tau
    in picoseconds (ps) for the fastest dispersions. Check the column
    header! If it says "tau (psec)", use tau_unit="ps".
    """
    if tau_unit not in _TAU_UNIT_FACTOR:
        raise ValueError(f"tau_unit must be one of {list(_TAU_UNIT_FACTOR)}")
    factor = _TAU_UNIT_FACTOR[tau_unit]

    return FourColeColeParams(
        tissue_name=tissue_name, eps_inf=eps_inf, sigma_s=sigma_s,
        delta_eps=[delta_eps_1, delta_eps_2, delta_eps_3, delta_eps_4],
        tau=[tau_1 * factor, tau_2 * factor, tau_3 * factor, tau_4 * factor],
        alpha=[alpha_1, alpha_2, alpha_3, alpha_4],
        source="IT'IS Foundation Tissue Properties Database V4.0/V5.0 "
               "(Gabriel et al., 1996) -- values read manually from the "
               "official spreadsheet by the user",
        verified=True,
    )


def generate_itis_csv_template(filepath="itis_params_template.csv"):
    """Generate a CSV template with the 14 exact columns to copy from the IT'IS spreadsheet."""
    header = ["tissue_name", "eps_inf", "sigma_s",
              "delta_eps_1", "tau_1", "alpha_1",
              "delta_eps_2", "tau_2", "alpha_2",
              "delta_eps_3", "tau_3", "alpha_3",
              "delta_eps_4", "tau_4", "alpha_4",
              "tau_unit"]
    with open(filepath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for tissue in ("muscle", "liver", "fat"):
            writer.writerow([tissue] + [""] * 14 + ["ps"])

    print(f"\nTemplate saved to: {filepath}")
    print("\nFill each row with the 14 parameters read from the official")
    print("IT'IS Foundation spreadsheet (Database-V4-0.zip or V5.0).")
    print("Check the tau unit in the spreadsheet (usually ps) and adjust")
    print("the 'tau_unit' column if needed.")
    return filepath


def load_itis_csv(filepath: str) -> dict:
    """Load multiple tissues from a CSV filled with IT'IS Foundation parameters.

    Returns dict {tissue_name: FourColeColeParams}.
    """
    result = {}
    with open(filepath, "r") as fh:
        for row in csv.DictReader(fh):
            try:
                result[row["tissue_name"]] = build_tissue_params(
                    tissue_name=row["tissue_name"],
                    eps_inf=float(row["eps_inf"]), sigma_s=float(row["sigma_s"]),
                    delta_eps_1=float(row["delta_eps_1"]), tau_1=float(row["tau_1"]), alpha_1=float(row["alpha_1"]),
                    delta_eps_2=float(row["delta_eps_2"]), tau_2=float(row["tau_2"]), alpha_2=float(row["alpha_2"]),
                    delta_eps_3=float(row["delta_eps_3"]), tau_3=float(row["tau_3"]), alpha_3=float(row["alpha_3"]),
                    delta_eps_4=float(row["delta_eps_4"]), tau_4=float(row["tau_4"]), alpha_4=float(row["alpha_4"]),
                    tau_unit=row.get("tau_unit", "s") or "s",
                )
            except (ValueError, KeyError) as e:
                print(f"  [warning] row '{row.get('tissue_name', '?')}' incomplete or invalid -- skipping ({e})")

    if not result:
        raise ValueError(
            f"No valid tissue loaded from '{filepath}'. "
            "Check that all 14 numeric columns are filled in."
        )

    print(f"\n  {len(result)} tissue(s) loaded from '{filepath}':")
    for name in result:
        print(f"    - {name}")
    return result


def validate_physical_parameters(p: FourColeColeParams, verbose: bool = True) -> bool:
    """
    Check that the loaded parameters produce a physically valid spectrum
    (eps' monotonically decreasing, phase always negative). Detects the
    type of error found previously (positive phase from a transcription bug).
    """
    from .gabriel_4cc_model import conductivity_and_permittivity, impedance_from_4cc

    f_test = np.logspace(1, 9, 200)
    sigma, eps_real = conductivity_and_permittivity(f_test, p)
    phase = np.angle(impedance_from_4cc(f_test, p), deg=True)

    issues = []

    d_eps = np.diff(eps_real)
    n_increases = np.sum(d_eps > eps_real[:-1] * 0.01)  # 1% tolerance
    if n_increases > 2:  # small numerical fluctuations are fine
        issues.append(
            f"eps'(f) increases at {n_increases} points -- should be "
            "monotonically decreasing. Possible transcription error."
        )

    n_positive_phase = np.sum(phase > 0.5)  # numerical tolerance
    if n_positive_phase > 0:
        f_bad = f_test[phase > 0.5]
        issues.append(
            f"Positive phase detected at {n_positive_phase} points "
            f"(f~{f_bad.min():.0f}-{f_bad.max():.0f} Hz). This is "
            "physically impossible for this model -- check the signs "
            "and units of the transcribed parameters."
        )

    if np.any(sigma <= 0) or np.any(~np.isfinite(sigma)):
        issues.append("sigma(f) is negative, zero, or non-finite at some point.")

    if verbose:
        if issues:
            print(f"\n  ISSUES DETECTED in '{p.tissue_name}':")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"\n  '{p.tissue_name}' parameters passed basic physical validation.")

    return not issues
