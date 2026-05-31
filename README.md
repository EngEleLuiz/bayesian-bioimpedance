# Bayesian Bioimpedance

**Bayesian Parameter Estimation and Tissue State Classification for Bioelectrical Impedance Spectroscopy**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen.svg)](#testing)

This repository provides a fully Bayesian pipeline for EIS analysis of biological tissue, implementing:

- **AM-MCMC** posterior inference for Cole-Cole model parameters with full uncertainty quantification
- **WAIC / PSIS-LOO** formal comparison of four equivalent circuit models
- **IS-Posterior classifier** for unsupervised tissue state classification (normal / oedema / ischaemia)
- **Calibration assessment** via reliability diagrams and Expected Calibration Error (ECE)
- **Real data validation** using three open-access public datasets

Associated paper submitted to *Biomedical Signal Processing and Control* (Elsevier).

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Pipeline overview](#pipeline-overview)
- [Public datasets](#public-datasets)
- [Using your own data](#using-your-own-data)
- [Results](#results)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [References](#references)
- [License](#license)

---

## Installation

```bash
git clone https://github.com/EngEleLuiz/bayesian-bioimpedance
cd bayesian-bioimpedance

python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate.bat     # Windows CMD
# venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
```

Verify:
```bash
python -c "import numpy, scipy, matplotlib; print('OK')"
```

---

## Quick start

```bash
# Smoke test — synthetic data, fast run (~3 min)
python main_ext24.py --rapido

# Real tissue data
python main_real_data.py --rapido

# List all available public datasets
python main_real_data.py --lista
```

---

## Pipeline overview

### Synthetic data pipeline

| Script | What it does | Typical runtime |
|---|---|---|
| `main.py` | Base MCMC + Cole-Cole on synthetic spectra | 10–15 min |
| `main_ext24.py` | Model comparison (WAIC/LOO) + tissue classification | 30–50 min |
| `main_ext24.py --snr-scan` | Full pipeline + SNR sensitivity study | 2–3 h |

### Real data pipeline

| Script | What it does |
|---|---|
| `main_real_data.py --lista` | List all available public datasets |
| `main_real_data.py --rapido` | Quick validation on Gabriel + Dipa datasets |
| `main_real_data.py --dataset <key>` | Single dataset |
| `main_real_data.py --csv <file>` | Your own CSV measurement |

### All command-line options

```
python main_ext24.py [options]

  --snr FLOAT            Measurement SNR in dB (default: 30.0)
  --tecido STR           Tissue: musculo | gordura | sangue (default: musculo)
  --n-freq INT           Number of frequency points (default: 30)
  --n-amostras INT       MCMC post-warmup samples (default: 4000)
  --n-warmup INT         MCMC warmup samples (default: 2000)
  --seed INT             Random seed (default: 42)
  --rapido               Fast mode for testing (~5 min)
  --so-modelos           Run Extension 2 only (model comparison)
  --so-classificacao     Run Extension 4 only (tissue classification)
  --snr-scan             Add SNR sensitivity sweep 15–40 dB
  --saida STR            Output folder (default: resultados_v3)
```

---

## Public datasets

All datasets are embedded as tabulated Cole-Cole parameters from peer-reviewed publications.
No download required — spectra are computed on-the-fly from published parameters.

| Dataset key | Source | Tissue | Type |
|---|---|---|---|
| `gabriel_muscle_longitudinal` | Gabriel et al. (1996) *Phys Med Biol* | Skeletal muscle (longitudinal) | Human/animal |
| `gabriel_muscle_transverse` | Gabriel et al. (1996) | Skeletal muscle (transverse) | Human/animal |
| `gabriel_liver` | Gabriel et al. (1996) | Liver | Human/animal |
| `gabriel_fat` | Gabriel et al. (1996) | Adipose tissue | Human/animal |
| `gabriel_blood` | Gabriel et al. (1996) | Whole blood | Human/animal |
| `gabriel_skin_wet` | Gabriel et al. (1996) | Wet skin | Human |
| `gabriel_brain_grey` | Gabriel et al. (1996) | Brain (grey matter) | Human |
| `dipa_chicken_muscle_37C` | Dipa et al. (2024) *J. Electr. Bioimpedance* | Chicken muscle at 37 °C | Ex vivo animal |
| `dipa_chicken_muscle_25C` | Dipa et al. (2024) | Chicken muscle at 25 °C | Ex vivo animal |
| `dipa_chicken_muscle_15C` | Dipa et al. (2024) | Chicken muscle at 15 °C | Ex vivo animal |
| `dipa_cow_liver_37C` | Dipa et al. (2024) | Bovine liver at 37 °C | Ex vivo animal |
| `dipa_fish_muscle_37C` | Dipa et al. (2024) | Fish muscle at 37 °C | Ex vivo animal |
| `fruit_potato_day1` | Catalan-Jorba et al. (2025) *Sensors* | Potato (fresh, day 1) | Vegetable |
| `fruit_potato_day5` | Catalan-Jorba et al. (2025) | Potato (day 5) | Vegetable |
| `fruit_potato_day9` | Catalan-Jorba et al. (2025) | Potato (deteriorated, day 9) | Vegetable |
| `fruit_apple_day1` | Catalan-Jorba et al. (2025) | Apple (fresh) | Vegetable/fruit |
| `fruit_banana_day1` | Catalan-Jorba et al. (2025) | Banana (fresh) | Vegetable/fruit |

```bash
# Load and inspect any dataset
python -c "
from src.real_data import DatasetRegistry
reg  = DatasetRegistry()
data = reg.load('gabriel_liver', snr_db=30)
print('f range:', data['f'].min(), '–', data['f'].max(), 'Hz')
print('R_inf:', data['R_inf'], 'Ω  |  R0:', data['R0'], 'Ω')
"
```

---

## Using your own data

### CSV format

Create a `.csv` file with columns `freq_hz`, `Re_Z`, `Im_Z`:

```csv
freq_hz,Re_Z,Im_Z
100,185.32,-2.14
200,184.87,-4.21
500,183.06,-9.76
1000,180.21,-17.32
2000,175.43,-28.11
5000,163.87,-41.25
10000,151.20,-48.73
20000,138.64,-49.86
50000,122.38,-42.11
100000,112.74,-32.18
```

> Semicolon-separated files are also supported. Column names are case-insensitive.

```bash
# Generate a template CSV for reference
python main_real_data.py --exemplo-csv
# → saves 'exemplo_medicao.csv' with 30 frequency points

# Run pipeline on your file
python main_real_data.py --csv your_measurement.csv --snr 25
```

### Python API

```python
from src.real_data    import DatasetRegistry
from src.ls_fitting   import nlls_fit
from src.mcmc_sampler import AdaptiveMCMC, log_posterior
from src.analysis     import samples_to_physical, compute_hdi
import numpy as np

# --- Load data ---
reg  = DatasetRegistry()
data = reg.load("gabriel_muscle_longitudinal", snr_db=30, seed=42)
# or: data = reg.load_csv("my_measurement.csv")
f, Z = data["f"], data["Z_noisy"]

# --- NLLS initialisation ---
ls = nlls_fit(f, Z, n_restarts=5)
p  = ls["params"]
theta0 = np.array([p["R_inf"], p["R0"] - p["R_inf"],
                   np.log(p["tau"]), p["alpha"]])

# --- AM-MCMC posterior ---
fn      = lambda th: log_posterior(th, f, Z, snr_db=30)
sampler = AdaptiveMCMC(n_samples=4000, n_warmup=2000, seed=42)
result  = sampler.run(fn, theta0, verbose=True)
samples = result["samples"]

# --- Posterior summaries ---
phys = samples_to_physical(samples)
for param in ["R_inf", "R0", "tau", "alpha", "f_c"]:
    lo, hi = compute_hdi(phys[param])
    print(f"{param}: median={np.median(phys[param]):.4g}  "
          f"HDI95=[{lo:.4g}, {hi:.4g}]")
```

---

## Results

### Model comparison (WAIC / PSIS-LOO)

| Model | WAIC | ΔWAIC | p_WAIC | Evidence against Cole-Cole |
|---|---|---|---|---|
| **Cole-Cole** ★ | 207.4 | 0 | 2.1 | — |
| Randles + CPE | 208.7 | +1.3 | 2.5 | Insignificant |
| Double Cole-Cole | 210.0 | +2.6 | 3.3 | Insignificant |
| **Debye (α = 1)** | 445.5 | **+238.1** | 16.4 | **Very strong** |

The Debye model is formally rejected (ΔWAIC/2 = 119 on the Jeffreys scale).
Double Cole-Cole provides no improvement despite 7 vs 4 parameters (p_WAIC = 3.3 ≪ 7).

### Tissue state classification (IS-Posterior, SNR = 30 dB)

|  | Normal | Oedema | Ischaemia | Recall | AUC-ROC |
|---|---|---|---|---|---|
| **Normal** | 22 | 1 | 7 | 73% | 0.83 |
| **Oedema** | 3 | 27 | 0 | 90% | 0.97 |
| **Ischaemia** | 6 | 0 | 24 | 80% | 0.87 |

Overall accuracy: **79%** (71/90) · Mean AUC: **0.89** · ECE: **0.11**  
No labelled training data required. ESS_IS ∈ [1400, 2000].

---

## Testing

```bash
# With pytest (if installed)
pip install pytest
pytest tests/ -v

# Without pytest
python tests/test_pipeline.py
```

38 tests covering: Cole-Cole model correctness, data generation, NLLS fitting,
MCMC diagnostics, WAIC/LOO computation, and real data loading.

---

## Project structure

```
bayesian-bioimpedance/
│
├── main.py                     # Base pipeline: MCMC + Cole-Cole
├── main_ext24.py               # Ext.2 (WAIC/LOO) + Ext.4 (classification)
├── main_real_data.py           # Validation on real public datasets
├── requirements.txt
├── setup.py
├── CITATION.cff
│
├── src/
│   ├── cole_model.py           # Cole-Cole impedance model
│   ├── data_generation.py      # Synthetic EIS spectrum generation
│   ├── ls_fitting.py           # NLLS reference fitting
│   ├── mcmc_sampler.py         # Adaptive Metropolis MCMC
│   ├── models.py               # Four ECMs: Debye, Cole-Cole, Double CC, Randles+CPE
│   ├── model_comparison.py     # WAIC and PSIS-LOO
│   ├── tissue_states.py        # Tissue state priors + v1 classifier
│   ├── classifier_v2.py        # IS-Posterior classifier (two-stage)
│   ├── real_data.py            # Public dataset loaders + CSV reader
│   ├── analysis.py             # Base pipeline figures
│   └── analysis_ext24.py       # Extension 2 & 4 figures
│
├── tests/
│   └── test_pipeline.py        # 38 automated tests
│
└── data/
    ├── raw/                    # Place your CSV files here
    └── cache/                  # Auto-generated cache
```

---

## Dependencies

```
numpy  >= 1.24
scipy  >= 1.10
matplotlib >= 3.7
```

No PyMC, TensorFlow, or other heavy dependencies. MCMC implemented from scratch.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bayesian_bioimpedance_2026,
  author  = {[Authors]},
  title   = {Bayesian Parameter Estimation and Tissue State Classification
             for Bioelectrical Impedance Spectroscopy:
             An Importance-Sampling Posterior Approach with Formal
             Equivalent Circuit Model Comparison},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Submitted}
}
```

Software citation (Zenodo DOI after release):
```bibtex
@software{bayesian_bioimpedance_code,
  author  = {[Authors]},
  title   = {bayesian-bioimpedance: Bayesian EIS pipeline},
  version = {1.0.0},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/[USERNAME]/bayesian-bioimpedance}
}
```

---

## References

1. Cole, K.S. & Cole, R.H. (1941). *J. Chem. Phys.*, 9(4), 341–351.
2. Gabriel, S. et al. (1996). *Phys. Med. Biol.*, 41, 2271–2293.
3. Haario, H. et al. (2001). An adaptive Metropolis algorithm. *Bernoulli*, 7(2), 223–242.
4. Vehtari, A. et al. (2017). Practical Bayesian model evaluation using LOO and WAIC. *Stat. Comput.*, 27, 1413–1432.
5. Vehtari, A. et al. (2024). Pareto smoothed importance sampling. *JMLR*, 25, 72.
6. Watanabe, S. (2010). Asymptotic equivalence of Bayes CV and WAIC. *JMLR*, 11, 3571–3594.
7. Huang, J. et al. (2021). Hierarchical Bayesian EIS inversion. *Electrochim. Acta*, 367, 137493.
8. Zhang, R. et al. (2024). Bayesian assessment of ECMs for corrosion EIS. *npj Mater. Degrad.*, 8, 111.
9. Dipa, S.A. et al. (2024). *J. Electr. Bioimpedance*, 15. DOI: 10.2478/joeb-2024-0013.
10. Catalan-Jorba, G. et al. (2025). *Sensors*, 25(3).
11. Guo, C. et al. (2017). On calibration of modern neural networks. *ICML 2017*, PMLR 70.
12. McDermott, C. et al. (2024). *Med. Biol. Eng. Comput.*, 62, 1177–1189.

---

## License

[MIT](LICENSE) © 2026 PPGEEL-UFSC
