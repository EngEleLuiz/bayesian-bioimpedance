"""
tests/test_pipeline.py
======================
Testes automatizados para validação do pipeline.
Execute: pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.cole_model       import (cole_cole_impedance, characteristic_frequency,
                                   validate_params, TISSUE_PARAMS)
from src.data_generation  import frequency_grid, generate_eis_data, snr_to_sigma
from src.ls_fitting       import nlls_fit
from src.mcmc_sampler     import compute_ess, compute_rhat, AdaptiveMCMC, log_posterior
from src.model_comparison import compute_waic, compute_loo_psis
from src.real_data        import DatasetRegistry


# ===========================================================================
# 1. Cole-Cole model
# ===========================================================================

class TestColeColaModel:

    def test_debye_limit(self):
        """alpha=1 deve reduzir para modelo Debye."""
        f    = np.array([1e3, 1e4, 1e5])
        # cole_cole_impedance(f, R_inf, R0, tau, alpha)
        # Z = R_inf + (R0-R_inf) / (1 + (jwτ)^α)
        Z_cc = cole_cole_impedance(f, 50, 150, 1e-5, alpha=1.0)
        omega = 2 * np.pi * f
        Z_debye = 50 + 100 / (1 + 1j * omega * 1e-5)  # ΔR = R0-R_inf = 100
        np.testing.assert_allclose(Z_cc, Z_debye, rtol=1e-10)

    def test_high_frequency_limit(self):
        """Em f→∞, Z deve tender a R_inf."""
        f = np.array([1e9, 1e10, 1e11])
        Z = cole_cole_impedance(f, 50, 150, 1e-5, alpha=0.8)
        assert np.allclose(Z.real, 50, atol=1.0), "Z(f→∞) deve → R_inf"

    def test_dc_limit(self):
        """Em f→0, Z deve tender a R0."""
        f = np.array([1e-3, 1e-4, 1e-5])
        Z = cole_cole_impedance(f, 50, 150, 1e-5, alpha=0.8)
        assert np.allclose(Z.real, 150, atol=5.0), "Z(f→0) deve → R0"

    def test_imaginary_part_negative(self):
        """Parte imaginária deve ser negativa (caráter capacitivo)."""
        f = frequency_grid(100, 1e6, 20)
        Z = cole_cole_impedance(f, 50, 150, 7.96e-6, alpha=0.75)
        assert np.all(Z.imag < 0), "Im[Z] deve ser negativo"

    def test_validate_params_raises(self):
        with pytest.raises(ValueError):
            validate_params(-1, 100, 1e-5, 0.8)   # R_inf negativo
        with pytest.raises(ValueError):
            validate_params(50, 40, 1e-5, 0.8)    # R0 < R_inf
        with pytest.raises(ValueError):
            validate_params(50, 100, 1e-5, 1.5)   # alpha > 1

    def test_characteristic_frequency(self):
        tau = 7.96e-6
        fc  = characteristic_frequency(tau, alpha=0.75)
        assert abs(fc - 1/(2*np.pi*tau)) < 1.0


# ===========================================================================
# 2. Data generation
# ===========================================================================

class TestDataGeneration:

    def test_snr_to_sigma(self):
        from src.data_generation import snr_to_sigma
        sigma_30 = snr_to_sigma(30)
        assert abs(sigma_30 - 10**(-30/20)) < 1e-10

    def test_noise_level(self):
        """Ruído relativo deve corresponder ao SNR especificado."""
        f    = frequency_grid(100, 1e6, 50)
        data = generate_eis_data(f, 50, 200, 7.96e-6, 0.75,
                                  snr_db=30, seed=42)
        Z_true  = data["Z_true"]
        Z_noisy = data["Z_noisy"]
        rel_err = np.abs(Z_noisy - Z_true) / np.abs(Z_true)
        mean_err = rel_err.mean()
        sigma_expected = snr_to_sigma(30)
        assert mean_err < 3 * sigma_expected, (
            f"Ruído médio {mean_err:.3f} > 3σ = {3*sigma_expected:.3f}")

    def test_frequency_grid(self):
        f = frequency_grid(100, 1e6, 30)
        assert len(f) == 30
        assert abs(f[0] - 100) < 1
        assert abs(f[-1] - 1e6) < 100


# ===========================================================================
# 3. NLLS fitting
# ===========================================================================

class TestNLLSFitting:

    def test_noiseless_recovery(self):
        """Sem ruído, NLLS deve recuperar os parâmetros exatos."""
        true = dict(R_inf=50., R0=200., tau=7.96e-6, alpha=0.75)
        f    = frequency_grid(100, 1e6, 30)
        Z    = cole_cole_impedance(f, true["R_inf"],
                                    true["R0"] - true["R_inf"],
                                    true["tau"], true["alpha"])
        result = nlls_fit(f, Z, method="complex", n_restarts=5)
        assert result.get("converged"), "NLLS deve convergir"
        p = result["params"]
        assert abs(p["R_inf"] - true["R_inf"]) < 2.0
        assert abs(p["R0"]    - true["R0"])    < 5.0
        assert abs(p["tau"]   - true["tau"])   < true["tau"] * 0.1
        assert abs(p["alpha"] - true["alpha"]) < 0.05

    def test_noisy_convergence(self):
        """Com ruído (SNR=30 dB), NLLS deve convergir."""
        f    = frequency_grid(100, 1e6, 30)
        data = generate_eis_data(f, 50, 200, 7.96e-6, 0.75,
                                  snr_db=30, seed=0)
        result = nlls_fit(f, data["Z_noisy"])
        assert result.get("converged"), "NLLS deve convergir com SNR=30dB"
        assert result["chi2"] < 10.0


# ===========================================================================
# 4. MCMC diagnostics
# ===========================================================================

class TestMCMCDiagnostics:

    def test_ess_iid(self):
        """Para amostras i.i.d., ESS deve ser próximo de N."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 5000)
        ess = compute_ess(samples)
        assert ess > 3000, f"ESS para i.i.d. deve ser alto, obtido {ess:.0f}"

    def test_ess_autocorrelated(self):
        """Série autocorrelacionada deve ter ESS << N."""
        ar = np.zeros(5000)
        ar[0] = 0.
        rng = np.random.default_rng(0)
        for i in range(1, 5000):
            ar[i] = 0.95 * ar[i-1] + rng.normal(0, 0.1)
        ess = compute_ess(ar)
        assert ess < 1000, f"AR(0.95) deve ter ESS baixo, obtido {ess:.0f}"

    def test_rhat_converged(self):
        """Duas cadeias idênticas devem ter R-hat = 1."""
        rng = np.random.default_rng(7)
        chain = rng.normal(0, 1, (500, 3))
        rhat  = compute_rhat([chain, chain + rng.normal(0, 0.01, chain.shape)])
        assert np.all(rhat < 1.05), f"R-hat de cadeias similares: {rhat}"

    def test_mcmc_simple(self):
        """MCMC em função Gaussiana deve recuperar média e variância."""
        def log_gauss(th):
            return -0.5 * np.sum(((th - np.array([2., 3.])) / np.array([0.5, 1.]))**2)

        sampler = AdaptiveMCMC(n_samples=3000, n_warmup=1000, seed=42)
        result  = sampler.run(log_gauss, np.array([0., 0.]), verbose=False)
        s = result["samples"]
        assert abs(s[:, 0].mean() - 2.0) < 0.1, "Média dim0 deve ≈ 2.0"
        assert abs(s[:, 1].mean() - 3.0) < 0.2, "Média dim1 deve ≈ 3.0"
        assert abs(s[:, 0].std()  - 0.5) < 0.1, "Std dim0 deve ≈ 0.5"


# ===========================================================================
# 5. WAIC / LOO
# ===========================================================================

class TestModelComparison:

    @pytest.fixture
    def sample_ll_matrix(self):
        """Matriz de log-verossimilhança de teste (S=200, n_data=60)."""
        rng = np.random.default_rng(99)
        return rng.normal(-2.0, 0.5, size=(200, 60))

    def test_waic_shape(self, sample_ll_matrix):
        result = compute_waic(sample_ll_matrix)
        assert "waic"    in result
        assert "lppd"    in result
        assert "p_waic"  in result
        assert "se_waic" in result
        assert np.isfinite(result["waic"])

    def test_waic_better_model(self, sample_ll_matrix):
        """Modelo com maior ll_mean deve ter menor WAIC."""
        ll_good = sample_ll_matrix - 0.1   # melhor log-lik
        ll_bad  = sample_ll_matrix - 5.0   # pior
        w_good  = compute_waic(ll_good)["waic"]
        w_bad   = compute_waic(ll_bad)["waic"]
        assert w_good < w_bad, "Modelo melhor deve ter menor WAIC"

    def test_loo_shape(self, sample_ll_matrix):
        result = compute_loo_psis(sample_ll_matrix)
        assert "loo"     in result
        assert "k_hats"  in result
        assert len(result["k_hats"]) == sample_ll_matrix.shape[1]
        assert np.isfinite(result["loo"])


# ===========================================================================
# 6. Real data loader
# ===========================================================================

class TestRealDataLoader:

    def test_list_datasets(self):
        reg  = DatasetRegistry()
        keys = reg.list_datasets(verbose=False)
        assert len(keys) >= 10, "Deve haver >= 10 datasets"
        assert "gabriel_muscle_longitudinal" in keys
        assert "dipa_chicken_muscle_37C"     in keys
        assert "fruit_potato_day1"            in keys

    def test_load_gabriel(self):
        reg  = DatasetRegistry()
        data = reg.load("gabriel_muscle_longitudinal", snr_db=30, seed=0)
        assert "f"       in data
        assert "Z"       in data
        assert "Z_noisy" in data
        assert len(data["f"]) == 30
        assert np.all(data["f"] > 0)
        assert np.all(np.isfinite(data["Z_noisy"]))

    def test_noise_added(self):
        reg    = DatasetRegistry()
        clean  = reg.load("gabriel_muscle_longitudinal")
        noisy  = reg.load("gabriel_muscle_longitudinal", snr_db=20, seed=1)
        assert not np.allclose(clean["Z"], noisy["Z_noisy"]), \
            "Com SNR especificado, Z_noisy != Z_clean"

    def test_load_csv(self, tmp_path):
        """Teste de carregamento de CSV do usuário."""
        import csv
        csvfile = tmp_path / "test.csv"
        with open(csvfile, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["freq_hz", "Re_Z", "Im_Z"])
            for f in [100, 1000, 10000, 100000]:
                writer.writerow([f, 100.0 + f/1000, -10.0 - f/10000])
        reg  = DatasetRegistry()
        data = reg.load_csv(str(csvfile))
        assert len(data["f"]) == 4
        assert np.all(np.isfinite(data["Z"]))

    def test_unknown_key_raises(self):
        reg = DatasetRegistry()
        with pytest.raises(KeyError):
            reg.load("chave_que_nao_existe")


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
