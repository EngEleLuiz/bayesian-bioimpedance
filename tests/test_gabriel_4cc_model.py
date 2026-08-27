"""
tests/test_gabriel_4cc_model.py
================================
Regression tests for the 4-Cole-Cole model, guarding against the phase
sign bug in impedance_from_4cc that was fixed in commit d9b2d09
("fix critical sign bug in admittance calculation").

A sum of passive Cole-Cole dispersions with delta_eps>0, tau>0,
0<=alpha<1, sigma_s>=0 must always have phase <= 0 (Herglotz function
argument). A positive phase can only come from an invalid parameter or
a sign bug in the code -- never from "interference" between dispersions.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.gabriel_4cc_model import (
    FourColeColeParams, MUSCLE_4CC_PARTIAL,
    conductivity_and_permittivity, impedance_from_4cc, impedance_from_eps_sigma,
)

# A fully specified, physically valid parameter set (tau strictly
# decreasing, delta_eps>0, 0<=alpha<1, sigma_s>=0) used as a ground truth
# for the passivity checks below.
SAFE_PARAMS = FourColeColeParams(
    tissue_name="synthetic",
    eps_inf=4.0, sigma_s=0.2,
    delta_eps=[1.0e7, 3.0e3, 5.0e4, 10.0],
    tau=[1.0e-2, 1.0e-5, 1.0e-8, 1.0e-11],
    alpha=[0.1, 0.1, 0.1, 0.1],
    verified=True,
)


def _max_phase_deg(params, f_low=1.0, f_high=1e9, n=500):
    f = np.logspace(np.log10(f_low), np.log10(f_high), n)
    return np.angle(impedance_from_4cc(f, params), deg=True).max()


class TestPassivity:

    def test_synthetic_params_never_positive_phase(self):
        assert _max_phase_deg(SAFE_PARAMS) <= 1e-6

    def test_muscle_demo_params_flagged_unverified(self):
        # Regardless of the demo parameters' phase, they must never be
        # mistaken for validated ones downstream.
        assert MUSCLE_4CC_PARTIAL.verified is False

    def test_eps_sigma_impedance_matches_4cc_impedance(self):
        """impedance_from_4cc and impedance_from_eps_sigma must agree
        exactly for the same eps*/sigma pair. These are exactly the two
        implementations whose sign mismatch caused the original bug.
        """
        f = np.logspace(1, 8, 50)
        sigma, eps_real = conductivity_and_permittivity(f, SAFE_PARAMS)
        z_direct = impedance_from_4cc(f, SAFE_PARAMS)
        z_eps_sigma = impedance_from_eps_sigma(f, eps_real, sigma)
        np.testing.assert_allclose(z_direct, z_eps_sigma, rtol=1e-6)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
