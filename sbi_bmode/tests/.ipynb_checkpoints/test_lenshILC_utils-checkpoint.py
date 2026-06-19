# tests/test_lenshILC_utils.py

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import healpy as hp


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "lenshILC_utils.py"


@pytest.fixture(scope="session")
def m():
    spec = importlib.util.spec_from_file_location("lenshILC_utils", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeAinfo:
    def __init__(self, lmax):
        self.lmax = int(lmax)
        self.nelem = hp.Alm.getsize(self.lmax)
        self.ell, _ = hp.Alm.getlm(self.lmax)

    def alm2cl(self, a, b):
        """
        a: (nfield, 1, nalm)
        b: (1, nfield, nalm)
        return: (nfield, nfield, lmax+1)
        """
        nfield = a.shape[0]
        out = np.zeros((nfield, nfield, self.lmax + 1))

        for i in range(nfield):
            for j in range(nfield):
                out[i, j] = hp.alm2cl(a[i, 0], b[0, j], lmax=self.lmax)

        return out

    def lmul(self, x, fl):
        """
        x:  (nfield, nalm)
        fl: (nfield, nfield, lmax+1)
        """
        out = np.zeros_like(x, dtype=np.complex128)

        for ell in range(self.lmax + 1):
            idx = self.ell == ell
            out[:, idx] = fl[:, :, ell] @ x[:, idx]

        return out


class IdentityLensing:
    def forward(self, x):
        return x.copy()

    def adjoint(self, x):
        return x.copy()


def rand_alm(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_import(m):
    assert hasattr(m, "build_A")
    assert hasattr(m, "Lmix")
    assert hasattr(m, "compute_cinv_pol")
    assert hasattr(m, "CGComponentReconstructor")


def test_data_path_exists(m):
    assert m.LENSPOTENTIAL_CLS.exists(), m.LENSPOTENTIAL_CLS


def test_build_A_basic(m):
    A, names = m.build_A(
        A_d_BB=28.0,
        alpha_d_BB=-0.2,
        beta_dust=1.5,
    )

    assert names == ("cmb", "dust")
    assert A.shape == (8, 2)
    assert np.all(np.isfinite(A))
    assert np.allclose(A[:, 0], 1.0)


def test_build_A_with_extra_components(m):
    A, names = m.build_A(
        A_d_BB=28.0,
        alpha_d_BB=-0.2,
        beta_dust=1.5,
        amp_beta_dust=0.1,
        gamma_beta_dust=-0.1,
        A_s_BB=1.0,
        alpha_s_BB=-0.7,
        beta_sync=-3.0,
        amp_beta_sync=0.1,
        gamma_beta_sync=-0.1,
    )

    assert names == ("cmb", "dust", "dust_beta1", "sync", "sync_beta1")
    assert A.shape == (8, 5)
    assert np.all(np.isfinite(A))


def test_contract_almxblm(m):
    lmax = 8
    nalm = hp.Alm.getsize(lmax)

    a = rand_alm(nalm, seed=1)
    b = rand_alm(nalm, seed=2)

    val = m.contract_almxblm(a, b)

    assert np.isfinite(val)

    with pytest.raises(ValueError):
        m.contract_almxblm(a, b[:-1])


def test_compute_and_apply_cinv_pol(m):
    lmax = 8
    ainfo = FakeAinfo(lmax)
    nalm = ainfo.nelem

    nfreq = 3
    dpol = rand_alm((nfreq, 2, nalm), seed=3)

    cl, clb, cinv = m.compute_cinv_pol(
        dpol,
        ainfo=ainfo,
        bin_size=3,
        eps=1e-6,
    )

    assert cl.shape == (2 * nfreq, 2 * nfreq, lmax + 1)
    assert clb.shape == cl.shape
    assert cinv.shape == cl.shape
    assert np.all(np.isfinite(cinv))

    wd = m.apply_cinv_pol(dpol, cinv, ainfo)

    assert wd.shape == dpol.shape
    assert np.all(np.isfinite(wd))


def test_Lmix_forward_adjoint_shape(m):
    lmax = 8
    nalm = hp.Alm.getsize(lmax)

    nfreq = 4
    ncomp = 2

    A = np.arange(nfreq * ncomp, dtype=float).reshape(nfreq, ncomp) + 1.0
    L = IdentityLensing()

    Mx = m.Lmix(L=L, A=A, lens_components=[0])

    s = rand_alm((ncomp, 2, nalm), seed=4)
    d = Mx.forward(s)
    back = Mx.adjoint(d)

    assert d.shape == (nfreq, 2, nalm)
    assert back.shape == s.shape
    assert np.all(np.isfinite(d))
    assert np.all(np.isfinite(back))


def test_Lmix_adjointness_identity_lensing(m):
    lmax = 8
    nalm = hp.Alm.getsize(lmax)

    nfreq = 4
    ncomp = 2

    rng = np.random.default_rng(5)
    A = rng.normal(size=(nfreq, ncomp))

    Mx = m.Lmix(
        L=IdentityLensing(),
        A=A,
        lens_components=[0],
    )

    s = rand_alm((ncomp, 2, nalm), seed=6)
    d = rand_alm((nfreq, 2, nalm), seed=7)

    lhs = np.vdot(Mx.forward(s), d)
    rhs = np.vdot(s, Mx.adjoint(d))

    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_preconditioner_and_normal_operator(m):
    lmax = 8
    ainfo = FakeAinfo(lmax)
    nalm = ainfo.nelem

    nfreq = 4
    ncomp = 2

    rng = np.random.default_rng(8)
    A = rng.normal(size=(nfreq, ncomp))

    nfield = 2 * nfreq
    cinv = np.zeros((nfield, nfield, lmax + 1), dtype=np.complex128)

    for ell in range(lmax + 1):
        X = rng.normal(size=(nfield, nfield))
        cinv[:, :, ell] = X.T @ X + np.eye(nfield)

    _, precond = m.build_preconditioner_pol(A, cinv, ainfo)

    s = rand_alm((ncomp, 2, nalm), seed=9)
    ps = precond(s)

    assert ps.shape == s.shape
    assert np.all(np.isfinite(ps))

    Mx = m.Lmix(L=IdentityLensing(), A=A, lens_components=[])
    N = m.make_normal_operator(Mx, cinv, ainfo)

    Ns = N(s)

    assert Ns.shape == s.shape
    assert np.all(np.isfinite(Ns))


def test_CGComponentReconstructor_smoke(m):
    lmax = 8
    ainfo = FakeAinfo(lmax)
    nalm = ainfo.nelem

    nsplit = 1
    nfreq = 4
    ncomp = 2

    rng = np.random.default_rng(10)
    A = rng.normal(size=(nfreq, ncomp))

    Mx = m.Lmix(
        L=IdentityLensing(),
        A=A,
        lens_components=[],
    )

    d = rand_alm((nsplit, nfreq, 2, nalm), seed=11)

    recon = m.CGComponentReconstructor(
        bin_size=3,
        eps=1e-6,
        cg_maxiter=5,
        cg_tol=1e-6,
    )
    recon.set_operator(A, Mx)

    out = recon.solve_components(d, ainfo)

    assert out.shape == (nsplit, ncomp, 2, nalm)
    assert np.all(np.isfinite(out))


@pytest.mark.slow
def test_generate_defl_smoke(m):
    pytest.importorskip("lenspyx")

    defl = m.generate_defl(
        lmax_len=8,
        nside=8,
        seed=0,
        dlmax=4,
        epsilon=1e-6,
    )

    assert defl is not None