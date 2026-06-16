"""
Pytest tests for lenshILC_utils.py.

Usage:
    # put this file next to lenshILC_utils.py
    pytest -q test_lenshILC_utils.py

The first test is an import/syntax smoke test. If it fails, fix the module first.
Known required fixes in the current pasted version:
    - compute_cinv_pol docstring indentation
    - define/import cmb_sed, Bnu, f_nu
    - define/import SimpleMixingOperator, or always pass Mx into CGComponentReconstructor.set_operator
"""

import importlib
import numpy as np
import pytest
import healpy as hp


MODULE_NAME = "lenshILC_utils"


class MockAlmInfo:
    """Minimal ainfo replacement for unit tests."""

    def __init__(self, lmax):
        self.lmax = int(lmax)
        self.nalm = hp.Alm.getsize(self.lmax)
        self.ell_of_alm, _ = hp.Alm.getlm(self.lmax)

    def alm2cl(self, x, y=None):
        x = np.asarray(x)
        if y is None:
            if x.ndim == 1:
                return hp.alm2cl(x, lmax=self.lmax)
            # Auto spectra for leading dimensions.
            lead = x.shape[:-1]
            out = np.empty(lead + (self.lmax + 1,), dtype=float)
            for idx in np.ndindex(lead):
                out[idx] = hp.alm2cl(x[idx], lmax=self.lmax)
            return out

        y = np.asarray(y)
        xb, yb = np.broadcast_arrays(x, y)
        lead = xb.shape[:-1]
        out = np.empty(lead + (self.lmax + 1,), dtype=np.result_type(xb, yb, float))
        for idx in np.ndindex(lead):
            out[idx] = hp.alm2cl(xb[idx], yb[idx], lmax=self.lmax)
        return out

    def lmul(self, x, mat_ell):
        """
        Apply ell-dependent matrix mat_ell[:, :, ell] to x[:, alm].
        x:       (p, nalm)
        mat_ell: (p, p, lmax+1)
        """
        x = np.asarray(x)
        mat_ell = np.asarray(mat_ell)
        out = np.zeros_like(x, dtype=np.result_type(x, mat_ell))
        for ell in range(self.lmax + 1):
            idx = self.ell_of_alm == ell
            if np.any(idx):
                out[:, idx] = mat_ell[:, :, ell] @ x[:, idx]
        return out


@pytest.fixture(scope="module")
def m():
    return importlib.import_module(MODULE_NAME)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def rand_alm(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def inner_alm(m, x, y):
    return m.contract_almxblm(x, y)


def test_module_imports():
    importlib.import_module(MODULE_NAME)


def test_contract_almxblm_matches_healpy_inner_product(m, rng):
    lmax = 5
    nalm = hp.Alm.getsize(lmax)
    a = rand_alm(rng, (2, nalm))
    b = rand_alm(rng, (2, nalm))

    got = m.contract_almxblm(a, b)

    ell, mm = hp.Alm.getlm(lmax)
    weights = np.where(mm == 0, 1.0, 2.0)
    expected = np.real(np.sum(weights[None, :] * a * np.conj(b)))
    assert np.allclose(got, expected)

    with pytest.raises(ValueError):
        m.contract_almxblm(a, b[:, :-1])


def test_compute_cinv_pol_shapes_and_inverse_property(m, rng):
    lmax = 6
    ainfo = MockAlmInfo(lmax)
    nfreq = 3
    dpol = rand_alm(rng, (nfreq, 2, ainfo.nalm))

    cl, clb, cinv = m.compute_cinv_pol(dpol, ainfo, bin_size=2, eps=1e-4)

    p = 2 * nfreq
    assert cl.shape == (p, p, lmax + 1)
    assert clb.shape == (p, p, lmax + 1)
    assert cinv.shape == (p, p, lmax + 1)

    for ell in range(lmax + 1):
        assert np.allclose(clb[:, :, ell], clb[:, :, ell].T.conj())
        ident = clb[:, :, ell] @ cinv[:, :, ell]
        assert np.allclose(ident, np.eye(p), atol=1e-6)

    with pytest.raises(ValueError):
        m.compute_cinv_pol(rand_alm(rng, (nfreq, 3, ainfo.nalm)), ainfo)


def test_apply_cinv_pol_matches_manual_ell_matrix(m, rng):
    lmax = 4
    ainfo = MockAlmInfo(lmax)
    nfreq = 2
    p = 2 * nfreq
    dpol = rand_alm(rng, (nfreq, 2, ainfo.nalm))

    cinv = np.zeros((p, p, lmax + 1), dtype=np.complex128)
    for ell in range(lmax + 1):
        cinv[:, :, ell] = np.eye(p) * (ell + 1.0)

    out = m.apply_cinv_pol(dpol, cinv, ainfo)

    dflat = dpol.reshape(p, ainfo.nalm)
    expected = np.zeros_like(dflat)
    for ell in range(lmax + 1):
        idx = ainfo.ell_of_alm == ell
        expected[:, idx] = (ell + 1.0) * dflat[:, idx]
    assert np.allclose(out.reshape(p, ainfo.nalm), expected)


def test_preconditioner_pol_shapes_and_action(m, rng):
    lmax = 5
    ainfo = MockAlmInfo(lmax)
    A = np.array([[1.0, 0.2], [0.8, 1.0], [1.3, -0.4]])
    nfreq, ncomp = A.shape
    p = 2 * nfreq

    cinv = np.zeros((p, p, lmax + 1), dtype=np.complex128)
    for ell in range(lmax + 1):
        cinv[:, :, ell] = np.eye(p) * (1.0 + 0.1 * ell)

    minv, precond = m.build_preconditioner_pol(A, cinv, ainfo)
    assert minv.shape == (2 * ncomp, 2 * ncomp, lmax + 1)

    x = rand_alm(rng, (ncomp, 2, ainfo.nalm))
    y = precond(x)
    assert y.shape == x.shape

    A_pol = np.kron(A, np.eye(2))
    xflat = x.reshape(2 * ncomp, ainfo.nalm)
    yflat = y.reshape(2 * ncomp, ainfo.nalm)
    for ell in range(lmax + 1):
        idx = ainfo.ell_of_alm == ell
        expected_mat = np.linalg.inv(A_pol.T @ cinv[:, :, ell] @ A_pol)
        assert np.allclose(yflat[:, idx], expected_mat @ xflat[:, idx])

    with pytest.raises(ValueError):
        m.build_preconditioner_pol(A, cinv[:-1, :, :], ainfo)
    with pytest.raises(ValueError):
        precond(rand_alm(rng, (ncomp + 1, 2, ainfo.nalm)))


def test_op_forward_and_adjoint(m):
    op = m.Op(lambda x: 2 * x, lambda y: 3 * y)
    assert op.forward(4) == 8
    assert op.adjoint(4) == 12


class IdentityPolOp:
    def forward(self, x):
        return x.copy()

    def adjoint(self, y):
        return y.copy()


class ScalePolOp:
    def __init__(self, scale):
        self.scale = scale

    def forward(self, x):
        return self.scale * x

    def adjoint(self, y):
        return np.conj(self.scale) * y


def test_Lmix_forward_adjoint_and_inner_product(m, rng):
    lmax = 5
    nalm = hp.Alm.getsize(lmax)
    A = np.array([[1.0, 2.0], [0.5, -1.0], [1.2, 0.3]])
    L = ScalePolOp(1.7 + 0.2j)
    Mx = m.Lmix(L=L, A=A, lens_components=[0])

    s = rand_alm(rng, (2, 2, nalm))
    d = rand_alm(rng, (3, 2, nalm))

    Ms = Mx.forward(s)
    Mtd = Mx.adjoint(d)
    assert Ms.shape == (3, 2, nalm)
    assert Mtd.shape == (2, 2, nalm)

    lhs = inner_alm(m, Ms, d)
    rhs = inner_alm(m, s, Mtd)
    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-10)

    with pytest.raises(ValueError):
        Mx.forward(rand_alm(rng, (3, 2, nalm)))
    with pytest.raises(ValueError):
        Mx.adjoint(rand_alm(rng, (4, 2, nalm)))


def test_make_normal_operator_self_adjoint_positive(m, rng):
    lmax = 5
    ainfo = MockAlmInfo(lmax)
    A = np.array([[1.0, 0.2], [0.5, -1.0], [1.2, 0.3]])
    Mx = m.Lmix(L=IdentityPolOp(), A=A, lens_components=[])

    nfreq = A.shape[0]
    p = 2 * nfreq
    cinv = np.zeros((p, p, lmax + 1), dtype=np.complex128)
    for ell in range(lmax + 1):
        cinv[:, :, ell] = np.eye(p) * (ell + 1.0)

    N = m.make_normal_operator(Mx, cinv, ainfo)
    x = rand_alm(rng, (A.shape[1], 2, ainfo.nalm))
    y = rand_alm(rng, (A.shape[1], 2, ainfo.nalm))

    Nx = N(x)
    Ny = N(y)
    assert Nx.shape == x.shape
    assert np.allclose(inner_alm(m, Nx, y), inner_alm(m, x, Ny), atol=1e-9)
    assert inner_alm(m, x, Nx) > 0


def test_lensing_operator_with_mocked_deflection(m, monkeypatch, rng):
    lmax = 3
    nalm = hp.Alm.getsize(lmax)
    EBlm = rand_alm(rng, (2, nalm))

    class FakeDefl:
        def gclm2lenmap(self, EBlm_in, *args, **kwargs):
            self.last_forward_input = EBlm_in.copy()
            # Return fake Q/U maps. hp.map2alm is monkeypatched below.
            return np.zeros((2, 12), dtype=float)

        def lensgclm(self, gclm, gclm_out, *args, **kwargs):
            gclm_out[...] = gclm

    def fake_map2alm(maps, lmax, pol):
        assert pol is True
        assert maps.shape[0] == 3
        return np.zeros(nalm), EBlm[0].copy(), EBlm[1].copy()

    monkeypatch.setattr(m.hp, "map2alm", fake_map2alm)

    L = m.LensingOperator(FakeDefl(), lmax_unl=lmax)
    assert L.pol is L
    assert L.forward(EBlm).shape == (2, nalm)
    assert np.allclose(L.forward(EBlm), EBlm)
    assert np.allclose(L.adjoint(EBlm), EBlm)

    with pytest.raises(ValueError):
        L.forward(EBlm[0])


def test_sed_functions_and_build_A_with_monkeypatched_physics(m, monkeypatch):
    # Patch missing external physics helpers so this tests this module's logic only.
    monkeypatch.setattr(m, "cmb_sed", lambda f: np.ones_like(np.asarray(f, dtype=float)), raising=False)
    monkeypatch.setattr(m, "Bnu", lambda nu, Td: np.asarray(nu, dtype=float) ** 2, raising=False)
    monkeypatch.setattr(m, "f_nu", lambda nu: np.ones_like(np.asarray(nu, dtype=float)), raising=False)

    freqs = np.array([23.0, 93.0, 353.0])

    dust = m.dust_sed(freqs, beta=1.5, Td=19.6, nu0=353.0)
    assert np.isclose(m.dust_sed(353.0, beta=1.5, Td=19.6, nu0=353.0), 1.0)
    assert np.allclose(m.dust_sed_beta1(freqs, beta=1.5, Td=19.6, nu0=353.0), dust * np.log(freqs / 353.0))

    sync = m.sync_sed(freqs, beta=-3.0, nu0=23.0)
    assert np.isclose(m.sync_sed(23.0, beta=-3.0, nu0=23.0), 1.0)
    assert np.allclose(m.sync_sed_beta1(freqs, beta=-3.0, nu0=23.0), sync * np.log(freqs / 23.0))

    A, names = m.build_A(A_d_BB=1, alpha_d_BB=0, beta_dust=1.5)
    assert A.shape == (8, 2)
    assert names == ("cmb", "dust")
    assert np.allclose(A[:, 0], 1.0)

    A2, names2 = m.build_A(
        A_d_BB=1,
        alpha_d_BB=0,
        beta_dust=1.5,
        amp_beta_dust=0.1,
        beta_sync=-3.0,
        amp_beta_sync=0.2,
    )
    assert A2.shape == (8, 5)
    assert names2 == ("cmb", "dust", "dust_beta1", "sync", "sync_beta1")


def test_bin_cls(m):
    cls = np.arange(10, dtype=float)
    bins = np.array([0, 3, 5, 10])
    got = m.bin_cls(cls, bins)
    assert np.allclose(got, [1.0, 3.5, 7.0])


def test_compress_to_data_vector(m, rng):
    lmax = 5
    ainfo = MockAlmInfo(lmax)
    s = rand_alm(rng, (2, 2, ainfo.nalm))
    bins = np.array([0, 2, 4, 6])

    data = m.compress_to_data_vector(s, ainfo, bins)
    assert data.shape == (2 * (len(bins) - 1),)

    expected_EE = m.bin_cls(ainfo.alm2cl(s[0, 0]), bins)
    expected_BB = m.bin_cls(ainfo.alm2cl(s[0, 1]), bins)
    assert np.allclose(data, np.concatenate([expected_EE, expected_BB]))


def test_CGComponentReconstructor_identity_mixing(m, monkeypatch, rng):
    """
    End-to-end test with identity mixing and identity Cinv.
    This avoids depending on SimpleMixingOperator and checks solve_components wiring.
    """
    lmax = 4
    ainfo = MockAlmInfo(lmax)
    A = np.eye(2)
    nsplit, nfreq, ncomp = 1, 2, 2
    d = rand_alm(rng, (nsplit, nfreq, 2, ainfo.nalm))

    Mx = m.Lmix(L=IdentityPolOp(), A=A, lens_components=[])

    def fake_compute_cinv_pol(dpol, ainfo, bin_size=20, eps=1e-6):
        p = 2 * dpol.shape[0]
        cinv = np.repeat(np.eye(p)[:, :, None], ainfo.lmax + 1, axis=2).astype(np.complex128)
        return cinv.copy(), cinv.copy(), cinv

    monkeypatch.setattr(m, "compute_cinv_pol", fake_compute_cinv_pol)

    rec = m.CGComponentReconstructor(cg_maxiter=20, cg_tol=1e-14)
    rec.set_operator(A, Mx=Mx)
    out = rec.solve_components(d, ainfo)

    assert out.shape == (nsplit, ncomp, 2, ainfo.nalm)
    assert np.allclose(out[0], d[0], atol=1e-8)
