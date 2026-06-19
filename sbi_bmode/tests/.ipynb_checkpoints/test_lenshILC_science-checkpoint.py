# tests/test_lenshILC_science.py

import sys
from pathlib import Path

import numpy as np
from pixell import curvedsky


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lenshILC_utils as lu


class IdentityLensing:
    def forward(self, x):
        return x.copy()

    def adjoint(self, x):
        return x.copy()


def make_test_ainfo(lmax):
    return curvedsky.alm_info(lmax)


def rand_complex(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def inner(x, y):
    return np.vdot(x, y)


def scientific_A():
    A, names = lu.build_A(
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
    return A, names


def make_white_cinv(nfreq, lmax):
    nfield = 2 * nfreq
    cinv = np.zeros((nfield, nfield, lmax + 1), dtype=np.complex128)

    for ell in range(lmax + 1):
        cinv[:, :, ell] = np.eye(nfield)

    return cinv


def test_scientific_A_full_rank():
    A, names = scientific_A()

    assert names == ("cmb", "dust", "dust_beta1", "sync", "sync_beta1")
    assert A.shape == (8, 5)
    assert np.all(np.isfinite(A))

    rank = np.linalg.matrix_rank(A)
    cond = np.linalg.cond(A)

    assert rank == A.shape[1], (rank, names, A)
    assert np.isfinite(cond)
    assert cond < 1e8, (cond, names, A)


def test_scientific_A_columns_not_parallel():
    A, names = scientific_A()

    cols = A / np.linalg.norm(A, axis=0, keepdims=True)
    corr = np.abs(cols.T @ cols)
    np.fill_diagonal(corr, 0.0)

    assert np.max(corr) < 0.9999, (np.max(corr), names, corr)


def test_scientific_preconditioner_is_invertible():
    lmax = 12
    ainfo = make_test_ainfo(lmax)

    A, names = scientific_A()
    nfreq, ncomp = A.shape

    cinv = make_white_cinv(nfreq, lmax)

    minv_pol, precond = lu.build_preconditioner_pol(A, cinv, ainfo)

    assert minv_pol.shape == (2 * ncomp, 2 * ncomp, lmax + 1)
    assert np.all(np.isfinite(minv_pol))

    x = rand_complex((ncomp, 2, ainfo.nelem), seed=1)
    y = precond(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_scientific_Lmix_adjointness():
    lmax = 12
    ainfo = make_test_ainfo(lmax)

    A, names = scientific_A()
    nfreq, ncomp = A.shape

    M = lu.Lmix(
        L=IdentityLensing(),
        A=A,
        lens_components=[0],
    )

    s = rand_complex((ncomp, 2, ainfo.nelem), seed=2)
    d = rand_complex((nfreq, 2, ainfo.nelem), seed=3)

    lhs = inner(M.forward(s), d)
    rhs = inner(s, M.adjoint(d))

    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12), names


def test_scientific_normal_operator_is_hermitian():
    lmax = 12
    ainfo = make_test_ainfo(lmax)

    A, names = scientific_A()
    nfreq, ncomp = A.shape

    M = lu.Lmix(
        L=IdentityLensing(),
        A=A,
        lens_components=[],
    )

    cinv = make_white_cinv(nfreq, lmax)
    N = lu.make_normal_operator(M, cinv, ainfo)

    x = rand_complex((ncomp, 2, ainfo.nelem), seed=4)
    y = rand_complex((ncomp, 2, ainfo.nelem), seed=5)

    lhs = inner(x, N(y))
    rhs = inner(N(x), y)

    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-10), names


def test_scientific_normal_operator_is_positive():
    lmax = 12
    ainfo = make_test_ainfo(lmax)

    A, names = scientific_A()
    nfreq, ncomp = A.shape

    M = lu.Lmix(
        L=IdentityLensing(),
        A=A,
        lens_components=[],
    )

    cinv = make_white_cinv(nfreq, lmax)
    N = lu.make_normal_operator(M, cinv, ainfo)

    x = rand_complex((ncomp, 2, ainfo.nelem), seed=6)

    val = inner(x, N(x))

    assert abs(val.imag) < 1e-10
    assert val.real >= -1e-10, (val, names)


def test_scientific_CG_recovers_components_without_noise_or_lensing():
    lmax = 10
    ainfo = make_test_ainfo(lmax)

    A, names = scientific_A()
    nfreq, ncomp = A.shape

    M = lu.Lmix(
        L=IdentityLensing(),
        A=A,
        lens_components=[],
    )

    s_true = rand_complex((ncomp, 2, ainfo.nelem), seed=7)

    d = M.forward(s_true)
    d = d[None, ...]

    recon = lu.CGComponentReconstructor(
        bin_size=4,
        eps=1e-8,
        cg_maxiter=200,
        cg_tol=1e-12,
    )
    recon.set_operator(A, M)

    s_hat = recon.solve_components(d, ainfo)[0]

    rel_err = np.linalg.norm(s_hat - s_true) / np.linalg.norm(s_true)

    assert rel_err < 1e-8, (rel_err, names)


def test_scientific_CG_recovers_CMB_without_noise_or_lensing():
    lmax = 10
    ainfo = make_test_ainfo(lmax)

    A, names = scientific_A()
    nfreq, ncomp = A.shape
    cmb_index = names.index("cmb")

    M = lu.Lmix(
        L=IdentityLensing(),
        A=A,
        lens_components=[],
    )

    s_true = rand_complex((ncomp, 2, ainfo.nelem), seed=8)

    d = M.forward(s_true)
    d = d[None, ...]

    recon = lu.CGComponentReconstructor(
        bin_size=4,
        eps=1e-8,
        cg_maxiter=200,
        cg_tol=1e-12,
    )
    recon.set_operator(A, M)

    s_hat = recon.solve_components(d, ainfo)[0]

    cmb_rel_err = (
        np.linalg.norm(s_hat[cmb_index] - s_true[cmb_index])
        / np.linalg.norm(s_true[cmb_index])
    )

    assert cmb_rel_err < 1e-8, (cmb_rel_err, names)