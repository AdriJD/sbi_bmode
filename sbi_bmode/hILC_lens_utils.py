import numpy as np
import healpy as hp
from pixell import utils


def contract_almxblm(alm, blm):
    if blm.shape != alm.shape:
        raise ValueError(f"Shape alm {alm.shape} != shape blm {blm.shape}")

    lmax = hp.Alm.getlmax(alm.shape[-1])
    blm = np.conj(blm)
    csum = complex(np.tensordot(alm, blm, axes=alm.ndim))
    had_sum = 2.0 * np.real(csum)
    had_sum -= np.real(np.sum(alm[..., :lmax+1] * blm[..., :lmax+1]))
    return had_sum


def bin_cls(cls, bins):
    ells = np.arange(len(cls))
    out = np.zeros(len(bins) - 1, dtype=float)
    for i in range(len(out)):
        m = (ells >= bins[i]) & (ells < bins[i + 1])
        if np.any(m):
            out[i] = np.mean(cls[m])
    return out


def compute_cinv_pol(dpol, ainfo, bin_size=20, eps=1e-6):
    nfreq, npol, nalm = dpol.shape
    if npol != 2:
        raise ValueError(f"dpol must have shape (nfreq, 2, nalm), got {dpol.shape}")

    lmax = ainfo.lmax
    L = lmax + 1
    ells = np.arange(L)
    bidx = ells // bin_size

    xpol = dpol.reshape(2 * nfreq, nalm)
    cl_pol = ainfo.alm2cl(xpol[:, None, :], xpol[None, :, :])

    clb_pol = np.zeros_like(cl_pol)
    for b in np.unique(bidx):
        sel = (bidx == b)
        cbin = cl_pol[:, :, sel].mean(axis=2)
        clb_pol[:, :, sel] = cbin[:, :, None]

    eye = np.eye(2 * nfreq, dtype=clb_pol.dtype)[:, :, None]
    clb_pol = clb_pol + eps * eye
    cinv_pol = utils.eigpow(clb_pol, -1, axes=[0, 1])

    return cl_pol, clb_pol, cinv_pol


def apply_cinv_pol(dpol, cinv_pol, ainfo):
    nfreq, npol, nalm = dpol.shape
    dflat = dpol.reshape(2 * nfreq, nalm)
    out = ainfo.lmul(dflat, cinv_pol)
    return out.reshape(nfreq, 2, nalm)


def build_preconditioner_pol(A, cinv_pol, ainfo):
    A = np.asarray(A)
    nfreq, ncomp = A.shape
    if cinv_pol.shape[0] != 2 * nfreq:
        raise ValueError("cinv_pol shape incompatible with A")

    L = cinv_pol.shape[-1]
    A_pol = np.kron(A, np.eye(2))
    minv_pol = np.zeros((2 * ncomp, 2 * ncomp, L), dtype=np.complex128)

    for ell in range(L):
        fisher = A_pol.T @ cinv_pol[:, :, ell] @ A_pol
        minv_pol[:, :, ell] = np.linalg.inv(fisher)

    ell_of_alm, _ = hp.Alm.getlm(ainfo.lmax)

    def precond_op(spol):
        ncomp_, npol, nalm = spol.shape
        if ncomp_ != ncomp or npol != 2:
            raise ValueError(f"spol must have shape ({ncomp}, 2, nalm), got {spol.shape}")

        sflat = spol.reshape(2 * ncomp, nalm)
        out = np.zeros_like(sflat)
        for ell in range(L):
            idx = (ell_of_alm == ell)
            if np.any(idx):
                out[:, idx] = minv_pol[:, :, ell] @ sflat[:, idx]
        return out.reshape(ncomp, 2, nalm)

    return minv_pol, precond_op


def make_normal_operator(mixing_operator, cinv_pol, ainfo):
    def normal_op(spol):
        dpol = mixing_operator.forward(spol)
        wd = apply_cinv_pol(dpol, cinv_pol, ainfo)
        return mixing_operator.adjoint(wd)
    return normal_op


class SimpleMixingOperator:
    def __init__(self, A):
        self.set_mixing_matrix(A)

    def set_mixing_matrix(self, A):
        A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise ValueError("A must have shape (nfreq, ncomp)")
        self.A = A

    def forward(self, x):
        return np.einsum("fc,cpl->fpl", self.A, x)

    def adjoint(self, d):
        return np.einsum("fc,fpl->cpl", self.A, d)


class CGComponentReconstructor:
    def __init__(self, bin_size=20, eps=1e-6, cg_maxiter=500, cg_tol=1e-12):
        self.bin_size = bin_size
        self.eps = eps
        self.cg_maxiter = cg_maxiter
        self.cg_tol = cg_tol
        self.A = None
        self.Mx = None

    def set_operator(self, A, Mx=None):
        self.A = np.asarray(A, dtype=float)
        if Mx is None:
            self.Mx = SimpleMixingOperator(self.A)
        else:
            self.Mx = Mx

    def solve_components(self, d_alm_obs, ainfo):
        if self.A is None or self.Mx is None:
            raise ValueError("Call set_operator(A, Mx) before solve_components")

        nsplit, nfreq, npol, nalm = d_alm_obs.shape
        ncomp = self.A.shape[1]
        out = np.zeros((nsplit, ncomp, 2, nalm), dtype=np.complex128)

        for s in range(nsplit):
            dpol = d_alm_obs[s]

            _, _, cinv_pol = compute_cinv_pol(
                dpol, ainfo=ainfo, bin_size=self.bin_size, eps=self.eps
            )

            rhs = self.Mx.adjoint(apply_cinv_pol(dpol, cinv_pol, ainfo))
            _, precond = build_preconditioner_pol(self.A, cinv_pol, ainfo)
            normal_op = make_normal_operator(self.Mx, cinv_pol, ainfo)

            x0 = np.zeros_like(rhs, dtype=rhs.dtype)
            cg = utils.CG(
                A=normal_op,
                b=rhs.copy(),
                x0=x0,
                M=precond,
                dot=contract_almxblm,
            )

            for _ in range(self.cg_maxiter):
                cg.step()
                if cg.err < self.cg_tol:
                    break

            out[s] = cg.x

        return out


def compress_to_data_vector(s_delensed_alm, ainfo, compression_config):
    bins = compression_config["bins"]
    component_pairs = compression_config["component_pairs"]
    pol_index = compression_config.get("pol_index", 1)

    nsplit = s_delensed_alm.shape[0]
    vecs = []

    for c1, c2 in component_pairs:
        cls = []
        for i in range(nsplit):
            for j in range(i + 1, nsplit):
                alm1 = s_delensed_alm[i, c1, pol_index]
                alm2 = s_delensed_alm[j, c2, pol_index]
                cl = ainfo.alm2cl(alm1, alm2=alm2)
                cls.append(cl)

        if len(cls) == 0:
            raise ValueError("Need nsplit >= 2 for cross-spectra compression")

        cl_mean = np.mean(np.asarray(cls), axis=0)
        vecs.append(bin_cls(cl_mean, bins))

    return np.concatenate(vecs, axis=0)
