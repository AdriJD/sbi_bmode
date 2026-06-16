import numpy as np
import healpy as hp
from pixell import utils
import lenspyx #Must for lensing operator construction.
from lenspyx import lensing
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl
import importlib.resources as ir

def contract_almxblm(alm, blm):
    """
    Handy function to properly contract. From Adri Duivenvoorden.
    """
    if blm.shape != alm.shape:
        raise ValueError(f"Shape alm {alm.shape} != shape blm {blm.shape}")

    lmax = hp.Alm.getlmax(alm.shape[-1])
    blm = np.conj(blm)
    csum = complex(np.tensordot(alm, blm, axes=alm.ndim))
    had_sum = 2.0 * np.real(csum)
    had_sum -= np.real(np.sum(alm[..., :lmax+1] * blm[..., :lmax+1]))
    return had_sum

def compute_cinv_pol(dpol, ainfo, bin_size=20, eps=1e-6):
    """
    Per-ell inverse covariance for pol=(E,B).
    Frequency correlatedpol covariance is FULL in stacked (E,B) space
    (i.e. do NOT force EB=0; invert the whole (2*nfreq)x(2*nfreq) block).

    dpol: (nfreq, 2, Nalm)  with stokes index = (E,B)

    Returns
    -------
    Cl_pol, Clb_pol, Cinv_pol: (2*nfreq, 2*nfreq, L), (2*nfreq, 2*nfreq, L), (2*nfreq, 2*nfreq, L)
    """
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
    dflat = dpol.reshape(2 * nfreq, nalm) # (nfreq, 2, nalm)
    out = ainfo.lmul(dflat, cinv_pol)     # (nfreq, 2, nalm)
    return out.reshape(nfreq, 2, nalm)


def build_preconditioner_pol(A, cinv_pol, ainfo):
    """
    pol preconditioners: (full EB coupling).

    Parameters
    ----------
    A        : (nfreq, ncomp) mixing matrix
    Cinv_pol : (2*nfreq, 2*nfreq, L)  (FULL in stacked [E;B] space)
    ainfo    : provides lmax

    Returns
    -------
    (Minv_pol, M_op_pol)
    """
    A = np.asarray(A)
    nfreq, ncomp = A.shape
    if cinv_pol.shape[0] != 2 * nfreq:
        raise ValueError("cinv_pol shape incompatible with A")

    L = cinv_pol.shape[-1]
    A_pol = np.kron(A, np.eye(2))
    minv_pol = np.zeros((2 * ncomp, 2 * ncomp, L), dtype=np.complex128)

    for ell in range(L):
        minv_pol[:, :, ell] = np.linalg.inv(A_pol.T @ cinv_pol[:, :, ell] @ A_pol)

    ell_of_alm, _ = hp.Alm.getlm(ainfo.lmax)

    def precond_op(spol): #The operator
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
    #N = Lmix^dagger C^-1 Lmix 
    def normal_op(spol):
        dpol = mixing_operator.forward(spol)
        wd = apply_cinv_pol(dpol, cinv_pol, ainfo)
        return mixing_operator.adjoint(wd)
    return normal_op

#Define a operator
class Op:
    """Generic linear operator with forward and adjoint."""
    def __init__(self, fwd, adj):
        self._fwd = fwd
        self._adj = adj

    def forward(self, x):
        return self._fwd(x)

    def adjoint(self, y):
        return self._adj(y)

#Define a lensing operator from lenspyx
class LensingOperator(Op):
    """
    Polarization-only lensing operator.

    forward : (2, Nalm) -> (2, Nalm)
    adjoint : (2, Nalm) -> (2, Nalm)
    Input order is [E, B].
    """

    def __init__(self, defl, lmax_unl):
        self.defl = defl
        self.lmax_unl = int(lmax_unl)

        def _as_pol(x, name):
            x = np.ascontiguousarray(np.atleast_2d(x), dtype=np.complex128)
            if x.ndim != 2 or x.shape[0] != 2:
                raise ValueError(f"{name} expects shape (2, Nalm), got {x.shape}")
            return x

        def fwd_pol(EBlm):
            EBlm = _as_pol(EBlm, "L.forward")
            
            omap_P = self.defl.gclm2lenmap(EBlm, None, 2, backwards=False, polrot=True,)
            # lenspyx may return only [Q, U]healpy.map2alm(pol=True)
            # expects [T, Q, U].
            if omap_P.shape[0] != 3:
                omap_P = np.vstack([np.zeros_like(omap_P[0]), omap_P])

            _, Elm_lensed, Blm_lensed = hp.map2alm(omap_P, lmax=self.lmax_unl, pol=True,)

            return np.vstack([Elm_lensed, Blm_lensed])

        def adj_pol(EBlm):
            EBlm = _as_pol(EBlm, "L.adjoint")
            
            out = np.zeros_like(EBlm, dtype=np.complex128)

            self.defl.lensgclm(gclm=EBlm, mmax=None, spin=2, lmax_out=self.lmax_unl, mmax_out=None,
                gclm_out=out, backwards=True, nomagn=False, polrot=True, out_sht_mode="STANDARD",)
            
            return out

        super().__init__(fwd=fwd_pol, adj=adj_pol)

        self.pol = self

class Lmix(Op):
    """
    Mixing + optional lensing operator, polarization only.

    Components:
        sPol : (ncomp, 2, Nalm)

    Frequency maps:
        dPol : (nfreq, 2, Nalm)

    Model:
        d_i = sum_c A[i, c] * L_c s_c

    where L_c is applied only if c in lens_components.
    """

    def __init__(self, L, A, lens_components=None):
        self.L = L
        self.A = np.asarray(A)

        if self.A.ndim != 2:
            raise ValueError(f"A expects shape (nfreq, ncomp), got {self.A.shape}")

        self.nfreq, self.ncomp = self.A.shape
        self.lens_components = set([0] if lens_components is None else lens_components)

        def _as_cplx(x):
            return np.ascontiguousarray(x, dtype=np.complex128)

        def fwd_pol(sPol):
            sPol = _as_cplx(sPol)

            if sPol.ndim != 3 or sPol.shape[1] != 2:
                raise ValueError("Lmix.forward expects shape (ncomp, 2, Nalm)")

            ncomp, _, nalm = sPol.shape
            if ncomp != self.ncomp:
                raise ValueError(
                    f"Lmix.forward expects ({self.ncomp}, 2, Nalm), got {sPol.shape}"
                )

            LsPol = np.empty((self.ncomp, 2, nalm), dtype=np.complex128)

            for c in range(self.ncomp):
                if c in self.lens_components:
                    LsPol[c] = self.L.forward(sPol[c])
                else:
                    LsPol[c] = sPol[c]

            return np.tensordot(self.A, LsPol, axes=(1, 0))

        def adj_pol(dPol):
            dPol = _as_cplx(dPol)

            if dPol.ndim != 3 or dPol.shape[1] != 2:
                raise ValueError("Lmix.adjoint expects shape (nfreq, 2, Nalm)")

            nfreq, _, nalm = dPol.shape
            if nfreq != self.nfreq:
                raise ValueError(
                    f"Lmix.adjoint expects ({self.nfreq}, 2, Nalm), got {dPol.shape}"
                )

            sPol = np.zeros((self.ncomp, 2, nalm), dtype=np.complex128)

            for i in range(self.nfreq):
                for c in range(self.ncomp):
                    contrib = np.conj(self.A[i, c]) * dPol[i]

                    if c in self.lens_components:
                        contrib = self.L.adjoint(contrib)

                    sPol[c] += contrib

            return sPol
        super().__init__(fwd=fwd_pol, adj=adj_pol)

# ---------------------------
#Generate mixing matrix.
# ---------------------------

def build_A(
    A_d_BB,
    alpha_d_BB,
    beta_dust,
    amp_beta_dust=None,
    gamma_beta_dust=None,
    A_s_BB=None,
    alpha_s_BB=None,
    beta_sync=None,
    amp_beta_sync=None,
    gamma_beta_sync=None
):
    """
    Build mixing matrix A from draw_data params.

    Returns
    -------
    A : ndarray
        Mixing matrix with Shape (n_freq, n_comp).
    """

    freqs_ghz = np.array([25., 27., 39., 93., 145., 225., 280., 350.]) #HardCode frequencies for now.
    beta_d = beta_dust
    beta_s = beta_sync if beta_sync is not None else -3.0
    Td = 19.6
    nu0_d = 353.0
    nu0_s = 23.0
        
    include = ["cmb", "dust"]

    if amp_beta_dust is not None:
        include.append("dust_beta1")

    if beta_sync is not None:
        include.append("sync")

        if amp_beta_sync is not None:
            include.append("sync_beta1")

    colmap = {
        "cmb": lambda f: cmb_sed(f),
        "dust": lambda f: dust_sed(f, beta=beta_d, Td=Td, nu0=nu0_d),
        "dust_beta1": lambda f: dust_sed_beta1(f, beta=beta_d, Td=Td, nu0=nu0_d),
        "sync": lambda f: sync_sed(f, beta=beta_s, nu0=nu0_s),
        "sync_beta1": lambda f: sync_sed_beta1(f, beta=beta_s, nu0=nu0_s),
    }

    Acols = [colmap[name](freqs_ghz) for name in include]
    A = np.vstack(Acols).T

    return A, tuple(include)

#Define sed functions:
# ---------------------------
#Dust SED + first moment wrt beta_d
# ---------------------------

def mu_dust(freq_ghz, beta=1.5, Td=19.6):
    nu = np.asarray(freq_ghz, float) * 1e9
    return (nu ** (beta - 2.0)) * Bnu(nu, Td) * f_nu(nu)

def dust_sed(freq_ghz, beta=1.5, Td=19.6, nu0=353.0):
    return mu_dust(freq_ghz, beta, Td) / mu_dust(nu0, beta, Td)

def dust_sed_beta1(freq_ghz, beta=1.5, Td=19.6, nu0=353.0):
    """
    First moment basis in frequency space:
      d/d beta [ dust_sed ].
    For normalized SED A(nu)=mu(nu)/mu(nu0):
      dA/dβ = A * ( d ln mu(nu)/dβ - d ln mu(nu0)/dβ )
    Here mu_dust ∝ nu^(β-2) * Bnu * f_nu, so only nu^(β-2) depends on β:
      d ln mu / dβ = ln(nu_hz)
    Thus:
      dA/dβ = A * ln(nu/nu0)
    (with nu and nu0 in the same units, GHz is fine)
    """
    A = dust_sed(freq_ghz, beta=beta, Td=Td, nu0=nu0)
    nu = np.asarray(freq_ghz, float)
    return A * np.log(nu / float(nu0))

# ---------------------------
#Sync SED + first moment wrt beta_s
# ---------------------------

def omega_sync(freq_ghz, beta=-3.0):
    nu = np.asarray(freq_ghz, float) * 1e9
    return (nu ** beta) * f_nu(nu)

def sync_sed(freq_ghz, beta=-3.0, nu0=23.0):
    return omega_sync(freq_ghz, beta) / omega_sync(nu0, beta)

def sync_sed_beta1(freq_ghz, beta=-3.0, nu0=23.0):
    """
    d/d beta [ sync_sed ].
    For normalized A(nu)=omega(nu)/omega(nu0) and omega ∝ nu^β * f_nu,
    only nu^β depends on β, so:
      dA/dβ = A * ln(nu/nu0)
    """
    A = sync_sed(freq_ghz, beta=beta, nu0=nu0)
    nu = np.asarray(freq_ghz, float)
    return A * np.log(nu / float(nu0))

# ---------------------------
#Generate deflection field from lensing field phi.
# ---------------------------

def generate_defl(lmax_len, nside, seed, , dlmax=500, epsilon=1e-6):
    if seed is not None:
        np.random.seed(int(seed))

    lmax_unl = int(lmax_len + dlmax)

    cls_path = ir.files("../data/") #Load it somewhere.
    cl_unl = camb_clfile(str(cls_path / "FFP10_wdipole_lenspotentialCls.dat"))
    plm = synalm(cl_unl["pp"], lmax=lmax_unl, mmax=lmax_unl)

    ell = np.arange(lmax_unl + 1)
    dlm_grad = almxfl(plm, np.sqrt(ell * (ell + 1.0)), None, False)

    geom = lensing.get_geom(("healpix", {"nside": int(nside)}))
    defl = lenspyx.remapping.deflection(geom, dlm_grad, None, epsilon=epsilon)
    return defl

class CGComponentReconstructor:
    def __init__(self, bin_size=20, eps=1e-6, cg_maxiter=500, cg_tol=1e-12):
        self.bin_size = bin_size
        self.eps = eps
        self.cg_maxiter = cg_maxiter
        self.cg_tol = cg_tol
        self.A = None
        self.Mx = None

    def set_operator(self, A, Mx):
        self.A = np.asarray(A, dtype=float)
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


#-----------Make CG output into binned Cl_EE and Cl__BB for training -------------------
def bin_cls(cls, bins):
    ells = np.arange(len(cls))
    out = np.zeros(len(bins) - 1, dtype=float)

    for i in range(len(out)):
        m = (ells >= bins[i]) & (ells < bins[i + 1])
        if np.any(m):
            out[i] = np.mean(cls[m])

    return out

def compress_to_data_vector(s_delensed_alm, ainfo, bins):
    """
    Output binned CMB EE and BB spectra.

    Parameters
    ----------
    s_delensed_alm : ndarray
        Shape (ncomp, 2, nalm), axis 1 is (E, B).
        Assumes component 0 is CMB.
    ainfo : object
        Has alm2cl method.
    bins : ndarray
        Bin edges.

    Returns
    -------
    data : ndarray
        Concatenated [Cl_EE_binned, Cl_BB_binned].
    """
    alm_E = s_delensed_alm[0, 0]
    alm_B = s_delensed_alm[0, 1]

    cl_EE = ainfo.alm2cl(alm_E)
    cl_BB = ainfo.alm2cl(alm_B)

    cl_EE_binned = bin_cls(cl_EE, bins)
    cl_BB_binned = bin_cls(cl_BB, bins)

    data = np.concatenate([cl_EE_binned, cl_BB_binned])

    return data
