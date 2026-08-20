import numpy as np
import healpy as hp
from pixell import utils, curvedsky
import lenspyx #Must for lensing operator construction.
from lenspyx import lensing
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl
from pathlib import Path #Import data files

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

def compute_cinv_pol(dpol, ainfo, bin_size=20):
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
    cl_pol = ainfo.alm2cl(xpol[:, None, :], xpol[None, :, :]) #Raw covariance dpol contract.

    clb_pol = np.zeros_like(cl_pol) #Binned covariance regulate.
    
    for b in np.unique(bidx):
        sel = (bidx == b)
        cbin = cl_pol[:, :, sel].mean(axis=2)
        clb_pol[:, :, sel] = cbin[:, :, None]

   # eye = np.eye(2 * nfreq, dtype=clb_pol.dtype)[:, :, None] XS: Shouldn't need; even tho suggested.
   # clb_pol = clb_pol + eps * eye
    cinv_pol = utils.eigpow(clb_pol, -1, axes=[0, 1]) #Inversed BINNED covariance.

    return cl_pol, clb_pol, cinv_pol


def apply_cinv_pol(dpol, cinv_pol, ainfo): #Apply C^-1 onto generated data d.
    nfreq, npol, nalm = dpol.shape
    dflat = dpol.reshape(2 * nfreq, nalm) # (nfreq, 2, nalm)
    out = ainfo.lmul(dflat, cinv_pol)     # (nfreq, 2, nalm)
    return out.reshape(nfreq, 2, nalm)


def build_preconditioner_pol(A, cinv_pol, ainfo): #Build M = (A^TC^-1A)
    """
    pol preconditioners: (full EB coupling).

    M = diag(M_E, M_B). M_E/B = A^T(Cinv_pol)A. Notice the full inversion of the covariance.

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

# ---------------------------
# Generate mixing matrix A
# ---------------------------

def build_A(
    freqs,
    fiducial_beta_dust,
    fiducial_T_dust,
    freq_pivot_dust,
    fiducial_beta_sync=None,
    freq_pivot_sync=None,
    default_T_dust=None,
    include_dust_beta1=False,
    include_sync_beta1=False,
):
    """
    Build the fiducial frequency mixing matrix used by reconstruction. 

    Parameters
    ----------
    freqs : array-like
        Observing frequencies in Hz. NOT GHz.

    fiducial_beta_dust : float
        Dust spectral index assumed by the reconstruction.

    fiducial_T_dust : float or None
        Dust temperature assumed by the reconstruction in K.
        If None, use fixed_params["temp_dust"].

    fiducial_beta_sync : float or None
        Synchrotron spectral index assumed by the reconstruction.
        If None, synchrotron is not included.

    include_dust_beta1 : bool
        Include the first-order dust beta-moment column.

    include_sync_beta1 : bool
        Include the first-order synchrotron beta-moment column.

    Returns
    -------
    A : ndarray
        Mixing matrix with shape (n_freq, n_comp).

    components : tuple[str]
        Component names corresponding to the columns of A.
    """

    freqs_hz = np.asarray(freqs, dtype=float)

    if freqs_hz.ndim != 1:
        raise ValueError("freqs must be one-dimensional.")

    if np.any(~np.isfinite(freqs_hz)) or np.any(freqs_hz <= 0):
        raise ValueError("All observing frequencies must be finite and positive.")

   # Fixed normalization conventions, stored internally in Hz
    freqs_ghz = freqs_hz / 1e9
    nu0_dust_ghz = float(freq_pivot_dust) / 1e9
    nu0_sync_ghz = fixed_params["freq_pivot_sync"] / 1e9

    if fiducial_T_dust is None:
        if default_T_dust is None:
            raise ValueError(
                "default_T_dust is required when fiducial_T_dust is None."
            )
        T_dust = float(default_T_dust)
    else:
        T_dust = float(fiducial_T_dust)
    
    beta_dust = float(fiducial_beta_dust)
    
    components = ["cmb", "dust"]
    
    columns = {
        "cmb": cmb_sed(freqs_ghz),
        "dust": dust_sed(
            freqs_ghz,
            beta=beta_dust,
            Td=T_dust,
            nu0=nu0_dust_ghz,
        ),
    }

    if include_dust_beta1:
        components.append("dust_beta1")
        columns["dust_beta1"] = dust_sed_beta1(
            freqs_ghz,
            beta=beta_dust,
            Td=T_dust,
            nu0=nu0_dust_ghz,
        )

    if fiducial_beta_sync is not None:
        if freq_pivot_sync is None:
            raise ValueError(
                "freq_pivot_sync is required when sync is included."
            )

        beta_sync = float(fiducial_beta_sync)
        nu0_sync_ghz = float(freq_pivot_sync) / 1e9

        components.append("sync")
        columns["sync"] = sync_sed(
            freqs_ghz,
            beta=beta_sync,
            nu0=nu0_sync_ghz,
        )

        if include_sync_beta1:
            components.append("sync_beta1")
            columns["sync_beta1"] = sync_sed_beta1(
                freqs_ghz,
                beta=beta_sync,
                nu0=nu0_sync_ghz,
            )
    
    elif include_sync_beta1:
        raise ValueError(
            "include_sync_beta1=True requires fiducial_beta_sync."
        )

    A = np.column_stack(
        [np.asarray(columns[name], dtype=float) for name in components]
    )

    expected_shape = (freqs_hz.size, len(components))

    if A.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape {A.shape}; expected {expected_shape}."
        )

    if not np.all(np.isfinite(A)):
        raise ValueError("Mixing matrix contains non-finite values.")

    return A

# ---------------------------
# SED functions
# ---------------------------

def cmb_sed(freq_ghz):
    """CMB SED in thermodynamic K_CMB units."""
    freq_ghz = np.asarray(freq_ghz, dtype=float)
    return np.ones_like(freq_ghz)


def Bnu(nu_hz, T):
    """Planck blackbody spectral radiance, apart from unit conventions."""
    h = 6.62607015e-34
    k = 1.380649e-23
    c = 299792458.0

    nu_hz = np.asarray(nu_hz, dtype=float)

    if np.any(nu_hz <= 0):
        raise ValueError("Frequency must be positive.")
    if T <= 0:
        raise ValueError("Temperature must be positive.")

    x = h * nu_hz / (k * T)

    return (
        2.0 * h * nu_hz**3 / c**2
        / np.expm1(x)
    )


def f_nu(nu_hz):
    """
    Frequency-dependent conversion factor from intensity-like units
    to thermodynamic CMB-temperature units, up to frequency-independent
    constants that cancel in normalized SED ratios.
    """
    h = 6.62607015e-34
    k = 1.380649e-23
    Tcmb = 2.7255

    nu_hz = np.asarray(nu_hz, dtype=float)

    if np.any(nu_hz <= 0):
        raise ValueError("Frequency must be positive.")

    x = h * nu_hz / (k * Tcmb)

    # Equivalent to (exp(x) - 1)^2 / (x^2 exp(x)),
    # but numerically somewhat clearer.
    return np.expm1(x)**2 / (x**2 * np.exp(x))


def mu_dust(freq_ghz, beta=1.5, Td=19.6):
    """
    Unnormalized dust SED.

    Input frequencies are in GHz. Frequencies passed to the Planck
    function and unit-conversion factor are converted to Hz.
    """
    freq_ghz = np.asarray(freq_ghz, dtype=float)
    nu_hz = freq_ghz * 1e9

    return (
        freq_ghz ** (beta - 2.0)
        * Bnu(nu_hz, Td)
        * f_nu(nu_hz)
    )


def dust_sed(freq_ghz, beta=1.5, Td=19.6, nu0=353.0):
    """
    Dust SED normalized to unity at nu0.
    """
    return (
        mu_dust(freq_ghz, beta=beta, Td=Td)
        / mu_dust(nu0, beta=beta, Td=Td)
    )


def dust_sed_beta1(freq_ghz, beta=1.5, Td=19.6, nu0=353.0):
    """
    First derivative of the normalized dust SED with respect to beta.
    """
    sed = dust_sed(
        freq_ghz,
        beta=beta,
        Td=Td,
        nu0=nu0,
    )

    freq_ghz = np.asarray(freq_ghz, dtype=float)

    return sed * np.log(freq_ghz / float(nu0))


# ---------------------------
# Synchrotron SED
# ---------------------------

def omega_sync(freq_ghz, beta=-3.0):
    """
    Unnormalized synchrotron SED.

    The power-law frequency can be expressed in GHz because the
    normalization ratio removes the associated constant. f_nu,
    however, must receive frequency in Hz.
    """
    freq_ghz = np.asarray(freq_ghz, dtype=float)
    nu_hz = freq_ghz * 1e9

    return (
        freq_ghz**beta
        * f_nu(nu_hz)
    )


def sync_sed(freq_ghz, beta=-3.0, nu0=23.0):
    """Synchrotron SED normalized to unity at nu0."""
    return (
        omega_sync(freq_ghz, beta=beta)
        / omega_sync(nu0, beta=beta)
    )


def sync_sed_beta1(freq_ghz, beta=-3.0, nu0=23.0):
    """
    First derivative of the normalized synchrotron SED with respect
    to beta.
    """
    sed = sync_sed(
        freq_ghz,
        beta=beta,
        nu0=nu0,
    )

    freq_ghz = np.asarray(freq_ghz, dtype=float)

    return sed * np.log(freq_ghz / float(nu0))

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

#Define a lensing operator L from lenspyx
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

#Define Lmix(A, L)
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
#Generate normal operator N = Lmix^dagger C^-1 Lmix 
# ---------------------------

def make_normal_operator(mixing_operator, cinv_pol, ainfo): 
    def normal_op(spol):
        dpol = mixing_operator.forward(spol)
        wd = apply_cinv_pol(dpol, cinv_pol, ainfo)
        return mixing_operator.adjoint(wd)
    return normal_op
    
# ---------------------------
#Generate deflection field from lensing file PP col. 
#Also a function to draw constraint phi from the file.
#Check if this file make sense.
# ---------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LENSPOTENTIAL_CLS = DATA_DIR / "planck_2018_lenspotentialCls.dat"

def generate_defl(lmax_len, nside, seed=None, dlmax=500, epsilon=1e-6):
    """Draw an unconstrained Gaussian phi realization and convert it to deflection."""

    lmax_phi = int(lmax_len + dlmax)

    if seed is not None:
        np.random.seed(int(seed))

    cl_phi = camb_clfile(str(LENSPOTENTIAL_CLS))["pp"]
    phi_alm = synalm(cl_phi, lmax=lmax_phi, mmax=lmax_phi)

    defl = phi_alm_to_defl(phi_alm, nside=nside, lmax=lmax_phi, epsilon=epsilon)

    return defl, phi_alm


def draw_constrained_phi(phi_obs_alm, S_l, N_l, lmax=None, seed=None):
    """Draw phi ~ P(phi | phi_obs) for phi_obs = phi + noise."""

    S_l = np.asarray(S_l, dtype=float)
    N_l = np.asarray(N_l, dtype=float)

    if lmax is None:
        lmax = min(len(S_l), len(N_l)) - 1

    S_l = S_l[:lmax + 1]
    N_l = N_l[:lmax + 1]

    invS = np.zeros_like(S_l)
    invN = np.zeros_like(N_l)
    sqrtInvS = np.zeros_like(S_l)
    sqrtInvN = np.zeros_like(N_l)

    np.divide(1.0, S_l, out=invS, where=S_l > 0)
    np.divide(1.0, N_l, out=invN, where=N_l > 0)
    np.divide(1.0, np.sqrt(S_l), out=sqrtInvS, where=S_l > 0)
    np.divide(1.0, np.sqrt(N_l), out=sqrtInvN, where=N_l > 0)

    post = np.zeros_like(S_l)
    den = invS + invN
    np.divide(1.0, den, out=post, where=den > 0)

    f_wiener = post * invN
    f_s = post * sqrtInvS
    f_n = post * sqrtInvN

    phi_wf_alm = hp.almxfl(phi_obs_alm, f_wiener, inplace=False)

    if seed is not None:
        np.random.seed(int(seed))

    zeta_s_alm = hp.synalm(np.ones(lmax + 1), lmax=lmax, new=True)
    zeta_n_alm = hp.synalm(np.ones(lmax + 1), lmax=lmax, new=True)

    residual_alm = hp.almxfl(zeta_s_alm, f_s, inplace=False) + hp.almxfl(zeta_n_alm, f_n, inplace=False)
    phi_cr_alm = phi_wf_alm + residual_alm

    return phi_cr_alm, phi_wf_alm, residual_alm


def phi_alm_to_defl(phi_alm, nside, lmax=None, epsilon=1e-6):
    """Convert phi alm to the gradient deflection field used by lenspyx."""

    if lmax is None:
        lmax = hp.Alm.getlmax(len(phi_alm))

    ell = np.arange(lmax + 1)
    dlm_grad = hp.almxfl(phi_alm, np.sqrt(ell * (ell + 1.0)), inplace=False)

    geom = lensing.get_geom(("healpix", {"nside": int(nside)}))
    defl = lenspyx.remapping.deflection(geom, dlm_grad, None, epsilon=epsilon)

    return defl
    

class CGComponentReconstructor:
    def __init__(self, bin_size=20, cg_maxiter=500, cg_tol=1e-12):
        self.bin_size = bin_size
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
            print(f"\nStarting CG split {s + 1}/{nsplit}", flush=True)
            
            dpol = d_alm_obs[s]

            _, _, cinv_pol = compute_cinv_pol(
                dpol, ainfo=ainfo, bin_size=self.bin_size)

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
            converged = False

            for iteration in range(1, self.cg_maxiter+1):
                cg.step()
                print(
                    f"split {s + 1}/{nsplit}, "
                    f"iteration {iteration:4d}, "
                    f"error = {cg.err:.6e}",
                    flush=True,
                )
                
                if cg.err < self.cg_tol:
                    converged = True
                    print(
                        f"CG split {s + 1}/{nsplit} converged "
                        f"after {iteration} iterations: "
                        f"error = {cg.err:.6e}",
                        flush=True,
                    )
                    break
                    
            if not converged:
                print(
                    f"CG split {s + 1}/{nsplit} did not converge "
                    f"after {self.cg_maxiter} iterations: "
                    f"final error = {cg.err:.6e}",
                    flush=True,
                )
            
            out[s] = cg.x

        return out
