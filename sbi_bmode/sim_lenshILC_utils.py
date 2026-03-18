import os

import healpy as hp
import numpy as np
from optweight import alm_c_utils, alm_utils, map_utils, sht
from pixell import curvedsky

from sbi_bmode import planck_utils, so_utils, spectra_utils, wmap_utils

from hILC_lens_utils import (
    CGComponentReconstructor,
    compress_to_data_vector,
)


opj = os.path.join


class CMBSimulator:
    def __init__(
        self,
        specdir,
        data_dict,
        fixed_params_dict,
        reconstructor,
        compression_config,
        apply_highpass_filter=True,
        mask_file=None,
        fg_template_files=None,
    ):
        if reconstructor is None:
            raise ValueError("reconstructor must be provided")
        if compression_config is None:
            raise ValueError("compression_config must be provided")

        self.lmax = data_dict["lmax"]
        self.lmin = data_dict["lmin"]
        self.nside = data_dict["nside"]
        self.nsplit = data_dict["nsplit"]
        self.delta_ell = data_dict["delta_ell"]
        self.bins = np.asarray(compression_config["bins"])

        self.cov_scalar_ell = spectra_utils.get_cmb_spectra(
            opj(specdir, "camb_lens_nobb.dat"), self.lmax
        )
        self.cov_tensor_ell = spectra_utils.get_cmb_spectra(
            opj(specdir, "camb_lens_r1.dat"), self.lmax
        )

        self.minfo = map_utils.MapInfo.map_info_healpix(self.nside)
        self.ainfo = curvedsky.alm_info(self.lmax)

        freq_strings = data_dict["freq_strings"]
        beam_fwhms = [self.get_beam_fwhms(fstr) for fstr in freq_strings]
        freq_strings_ordered = np.asarray(freq_strings)[np.argsort(np.asarray(beam_fwhms))][::-1]

        self.freq_strings = [str(fstr) for fstr in freq_strings_ordered]
        self.beam_fwhms = [self.get_beam_fwhms(fstr) for fstr in self.freq_strings]
        self.freqs = [self.get_freqs(fstr) for fstr in self.freq_strings]
        self.nfreq = len(self.freqs)

        self.sensitivity_mode = data_dict["sensitivity_mode"]
        self.lknee_mode = data_dict["lknee_mode"]
        self.b_ells = self.get_gaussian_beams(self.beam_fwhms, self.lmax)

        if apply_highpass_filter:
            self.highpass_filter = get_highpass_filter(
                self.lmin, self.lmax, data_dict["highpass_delta_ell"]
            )
        else:
            self.highpass_filter = None

        self.noise_cov_ell = np.ones((self.nfreq, 2, 2, self.lmax + 1))
        for fidx, fstr in enumerate(self.freq_strings):
            self.noise_cov_ell[fidx] = (
                np.eye(2)[:, :, np.newaxis] * self.get_noise_ps(fstr) * self.nsplit
            )

        self.freq_pivot_dust = fixed_params_dict["freq_pivot_dust"]
        self.freq_pivot_sync = fixed_params_dict.get("freq_pivot_sync")
        self.temp_dust = fixed_params_dict["temp_dust"]

        if self.freq_pivot_dust <= 1e9:
            raise ValueError("freq_pivot_dust must be in Hz")
        if self.freq_pivot_sync is not None and self.freq_pivot_sync <= 1e9:
            raise ValueError("freq_pivot_sync must be in Hz")

        self.reconstructor = reconstructor
        self.compression_config = compression_config

        if mask_file:
            self.mask = hp.read_map(mask_file).astype(np.float64)
        else:
            self.mask = None

        if fg_template_files is not None:
            self.fg_templates = {}
            for fstr in self.freq_strings:
                self.fg_templates[fstr] = hp.read_alm(
                    fg_template_files["dust"][fstr]
                ).astype(np.complex128)
                self.fg_templates[fstr] += hp.read_alm(
                    fg_template_files["sync"][fstr]
                ).astype(np.complex128)
        else:
            self.fg_templates = None

    @staticmethod
    def get_freqs(fstr):
        if fstr.startswith("f"):
            return so_utils.sat_central_freqs[fstr]
        if fstr.startswith("p"):
            return planck_utils.planck_central_freqs[fstr]
        if fstr.startswith("w"):
            return wmap_utils.wmap_central_freqs[fstr]
        raise ValueError(f"Unrecognized band {fstr}")

    @staticmethod
    def get_beam_fwhms(fstr):
        if fstr.startswith("f"):
            return so_utils.sat_beam_fwhms[fstr]
        if fstr.startswith("p"):
            return planck_utils.planck_beam_fwhms[fstr]
        if fstr.startswith("w"):
            return wmap_utils.wmap_beam_fwhms[fstr]
        raise ValueError(f"Unrecognized band {fstr}")

    def get_noise_ps(self, fstr):
        if fstr.startswith("f"):
            return so_utils.get_sat_noise(
                fstr, self.sensitivity_mode, self.lknee_mode, self.lmax
            )
        if fstr.startswith("p"):
            return planck_utils.get_planck_noise(fstr, self.lmax)
        if fstr.startswith("w"):
            return wmap_utils.get_wmap_noise(fstr, self.lmax)
        raise ValueError(f"Unrecognized band {fstr}")

    @staticmethod
    def get_gaussian_beams(fwhms, lmax):
        fwhms = np.atleast_1d(fwhms)
        out = np.zeros((len(fwhms), lmax + 1))
        for fidx, fwhm in enumerate(fwhms):
            out[fidx] = hp.gauss_beam(np.radians(fwhm / 60.0), lmax=lmax)
        return out

    def forward_model(self, theta, seed, draw_from_fg_template=False):
        params = _validate_theta(theta)
        rng = np.random.default_rng(None if seed == -1 else seed)

        if draw_from_fg_template:
            if self.fg_templates is None:
                raise ValueError(
                    "draw_from_fg_template=True but fg_template_files was not provided"
                )
            out_dict = gen_data_fg_template_alm(
                self.fg_templates,
                params["r_tensor"],
                params["A_lens"],
                self.freq_strings,
                rng,
                self.nsplit,
                self.noise_cov_ell,
                self.cov_scalar_ell,
                self.cov_tensor_ell,
                self.b_ells,
                self.minfo,
                self.ainfo,
                signal_filter=self.highpass_filter,
                no_cmb_ee=(self.mask is not None),
                mask=self.mask,
            )
        else:
            out_dict = gen_data_alm(
                params["A_d_BB"],
                params["alpha_d_BB"],
                params["beta_dust"],
                self.freq_pivot_dust,
                self.temp_dust,
                params["r_tensor"],
                params["A_lens"],
                self.freqs,
                rng,
                self.nsplit,
                self.noise_cov_ell,
                self.cov_scalar_ell,
                self.cov_tensor_ell,
                self.b_ells,
                self.minfo,
                self.ainfo,
                amp_beta_dust=params.get("amp_beta_dust"),
                gamma_beta_dust=params.get("gamma_beta_dust"),
                A_s_BB=params.get("A_s_BB"),
                alpha_s_BB=params.get("alpha_s_BB"),
                beta_sync=params.get("beta_sync"),
                freq_pivot_sync=self.freq_pivot_sync,
                amp_beta_sync=params.get("amp_beta_sync"),
                gamma_beta_sync=params.get("gamma_beta_sync"),
                rho_ds=params.get("rho_ds"),
                signal_filter=self.highpass_filter,
                no_cmb_ee=(self.mask is not None),
                mask=self.mask,
            )

        d_alm_obs = out_dict.pop("data")
        return d_alm_obs, out_dict

    def build_mixing_matrix_from_theta(self, theta, components=("cmb", "dust")):
        from sbi_bmode import spectra_utils

        freqs = np.asarray(self.freqs, dtype=float)
        A = np.zeros((len(freqs), len(components)), dtype=float)

        for cidx, comp in enumerate(components):
            if comp == "cmb":
                A[:, cidx] = 1.0

            elif comp == "dust":
                beta_dust = theta["beta_dust"]
                for fidx, freq in enumerate(freqs):
                    sed = spectra_utils.get_sed_dust(
                        freq, beta_dust, self.temp_dust, self.freq_pivot_dust
                    )
                    A[fidx, cidx] = np.sqrt(sed) * (
                        spectra_utils.get_g_fact(freq)
                        / spectra_utils.get_g_fact(self.freq_pivot_dust)
                    )

            elif comp == "sync":
                beta_sync = theta["beta_sync"]
                for fidx, freq in enumerate(freqs):
                    sed = spectra_utils.get_sed_sync(
                        freq, beta_sync, self.freq_pivot_sync
                    )
                    A[fidx, cidx] = np.sqrt(sed) * (
                        spectra_utils.get_g_fact(freq)
                        / spectra_utils.get_g_fact(self.freq_pivot_sync)
                    )

            else:
                raise ValueError(f"Unsupported component '{comp}'")

        return A



    def draw_data(self, theta, seed):
        d_alm_obs, extras = self.forward_model(theta, seed=seed)
        A = self.build_mixing_matrix_from_theta(theta, components=("cmb", "dust"))
        self.reconstructor.set_operator(A)

        s_delensed_alm = self.reconstructor.solve_components(
            d_alm_obs,
            ainfo=self.ainfo,
        )

        x = compress_to_data_vector(
            s_delensed_alm,
            ainfo=self.ainfo,
            compression_config=self.compression_config,
        )

        out = {"data": x}
        out.update(extras)
        return out

def _validate_theta(theta):
    required = ["r_tensor", "A_lens", "A_d_BB", "alpha_d_BB", "beta_dust"]
    missing = [k for k in required if k not in theta]
    if missing:
        raise KeyError(f"theta is missing required keys: {missing}")
    return dict(theta)


def get_delta_beta_cl(amp, gamma, lmax, ell_0=1, ell_cutoff=1):
    ells = np.arange(lmax + 1)
    cls = np.zeros_like(ells, dtype=np.float64)
    sel = ells >= ell_cutoff
    cls[sel] = amp * (ells[sel] / ell_0) ** gamma
    return cls


def get_beta_map(minfo, ainfo, beta0, amp, gamma, seed, ell_0=1, ell_cutoff=1):
    rng = np.random.default_rng(seed=seed)
    cls = get_delta_beta_cl(amp, gamma, ainfo.lmax, ell_0=ell_0, ell_cutoff=ell_cutoff)
    alm_beta = alm_utils.rand_alm(cls[np.newaxis, :], ainfo, rng, dtype=np.complex128)
    alm_beta[0, 0] += np.sqrt(4.0 * np.pi) * beta0

    beta_cl = ainfo.alm2cl(alm_beta[0])
    map_beta = np.zeros(minfo.npix)
    sht.alm2map(alm_beta, map_beta, ainfo, minfo, 0)
    return map_beta, beta_cl


def gen_data_fg_template_alm(
    fg_templates,
    r_tensor,
    A_lens,
    freq_strings,
    seed,
    nsplit,
    cov_noise_ell,
    cov_scalar_ell,
    cov_tensor_ell,
    b_ells,
    minfo,
    ainfo,
    signal_filter=None,
    no_cmb_ee=False,
    mask=None,
):
    nfreq = len(freq_strings)
    out = np.zeros((nsplit, nfreq, 2, ainfo.nelem), dtype=np.complex128)

    rng = np.random.default_rng(seed)
    rngs = rng.spawn(1 + nsplit)
    rng_cmb = rngs[0]
    rngs_noise = rngs[1:]

    cov_ell = spectra_utils.get_combined_cmb_spectrum(
        r_tensor, A_lens, cov_scalar_ell, cov_tensor_ell
    )
    cmb_alm = alm_utils.rand_alm(cov_ell, ainfo, rng_cmb, dtype=np.complex128)
    if no_cmb_ee:
        cmb_alm[0] = 0

    for fidx, fstr in enumerate(freq_strings):
        b_ell = b_ells[fidx].copy()
        if signal_filter is not None:
            b_ell *= signal_filter
        out[:, fidx] = _gen_data_per_freq_fg_template_alm(
            fstr,
            cov_noise_ell[fidx],
            cmb_alm,
            nsplit,
            rngs_noise,
            ainfo,
            b_ell,
            fg_templates,
        )

    if mask is not None:
        out = _apply_mask_in_map_space(out, minfo, ainfo, mask)

    return {"data": out}


def gen_data_alm(
    A_d_BB,
    alpha_d_BB,
    beta_dust,
    freq_pivot_dust,
    temp_dust,
    r_tensor,
    A_lens,
    freqs,
    seed,
    nsplit,
    cov_noise_ell,
    cov_scalar_ell,
    cov_tensor_ell,
    b_ells,
    minfo,
    ainfo,
    amp_beta_dust=None,
    gamma_beta_dust=None,
    A_s_BB=None,
    alpha_s_BB=None,
    beta_sync=None,
    freq_pivot_sync=None,
    amp_beta_sync=None,
    gamma_beta_sync=None,
    rho_ds=None,
    signal_filter=None,
    no_cmb_ee=False,
    mask=None,
):
    nfreq = len(freqs)
    out = np.zeros((nsplit, nfreq, 2, ainfo.nelem), dtype=np.complex128)

    rng = np.random.default_rng(seed)
    rngs = rng.spawn(3 + nsplit)
    rng_cmb = rngs[0]
    rng_dust = rngs[1]
    rng_beta = rngs[2]
    rngs_noise = rngs[3:]

    cov_ell = spectra_utils.get_combined_cmb_spectrum(
        r_tensor, A_lens, cov_scalar_ell, cov_tensor_ell
    )

    ncomp_fg = 2 if A_s_BB is not None else 1
    cov_fg_ell = np.zeros((ncomp_fg, ncomp_fg, ainfo.lmax + 1))
    cov_fg_ell[0, 0] = spectra_utils.get_ell_shape(ainfo.lmax, alpha_d_BB, ell_pivot=80)
    cov_fg_ell[0, 0] *= A_d_BB

    if A_s_BB is not None:
        cov_fg_ell[1, 1] = spectra_utils.get_ell_shape(ainfo.lmax, alpha_s_BB, ell_pivot=80)
        cov_fg_ell[1, 1] *= A_s_BB
        if rho_ds is not None:
            cov_fg_ell[0, 1] = rho_ds * np.sqrt(cov_fg_ell[0, 0] * cov_fg_ell[1, 1])
            cov_fg_ell[1, 0] = cov_fg_ell[0, 1]

    cmb_alm = alm_utils.rand_alm(cov_ell, ainfo, rng_cmb, dtype=np.complex128)
    if no_cmb_ee:
        cmb_alm[0] = 0
    fg_alm = alm_utils.rand_alm(cov_fg_ell, ainfo, rng_dust, dtype=np.complex128)

    gamma_dust_ell = None
    gamma_sync_ell = None

    if gamma_beta_dust is not None:
        if amp_beta_dust is None:
            raise ValueError("amp_beta_dust must be provided when gamma_beta_dust is set")

        alm_tmp = np.zeros((2, ainfo.nelem), dtype=np.complex128)
        alm_tmp[1] = fg_alm[0]
        dust_map = np.zeros((2, minfo.npix))
        sht.alm2map(alm_tmp, dust_map, ainfo, minfo, 2)

        beta_dust_map, gamma_dust_ell = get_beta_map(
            minfo, ainfo, beta_dust, amp_beta_dust, gamma_beta_dust, rng_beta
        )

        if A_s_BB is not None:
            if gamma_beta_sync is None or amp_beta_sync is None or beta_sync is None:
                raise ValueError(
                    "sync beta map parameters must be set when varying-beta sync is requested"
                )
            alm_tmp[1] = fg_alm[1]
            sync_map = np.zeros((2, minfo.npix))
            sht.alm2map(alm_tmp, sync_map, ainfo, minfo, 2)
            beta_sync_map, gamma_sync_ell = get_beta_map(
                minfo, ainfo, beta_sync, amp_beta_sync, gamma_beta_sync, rng_beta
            )
        else:
            sync_map = None
            beta_sync_map = None

        for fidx, freq in enumerate(freqs):
            b_ell = b_ells[fidx].copy()
            if signal_filter is not None:
                b_ell *= signal_filter
            out[:, fidx] = _gen_data_per_freq_gamma_alm(
                freq,
                cov_noise_ell[fidx],
                beta_dust_map,
                temp_dust,
                freq_pivot_dust,
                cmb_alm,
                dust_map,
                nsplit,
                rngs_noise,
                ainfo,
                minfo,
                b_ell,
                sync_map=sync_map,
                beta_sync=beta_sync_map,
                freq_pivot_sync=freq_pivot_sync,
            )
    else:
        for fidx, freq in enumerate(freqs):
            b_ell = b_ells[fidx].copy()
            if signal_filter is not None:
                b_ell *= signal_filter
            out[:, fidx] = _gen_data_per_freq_simple_alm(
                freq,
                cov_noise_ell[fidx],
                beta_dust,
                temp_dust,
                freq_pivot_dust,
                cmb_alm,
                fg_alm,
                nsplit,
                rngs_noise,
                ainfo,
                b_ell,
                beta_sync=beta_sync,
                freq_pivot_sync=freq_pivot_sync,
            )

    if mask is not None:
        out = _apply_mask_in_map_space(out, minfo, ainfo, mask)

    out_dict = {"data": out}
    if gamma_dust_ell is not None:
        out_dict["gamma_dust_ell"] = gamma_dust_ell
    if gamma_sync_ell is not None:
        out_dict["gamma_sync_ell"] = gamma_sync_ell
    return out_dict


def _gen_data_per_freq_simple_alm(
    freq,
    cov_noise_ell,
    beta_dust,
    temp_dust,
    freq_pivot_dust,
    cmb_alm,
    fg_alm,
    nsplit,
    rngs_noise,
    ainfo,
    b_ell,
    beta_sync=None,
    freq_pivot_sync=None,
):
    out = np.zeros((nsplit, 2, ainfo.nelem), dtype=np.complex128)

    dust_factor = np.sqrt(
        spectra_utils.get_sed_dust(freq, beta_dust, temp_dust, freq_pivot_dust)
    )
    dust_factor *= spectra_utils.get_g_fact(freq) / spectra_utils.get_g_fact(freq_pivot_dust)

    signal_alm = cmb_alm.copy()
    signal_alm[1] += fg_alm[0] * dust_factor

    if fg_alm.shape[0] == 2:
        if beta_sync is None or freq_pivot_sync is None:
            raise ValueError("beta_sync and freq_pivot_sync must be set when sync is included")
        sync_factor = np.sqrt(
            spectra_utils.get_sed_sync(freq, beta_sync, freq_pivot_sync)
        )
        sync_factor *= (
            spectra_utils.get_g_fact(freq) / spectra_utils.get_g_fact(freq_pivot_sync)
        )
        signal_alm[1] += fg_alm[1] * sync_factor

    signal_alm = alm_c_utils.lmul(signal_alm, b_ell, ainfo, inplace=False)

    for sidx in range(nsplit):
        noise_alm = alm_utils.rand_alm(
            cov_noise_ell, ainfo, rngs_noise[sidx], dtype=np.complex128
        )
        out[sidx] = np.asarray(signal_alm + noise_alm, dtype=np.complex128)

    return out


def _gen_data_per_freq_gamma_alm(
    freq,
    cov_noise_ell,
    beta_dust,
    temp_dust,
    freq_pivot_dust,
    cmb_alm,
    dust_map,
    nsplit,
    rngs_noise,
    ainfo,
    minfo,
    b_ell,
    sync_map=None,
    beta_sync=None,
    freq_pivot_sync=None,
):
    out = np.zeros((nsplit, 2, ainfo.nelem), dtype=np.complex128)

    sed_map = spectra_utils.get_sed_dust(freq, beta_dust, temp_dust, freq_pivot_dust)
    scaled_dust_map = dust_map * np.sqrt(sed_map)
    scaled_dust_map *= spectra_utils.get_g_fact(freq) / spectra_utils.get_g_fact(freq_pivot_dust)

    fg_map = scaled_dust_map
    if sync_map is not None:
        sed_sync_map = spectra_utils.get_sed_sync(freq, beta_sync, freq_pivot_sync)
        scaled_sync_map = sync_map * np.sqrt(sed_sync_map)
        scaled_sync_map *= (
            spectra_utils.get_g_fact(freq) / spectra_utils.get_g_fact(freq_pivot_sync)
        )
        fg_map += scaled_sync_map

    fg_alm = np.zeros_like(cmb_alm)
    sht.map2alm(fg_map, fg_alm, minfo, ainfo, 2)
    signal_alm = cmb_alm + fg_alm
    signal_alm = alm_c_utils.lmul(signal_alm, b_ell, ainfo, inplace=False)

    for sidx in range(nsplit):
        noise_alm = alm_utils.rand_alm(
            cov_noise_ell, ainfo, rngs_noise[sidx], dtype=np.complex128
        )
        out[sidx] = np.asarray(signal_alm + noise_alm, dtype=np.complex128)

    return out


def _gen_data_per_freq_fg_template_alm(
    fstr,
    cov_noise_ell,
    cmb_alm,
    nsplit,
    rngs_noise,
    ainfo,
    b_ell,
    fg_templates,
):
    out = np.zeros((nsplit, 2, ainfo.nelem), dtype=np.complex128)

    signal_alm = cmb_alm.copy()
    signal_alm[1] += fg_templates[fstr]
    signal_alm = alm_c_utils.lmul(signal_alm, b_ell, ainfo, inplace=False)

    for sidx in range(nsplit):
        noise_alm = alm_utils.rand_alm(
            cov_noise_ell, ainfo, rngs_noise[sidx], dtype=np.complex128
        )
        out[sidx] = np.asarray(signal_alm + noise_alm, dtype=np.complex128)

    return out


def _apply_mask_in_map_space(d_alm, minfo, ainfo, mask):
    out = np.zeros_like(d_alm)
    q_u_map = np.zeros((2, minfo.npix))
    for sidx in range(d_alm.shape[0]):
        for fidx in range(d_alm.shape[1]):
            sht.alm2map(d_alm[sidx, fidx], q_u_map, ainfo, minfo, 2)
            q_u_map *= mask
            sht.map2alm(q_u_map, out[sidx, fidx], minfo, ainfo, 2)
    return out


def get_highpass_filter(lmin, lmax, delta_ell):
    f_ell = np.ones(lmax + 1)
    if delta_ell <= 0:
        raise ValueError("delta_ell must be positive")
    if (lmin - delta_ell) < 0:
        raise ValueError("lmin - delta_ell must be non-negative")

    ells = np.arange(delta_ell)
    transition = 0.5 * (1.0 + np.cos(ells * np.pi / delta_ell))
    f_ell[lmin - delta_ell + 1 : lmin + 1] = transition[::-1]
    f_ell[: lmin - delta_ell + 1] = 0.0
    return f_ell
