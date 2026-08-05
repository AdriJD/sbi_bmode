import numpy as np
from scipy.sparse import load_npz
from toast.ops import ObsMat
from pathlib import Path

# In arcmin.
sat_beam_fwhms = {
    "f030": 91.0,
    "f040": 63.0,
    "f090": 30.0,
    "f150": 17.0,
    "f230": 11.0,
    f"f290": 9.0,
}

# In Hz.
sat_central_freqs = {
    "f030": 27e9,
    "f040": 39e9,
    "f090": 93e9,
    "f150": 145e9,
    "f230": 225e9,
    "f290": 280e9,
}

# In uK sqrt(s).
sat_noise_level = {
    "f030": {"baseline": 21, "goal": 15},
    "f040": {"baseline": 13, "goal": 10},
    "f090": {"baseline": 3.4, "goal": 2.4},
    "f150": {"baseline": 4.3, "goal": 2.7},
    "f230": {"baseline": 8.6, "goal": 5.7},
    "f290": {"baseline": 22, "goal": 14},
}

# From Table 1 in SO forecast paper (1808.07445). In uk arcmin.
sat_noise_level_forecast = {
    "f030": {"baseline": 35.0, "goal": 25.0},
    "f040": {"baseline": 21.0, "goal": 17.0},
    "f090": {"baseline": 2.6, "goal": 1.9},
    "f150": {"baseline": 3.3, "goal": 2.1},
    "f230": {"baseline": 6.3, "goal": 4.2},
    "f290": {"baseline": 16.0, "goal": 10.0},
}

# From Table 3 in Wolz et al (2302.04276). In uk armcin.
# These differ from sat_noise_level_forecast values by including a fudge
# factor that corrects for anisotropic noise. As a result they are very
# close to the noise levels corresponding to sat_noise_level and Fig 2
# in the SO forecast paper.
sat_noise_level_wolz = {
    "f030": {"baseline": 46.0, "goal": 33.0},
    "f040": {"baseline": 28.0, "goal": 22.0},
    "f090": {"baseline": 3.5, "goal": 2.5},
    "f150": {"baseline": 4.4, "goal": 2.8},
    "f230": {"baseline": 8.4, "goal": 5.5},
    "f290": {"baseline": 21.0, "goal": 14.0},
}

sat_lknee = {
    "f030": {"pessimistic": 30, "optimistic": 15},
    "f040": {"pessimistic": 30, "optimistic": 15},
    "f090": {"pessimistic": 50, "optimistic": 25},
    "f150": {"pessimistic": 50, "optimistic": 25},
    "f230": {"pessimistic": 70, "optimistic": 35},
    "f290": {"pessimistic": 100, "optimistic": 40},
}

sat_noise_alpha = {
    "f030": -2.4,
    "f040": -2.4,
    "f090": -2.5,
    "f150": -3.0,
    "f230": -3.0,
    "f290": -3.0,
}

OBSMAT_TEMPLATE = "sobs_{tag}_{sat}_mission_{flabel}_4way_coadd_sky_obsmat_healpix.npz"
FREQ_INST = {
    "f030": "SAT4",
    "f040": "SAT4",
    "f090": "SAT1",
    "f150": "SAT1",
    "f230": "SAT3",
    "f290": "SAT3",
}


def get_ntube(fstr, nyear_lf=1):
    """
    Return the "ntube" parameter for a given frequency band.

    Parameters
    ----------
    fstr : str
        Pick from 'f030', 'f040', 'f090', 'f150', 'f230' : 'f290'.
    nyear_lf : float
        Number of years where an lF is deployed.

    Returns
    -------
    ntube : float
        Effective number of tubes. Used to scale noise amplitudes.
    """

    if fstr in ("f230", "f290"):
        return 1.0
    elif fstr in ("f090", "f150"):
        return (2 - nyear_lf / 5) / 2
    else:
        return nyear_lf / 5


def get_sat_noise(fstr, sensitivity_mode, lknee_mode, lmax):
    """
    Generate noise curves in polarization for the SO small aperture telescopes

    Parameters
    ----------
    fstr : str
        Pick from 'f030', 'f040', 'f090', 'f150', 'f230' : 'f290'.
    sensitivity_mode : str
         Pick from "baseline" or "goal".
    lknee_mode : str
         Pick from "pessimistic" or "optimistic".
    lmax : int
        Maximum ell used for computation.

    Returns
    -------
    cov_noise_ell : (nell)
        Output noise spectra.
    """

    noise_level = sat_noise_level_wolz[fstr][sensitivity_mode]
    lknee = sat_lknee[fstr][lknee_mode]
    alpha = sat_noise_alpha[fstr]

    ells = np.arange(lmax + 1)
    n_ell = np.zeros(ells.size)

    n_ell[2:] = (ells[2:] / lknee) ** alpha
    n_ell[2:] += 1
    n_ell[2:] *= np.radians(noise_level / 60) ** 2

    return n_ell


def get_sat_noise_old(
    fstr,
    sensitivity_mode,
    lknee_mode,
    fsky,
    lmax,
    nyear_lf=1,
    include_kludge=True,
    nyear=5,
):
    """
    Generate noise curves in polarization for the SO small aperture telescopes

    Parameters
    ----------
    fstr : str
        Pick from 'f030', 'f040', 'f090', 'f150', 'f230' : 'f290'.
    sensitivity_mode : str
         Pick from "baseline" or "goal".
    lknee_mode : str
         Pick from "pessimistic" or "optimistic".
    fsky : float
        Fraction of sky, 0 <= fsky <= 1.
    lmax : int
        Maximum ell used for computation.
    nyear_lf : int
         Number of years where an LF is deployed on SAT.
    include_kludge : bool, optional
        Include a fudge factor that models the noise-uniformity at edge of map.
    nyear : float, optional
        Number of years of observations.

    Returns
    -------
    cov_noise_ell : (nell)
        Output noise spectra.
    """

    ntube = get_ntube(fstr, nyear_lf)
    noise_level = sat_noise_level[fstr][sensitivity_mode] / np.sqrt(ntube)
    lknee = sat_lknee[fstr][lknee_mode]
    alpha = sat_noise_alpha[fstr]

    tobs = nyear * 365 * 24 * 3600
    # Retention after observing efficiency and cuts + kludge for the noise non-uniformity
    # of the map edges
    tobs *= 0.2
    if include_kludge:
        tobs *= 0.85
    obs_area = 4 * np.pi * fsky

    tot_noise_level = noise_level / np.sqrt(tobs)

    ells = np.arange(lmax + 1)
    n_ell = np.zeros(ells.size)

    n_ell[2:] = (ells[2:] / lknee) ** alpha + 1.0
    n_ell[2:] *= (tot_noise_level * np.sqrt(2)) ** 2 * obs_area

    return n_ell


def load_obs_matrix(
    freqs: list[str], 
    obsmat_dir: Path, 
    tag="RC1.r01",
    release="mss2",
    ) -> dict:
    """
    Load observation matrices for multiple frequency bands.

    Parameters
    ----------
    freqs : list[str]
        List of frequency labels (e.g. "f030", "f090") to load.

    obsmat_dir : Path
        Directory containing the observation matrix files.

    tag : str, optional
        Simulation tag used in the observation matrix filename.
        Default is "RC1.r01".

    Returns
    -------
    obsmats : dict
        Dictionary mapping frequency labels to ObsMat objects.
        The keys are the frequency labels and the values are the
        corresponding observation matrices.
    """
    obsmat_dir = Path(obsmat_dir)
    obsmats = {}
    for freq in freqs:
        sat = FREQ_INST[freq]
        
        if release == "mss2":
            obsmat_file = obsmat_dir / OBSMAT_TEMPLATE.format(tag=tag, sat=sat, flabel=freq)
            print(f"[{freq} GHz] loading obsmat: {obsmat_file.name}")
            obsmats[freq] = ObsMat(obsmat_file)
        
        elif release == "mss3":
            north_file = (
                obsmat_dir /
                f"obsmat_{sat}_{freq}_01_QU.north.npz"
            )

            south_file = (
                obsmat_dir /
                f"obsmat_{sat}_{freq}_01_QU.south.npz"
            )

            print(f"[{freq}] loading MSS3 north:")
            print(f"        {north_file.name}")

            print(f"[{freq}] loading MSS3 south:")
            print(f"        {south_file.name}")


            north = ObsMat(north_file)
            south = ObsMat(south_file)

            obsmats[freq] = CombinedObsMat(
                north,
                south,
            )
    return obsmats

class CombinedObsMat:
    """
    Combined two ObsMat objects (e.g. MSS3 south + north)
    
    The combined ObsMat applied both matrices and concatenates the outputs.
    """
    def __init__(self, north, south):
        self.north = north
        self.south = south
    
    def apply(self, x):
        """
        Apply north and south observation matrices
        """
        y_north = self.north.apply(x)
        y_south = self.south.apply(x)
        
        return y_north + y_south
    