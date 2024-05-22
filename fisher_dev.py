import numpy as np
import fgbuster.component_model as fgc
import fgcls as fgl


class Parameters(object):
    def __init__(self):
        self.p0 = [1., 0., 1.59, 0., -0.2, 5., -3., -0.4, 2.]
        self.p_fixed = [('nu0_d', 353.0), ('temp_d', 19.6), ('l0_d_bb', 80.0), ('nu0_s', 23.0), ('l0_s_bb', 80.0)]
        self.p_free_names = ['A_lens', 'r_tensor', 'beta_d', 'epsilon_ds', 'alpha_d_bb', 'amp_d_bb', 'beta_s', 'alpha_s_bb', 'amp_s_bb']
        self.p_free_priors = [['A_lens', 'tophat', [0.0, 1.0, 2.0]],
                              ['r_tensor', 'tophat', [-0.1, 0.0, 0.1]],
                              ['beta_d', 'Gaussian', [1.59, 0.11]],
                              ['component_2', 'tophat', [-1.0, 0.0, 1.0]],
                              ['alpha', 'tophat', [-1.0, -0.2, 0.0]],
                              ['amp', 'tophat', [0.0, 5.0, 10.0]],
                              ['beta_pl', 'Gaussian', [-3.0, 0.3]],
                              ['alpha', 'tophat', [-1.0, -0.4, 0.0]],
                              ['amp', 'tophat', [0.0, 2.0, 4.0]]]

    def lnprior(self, par):
        lnp = 0
        for p, pr in zip(par, self.p_free_priors):
            if np.char.lower(pr[1]) == 'gaussian':  # Gaussian prior
                lnp += -0.5 * ((p - pr[2][0])/pr[2][1])**2
            else:  # Only other option is top-hat
                if not(float(pr[2][0]) <= p <= float(pr[2][2])):
                    return -np.inf
            ## TODO: implement half gaussian ++
        return lnp

class Bpass(object):
    def __init__(self,f):
        self.nu = np.array([f-1, f, f+1])
        self.bnu = np.array([0., 1., 0.])
        self.dnu = np.zeros_like(self.nu)
        try:
            self.dnu[:, 1:] = np.diff(self.nu)[0]
            self.dnu[:, 0] = self.dnu[:,1]
        except IndexError:
            self.dnu[1:] = np.diff(self.nu)
            self.dnu[0] = self.dnu[1]
        # CMB units
        norm = np.sum((self.dnu.T*self.bnu).T*self.nu**2*self.fcmb(self.nu))
        self.bnu /= norm
        self.bpss = self.bps(f) 
        
    def bps(self, f):
        try:
            self.bpss = []
            for i_f, ff in enumerate(f):
                self.bpss.append(self.bnu)
        except TypeError:
            self.bpss = self.bnu
        return self.bpss
        
    def convolve_sed(self,f):
        sed = np.sum(self.dnu*self.bnu*self.nu**2*f(self.nu))
        return sed

    def fcmb(self, nu):
        x = 0.017608676067552197*nu
        ex = np.exp(x)
        return ex*(x/(ex-1))**2

    
def rotate_cells_mat(mat1, mat2, cls):
    if mat1 is not None:
        cls = np.einsum('ijk,lk', cls, mat1)
    if mat2 is not None:
        cls = np.einsum('jk,ikl', mat2, cls)
    return cls


def rotate_cells(bp1, bp2, cls, params):
    m1 = bp1.get_rotation_matrix(params)
    m2 = bp2.get_rotation_matrix(params)
    return rotate_cells_mat(m1, m2, cls)


class Foregrounds():
    def __init__(self):
        self.n_components = 1 # TODO: currently only dust
        self.component_names = ['dust']
        self.components = {self.component_names[0]:
                            {'decorr': False,
                            'names_x_dict': {'component_1': 'epsilon_ds'},
                            'sed_parameters': {'beta_d': ['beta_d', 'Gaussian', [1.59, 0.11]],
                                               'temp_d': ['temp', 'fixed', [19.6]],
                                               'nu0_d': ['nu0', 'fixed', [353.0]]},
                            'names_sed_dict': {'beta_d': 'beta_d', 'temp': 'temp_d', 'nu0': 'nu0_d'},
                             'cmb_n0_norm': fgc.CMB('K_RJ').eval(353.0),
                             'names_cl_dict': {'BB':
                                               {'amp': 'amp_d_bb', 'alpha': 'alpha_d_bb', 'ell0': 'l0_d_bb'}}
        }}

        self.params_fgc = {'nu0': 353.0, 'beta_d': 1.59, 'temp': 19.6}
        sed_fnc = self.get_function(fgc, 'Dust') 
        self.components['dust']['sed'] = sed_fnc(**self.params_fgc, units='K_RJ')
        cl_fnc = self.get_function(fgl, 'ClPowerLaw')
        self.params_fgl = {'alpha': -0.2, 'amp': 5.0,
                           'ell0': 80}
        self.components['dust']['cl'] = cl_fnc(**self.params_fgl)
        
    def get_function(self, mod, sed_name):
        try:
            return getattr(mod, sed_name)
        except AttributeError:
            raise KeyError("Function named %s cannot be found" % (sed_name))

        
class FisherForecast:
    """Class to perform Fisher forecast"""

    def __init__(self, cls_file):
        self.cls_file = cls_file
        self.freqs = np.array([27., 39., 93., 145., 225., 280.])
        self.nfreqs = len(self.freqs)
        self.npol = 2
        self.nmaps = self.npol * self.nfreqs
        self.ncross = (self.nmaps * (self.nmaps + 1)) // 2
        self.nsplits = 1  # TODO: more splits?
        self.nside = 256 # TODO: specify nside
        self.fsky = 1 # TODO: masked sky
        self.dell = 10
        nbands = 77
        self.lmax = 3 * self.nside - 1
        self.bpw_l = np.arange(self.lmax+2)[2:]
        self.dl2cl = 2 * np.pi / (self.bpw_l * (self.bpw_l + 1))
        self.n_ell = len(self.bpw_l)
        self.params = Parameters()
        self.load_cmb()
        self.fg_model = Foregrounds()
        self.bpss = np.array([Bpass(f) for f in self.freqs])

        
    def read_cls(self):
        """Read input Cls"""
        nfreq, npol, _, _, nell = np.load(self.cls_file).shape 
        cls = np.load(self.cls_file).reshape(nfreq * npol, nfreq * npol, nell)
        return cls, self.nmaps, nell

    
    def calculate_covariance(self, cls, nmaps, nell):
        """Calculate covariance (Knox formula)"""
        # TODO: splits
        indices_tr = np.triu_indices(nmaps)
        cov = np.zeros([self.ncross, nell, self.ncross, nell])
        lbands = np.linspace(2, self.lmax, nell + 1, dtype=int)
        leff = 0.5 * (lbands[1:] + lbands[:-1])
        factor_modecount = 1. / ((2 * leff + 1) * self.dell * self.fsky)

        for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
            for jj, (j1, j2) in enumerate(zip(indices_tr[0], indices_tr[1])):
                covar = (cls[i1, j1, :] * cls[i2, j2, :] +
                         cls[i1, j2, :] * cls[i2, j1, :]) * factor_modecount
                cov[ii, :, jj, :] = np.diag(covar)
        
        return cov.reshape([self.ncross * nell, self.ncross * nell])

    
    def compute_fisher_matrix__old(self, covariance_matrix, parameters):
        """Compute Fisher forecast for parameters given the covariance matrix."""
        fisher_matrix = np.zeros((len(parameters), len(parameters)))

        for i, param_i in enumerate(parameters):
            for j, param_j in enumerate(parameters):
                grad_cov_i = np.gradient(np.gradient(covariance_matrix, param_i, axis=0), param_j, axis=1)
                fisher_matrix[i, j] = 0.5 * np.trace(np.dot(np.linalg.pinv(covariance_matrix), 
                                                            np.dot(grad_cov_i, np.linalg.pinv(covariance_matrix))))

        return fisher_matrix

    
    def lnprob(self, par):
        """
        Likelihood with priors.
        """
        prior = self.params.lnprior(par)
        if not np.isfinite(prior):
            return -np.inf

        return prior + self.lnlike(par)

    
    def build_params(self, par):
        params = dict(self.params.p_fixed)
        params.update(dict(zip(self.params.p_free_names, par)))
        return params

    
    def load_cmb(self):
        """
        Loads the CMB spectrum
        """
        cmb_lensingfile = np.loadtxt('data/camb_lens_nobb.dat')
        cmb_bbfile = np.loadtxt('data/camb_lens_r1.dat')

        self.cmb_ells = cmb_bbfile[:, 0]
        mask = (self.cmb_ells <= self.bpw_l.max()) & (self.cmb_ells > 1)
        self.cmb_ells = self.cmb_ells[mask]
        nell = len(self.cmb_ells)
        self.cmb_tens = np.zeros([self.npol, self.npol, nell])
        self.cmb_lens = np.zeros([self.npol, self.npol, nell])
        self.cmb_scal = np.zeros([self.npol, self.npol, nell])
        ind = 0 
        self.cmb_tens[ind, ind] = (cmb_bbfile[:, 3][mask] -
                                   cmb_lensingfile[:, 3][mask])
        self.cmb_lens[ind, ind] = cmb_lensingfile[:, 3][mask]
        ind = 1
        self.cmb_tens[ind, ind] = (cmb_bbfile[:, 2][mask] -
                                   cmb_lensingfile[:, 2][mask])
        self.cmb_scal[ind, ind] = cmb_lensingfile[:, 2][mask]
        return

    def integrate_seds(self, params):
        single_sed = np.zeros([self.fg_model.n_components,
                               self.nfreqs])
        comp_scaling = np.zeros([self.fg_model.n_components,
                                 self.nfreqs, self.nfreqs])
        fg_scaling = np.zeros([self.fg_model.n_components,
                               self.fg_model.n_components,
                               self.nfreqs, self.nfreqs])
        rot_matrices = []

        comp = self.fg_model.components['dust']
        units = comp['cmb_n0_norm']
        list_keys = list(comp['names_sed_dict'].keys())
        pars_fgc = ['nu0', 'temp', 'beta_d']
        sed_params = [self.fg_model.params_fgc[k] for k in pars_fgc]
        
        def sed(nu):
            return comp['sed'].eval(nu) 

        for tn in range(self.nfreqs):
            self.bpass = Bpass(tn) 
            sed_b = np.sum(self.bpass.dnu*self.bpass.bnu*self.bpass.nu**2*sed(tn))
            single_sed[0, tn] = sed_b * units
            comp_scaling[0]= np.outer(single_sed, single_sed)

        fg_scaling[0, 0] = comp_scaling

        return fg_scaling

    
    def evaluate_power_spectra(self, params):
        fg_pspectra = np.zeros([self.fg_model.n_components, self.npol,
                                self.npol, self.n_ell])
        # Fill diagonal
        comp = self.fg_model.components['dust'] 
        # TODO: BB only
        ip1 = 1 
        ip2 = 1 
        pspec_params = [params[comp['names_cl_dict']['BB'][k]] for k in comp['cl'].params]
        p_spec = comp['cl'].eval(self.bpw_l, *pspec_params) * self.dl2cl
        fg_pspectra[0, ip1, ip2] = p_spec

        return fg_pspectra

    
    def model(self, params):
        """
        Defines the total model and integrates over
        the bandpasses and windows.
        """
        # [npol,npol,nell]
        cmb_cell = (params['r_tensor'] * self.cmb_tens +
                    params['A_lens'] * self.cmb_lens +
                    self.cmb_scal) * self.dl2cl
        # [nell,npol,npol]
        cmb_cell = np.transpose(cmb_cell, axes=[2, 0, 1])
        # [ncomp, ncomp, nfreq, nfreq], [ncomp, nfreq,[matrix]]
        #fg_scaling, rot_m = self.integrate_seds(params)
        fg_scaling = self.integrate_seds(params)
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
        # [ncomp,nell,npol,npol]
        fg_cell = np.transpose(fg_cell, axes=[0, 3, 1, 2])

        # SED scaling
        cmb_scaling = np.ones(self.nfreqs)
        for f1, ff in enumerate(self.freqs):
            comp = self.fg_model.components['dust']
            cs = self.bpss[f1].convolve_sed(comp['sed']) #None) #params) ##
            cmb_scaling[f1] = cs

        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1, self.nfreqs):
                cls = cmb_cell * cmb_scaling[f1] * cmb_scaling[f2]

                # TODO: Loop over component pairs
                cls += 1 * fg_scaling[0, 0, f1, f2]
                cls_array_fg[f1, f2] = cls

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])
    
    def chi_sq_dx(self, params):
        """
        Chi^2 likelihood.
        """
        model_cls = self.model(params)
        return self.matrix_to_vector(self.bbdata - model_cls).flatten()

    
    def lnlike(self, par):
        """
        Likelihood without priors. 
        """
        params = self.build_params(par)
        dx = self.chi_sq_dx(params)
        like = -0.5 * np.dot(dx, np.dot(self.invcov, dx))
        
        return like

    
    def compute_fisher_matrix(self, covariance_matrix, parameters):
        """Compute Fisher forecast for parameters given the covariance matrix."""
        import numdifftools as nd
        from scipy.optimize import minimize

        self.invcov = np.linalg.pinv(covariance_matrix)
        ## TODO: invcov
        # Define the file path
        #file_path = "output/r0.0_100nsplits1_pinv_cov.npy"
        # Save pseudo-inverse to a .npy file
        #np.save(file_path, self.invcov)

        def chi2(par):
            c2 = -2*self.lnprob(par)
            return c2

        res = minimize(chi2, self.params.p0,
                       method="Powell")

        def lnprobd(p):
            l = self.lnprob(p)
            if l == -np.inf:
                l = -1E100
            return l

        fisher = - nd.Hessian(lnprobd)(res.x)
        return res.x, fisher

    def run_fisher_forecast(self, parameters):
        """Run the entire Fisher forecast process"""
        print("Read input Cls ...")
        cls, _, nell = self.read_cls()
        print("Calculate covariance matrix ...")
        covariance_matrix = self.calculate_covariance(cls, self.nmaps, nell)
        print("Calculate fisher matrix ...")
        fisher_matrix = self.compute_fisher_matrix(covariance_matrix, parameters)
        return fisher_matrix

import argparse

def main(cls_file, parameters):
    #parameters = [0.0, 0.1]
    fisher_forecaster = FisherForecast(cls_file)
    fisher_matrix = fisher_forecaster.run_fisher_forecast(parameters)
    # TODO: add sigma = inv(sqrt(fisher_matrix))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Fisher forecast")
    parser.add_argument("cls_file", type=str, help="Path to input Cls file")
    parser.add_argument("parameters", nargs="+", type=float, help="List of parameter values")
    args = parser.parse_args()
    main(args.cls_file, args.parameters)
