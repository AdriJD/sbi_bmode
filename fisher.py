import numpy as np

class FisherForecast:
    """Class to perform Fisher forecast"""

    def __init__(self, cls_file):
        self.cls_file = cls_file
        self.freqs = np.array([27., 39., 93., 145., 225., 280.])
        self.nfreqs = len(self.freqs)
        self.npol = 2
        self.nmaps = self.npol * self.nfreqs
        self.ncross = (self.nmaps * (self.nmaps + 1)) // 2
        self.nsplits = 1  # 4
        self.nside = 128
        self.lmax = 3 * self.nside - 1
        self.fsky = 1
        self.dell = 1

    def read_cls(self):
        """Read input Cls"""
        nfreq, npol, _, _, nell = np.load(self.cls_file).shape
        cls = np.load(self.cls_file).reshape(nfreq * npol, nfreq * npol, nell)
        return cls, self.nmaps, nell

    def calculate_covariance(self, cls, nmaps, nell):
        """Calculate covariance (Knox formula)"""
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

    def compute_fisher_matrix(self, covariance_matrix, parameters):
        """Compute Fisher forecast for parameters given the covariance matrix."""
        fisher_matrix = np.zeros((len(parameters), len(parameters)))

        for i, param_i in enumerate(parameters):
            for j, param_j in enumerate(parameters):
                grad_cov_i = np.gradient(np.gradient(covariance_matrix, param_i, axis=0), param_j, axis=1)
                fisher_matrix[i, j] = 0.5 * np.trace(np.dot(np.linalg.pinv(covariance_matrix), np.dot(grad_cov_i, np.linalg.pinv(covariance_matrix))))

        return fisher_matrix

    def run_fisher_forecast(self, parameters):
        """Run the entire Fisher forecast process"""
        cls, _, nell = self.read_cls()
        covariance_matrix = self.calculate_covariance(cls, self.nmaps, nell)
        fisher_matrix = self.compute_fisher_matrix(covariance_matrix, parameters)
        return fisher_matrix

import argparse

def main(cls_file, parameters):
    fisher_forecaster = FisherForecast(cls_file)
    fisher_matrix = fisher_forecaster.run_fisher_forecast(parameters)
    # TODO: add sigma = inv(sqrt(fisher_matrix))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Fisher forecast")
    parser.add_argument("cls_file", type=str, help="Path to input Cls file")
    parser.add_argument("parameters", nargs="+", type=float, help="List of parameter values")

    args = parser.parse_args()
    main(args.cls_file, args.parameters)
