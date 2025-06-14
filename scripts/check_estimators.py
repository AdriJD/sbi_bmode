import os
import yaml
import pickle
import argparse
import signal
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal, HalfNormal
from sbi.inference import SNPE, simulate_for_sbi, FMPE
from sbi.utils.sbiutils import seed_all_backends
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    MultipleIndependent
)
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.neural_nets import posterior_nn, flowmatching_nn
from getdist import plots as getdist_plots
from getdist import MCSamples
import optuna
import optuna.visualization.matplotlib as oplt
from optuna.trial import TrialState
from mpi4py import MPI

from sbi_bmode import (
    sim_utils, script_utils, compress_utils, custom_distributions)

opj = os.path.join
comm = MPI.COMM_WORLD
torch.manual_seed(0)

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def _handle_timeout(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")

    # Set the signal handler and an alarm.
    original_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Cancel the alarm and restore original handler.
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

def plot_training(opath, training_loss, validation_loss):
    '''
    Plot the training and validation loss.

    Parameters
    ----------
    opath : str
        Path to output png file.
    training_loss : (n_epoch) array
        Training loss per epoch.
    valication_loss : (n_epoch) array
        Validation loss per epoch.    
    '''

    fig, ax = plt.subplots(dpi=300)
    ax.plot(training_loss, label='training_loss')
    ax.plot(validation_loss, label='validation_loss')
    ax.legend(frameon=False)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    fig.savefig(opath)
    plt.close(fig)

def get_figure_from_ax(ax):
    '''Extract the matplotlib figure instance from an axes or array of axes.'''

    if isinstance(ax, np.ndarray):
        return ax.flat[0].get_figure()
    else:
        return ax.get_figure()

def plot_posterior(opath, samples, prior_samples, config):
    '''
    Plot corner plot of output posterior.

    Parameters
    ----------
    opath : str
        Path to output png file.
    samples :  (n_samples, n_parameters) array
        Posterior samples.
    prior_samples :  (n_prior_samples, n_parameters) array
        Prior samples.
    config : dict
        Dictionary with "data", "fixed_params" and "params" keys.
    '''

    param_label_dict = {'r_tensor' : r'$r$',
                        'A_lens' : r'$A_{\mathrm{lens}}$',
                        'A_d_BB' : r'$A_{\mathrm{d}}$',
                        'alpha_d_BB' : r'$\alpha_{\mathrm{d}}$',
                        'beta_dust' : r'$\beta_{\mathrm{d}}$',
                        'amp_beta_dust' : r'$B_{\mathrm{d}}$',
                        'gamma_beta_dust' : r'$\gamma_{\mathrm{d}}$',
                        'A_s_BB' : r'$A_{\mathrm{s}}$',
                        'alpha_s_BB' : r'$\alpha_{\mathrm{s}}$',
                        'beta_sync' : r'$\beta_{\mathrm{s}}$',
                        'amp_beta_sync' : r'$B_{\mathrm{s}}$',
                        'gamma_beta_sync' : r'$\gamma_{\mathrm{s}}$',
                        'rho_ds' : r'$\rho_{\mathrm{ds}}$'}

    data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
    prior, param_names = script_utils.get_prior(params_dict)
    param_labels = [param_label_dict[p] for p in param_names]
    param_truths = script_utils.get_true_params(params_dict)
    param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $.

    param_limits = script_utils.get_param_limits(prior, param_names)

    prior_samples = MCSamples(
        samples=prior_samples, names=param_names, labels=param_labels_g,
        ranges=param_limits)

    samples = MCSamples(
        samples=samples, names=param_names, labels=param_labels_g,
        ranges=param_limits)

    g = getdist_plots.get_subplot_plotter(width_inch=8)
    g.triangle_plot(samples, filled=True,
                    markers=param_truths)

    for i, name in enumerate(param_names):
        # This will add a line to the 1D marginal only
        g.add_1d(prior_samples, param=name, ls='--', color='gray', label='prior',
                 ax=g.subplots[i, i])

    g.export(opath, dpi=300)
    plt.close(g.fig)

def get_study_results(study):
    '''
    Summarize study.

    Parameters
    ----------
    study : optuna.study object
        The study to be summarized.

    Returns
    -------
    results : list of dicts
        Dictionary with `loss`, `trial_number` and `params` keys for each completed
        trial.
    '''

    trials = study.get_trials()
    results = []

    for trial in trials:
        if trial.state == TrialState.COMPLETE:
            results.append(
                {'loss' : trial.value, 'trial_number' : trial.number, 'params' : trial.params})

    return results

def save_results(opath, study):
    '''
    Save the ordered optuna results in a text file.

    Parameters
    ----------
    opath : str
        Path to output file.
    study : optuna.study object
        The study to be summarized.
    '''

    os.makedirs(os.path.dirname(opath), exist_ok=True)

    results = get_study_results(study)
    results = sorted(results, key=lambda d: d['loss'])
    param_names = results[0]['params'].keys()

    mat2save = np.zeros((len(results), 2 + len(param_names)))
    mat2save[:,0] = [res['trial_number'] for res in results]
    mat2save[:,1] = [res['loss'] for res in results]
    for pidx, param_name in enumerate(param_names):
        mat2save[:,2+pidx] = [res['params'][param_name] for res in results]

    header = '{:4s}\t{:22s}\t'.format('idx', 'loss')
    fmt = ['%4d', '%+22.15e']
    for param_name in param_names:
        header += '{:22s}\t'.format(param_name)
        fmt.append('%-22d' if isinstance(results[0]['params'][param_name], int) else '%+22.15e')

    np.savetxt(opath, mat2save, fmt=fmt, delimiter='\t', header=header)

def main(path_params, path_data, path_data_obs, odir, imgdir, config, n_samples,
         num_atoms=10, training_batch_size=200, learning_rate=0.0005, clip_max_norm=5.0,
         hidden_features=50, num_transforms=3,
         embed=False, embed_num_layers=2, embed_num_hiddens=25,  density_estimator_type='maf'):
    '''
    Run density estimation.

    Parameters
    ----------
    path_params : str
        Path to parameter draws.
    path_data : str
        Path to data draws.
    path_data_obs : str
        Path to observed data.
    odir : str
        Path to output directory.
    imgdir : str
        path to output directory for plots.
    config : dict
        Dictionary with "data", "fixed_params" and "params" keys.
    n_samples : int
        Number of posterior samples to draw.
    num_atoms : int, optional
        Number of atoms to use for classification.
    training_batch_size : int, optional
        Training batch size.
    learning_rate : float, optional
        Learning rate for Adam optimizer.
    clip_max_norm : float, optional
        Value at which to clip the total gradient norm in order to prevent exploding gradients.
        Use None for no clipping.
    embed : bool, optional
        Use an embedding network.
    embed_num_layers : int, optional
        Number of layers of embedding network
    embed_num_hiddens : int, optional
        Number of features in each hidden layer.
    density_estimator_type : str, optional
        String denoting density estimator for NPE.
    '''

    theta = torch.as_tensor(np.load(path_params))
    x = torch.as_tensor(np.load(path_data))
    x_obs = torch.as_tensor(np.load(path_data_obs))

    data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
    prior_list, param_names = script_utils.get_prior(params_dict)
    prior = MultipleIndependent(prior_list)
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    neural_posterior = posterior_nn(model=density_estimator_type,
                                    hidden_features=hidden_features,
                                    num_transforms=num_transforms)
    inference = SNPE(prior, density_estimator=neural_posterior)

    proposal = prior
    density_estimator = inference.append_simulations(
        theta, x, proposal=proposal).train(
            num_atoms=num_atoms, training_batch_size=training_batch_size,
            learning_rate=learning_rate, clip_max_norm=clip_max_norm,
            validation_fraction=0.1, stop_after_epochs=20,
            max_num_epochs=1000, use_combined_loss=True,
            show_train_summary=False)

    # Get validation loss for optuna.
    best_validation_loss = inference.summary['best_validation_loss'][0]

    # Plot and save training.
    training_loss = np.asarray(inference.summary['training_loss'])
    validation_loss = np.asarray(inference.summary['validation_loss'])

    np.save(opj(odir, 'training_loss.npy'), training_loss)
    np.save(opj(odir, 'validation_loss.npy'), validation_loss)

    plot_training(opj(imgdir, 'loss'), training_loss, validation_loss)

    # Build and plot posterior
    posterior = inference.build_posterior(density_estimator)
    samples = np.asarray(posterior.sample((n_samples,), x=x_obs))

    # Save samples.
    with open(opj(odir, 'posterior.pkl'), "wb") as handle:
        pickle.dump(posterior, handle)
    np.save(opj(odir, f'samples.npy'), samples)

    # Plot samples.
    try:
        with timeout(30): # Kill the sampling after 30 seconds.
            prior_samples = np.asarray(prior.sample((n_samples,)))
    except TimeoutException:
        print(f'{comm.rank=}: timeout sampling')
    else:
        plot_posterior(opj(imgdir, 'corner.png'), samples, prior_samples, config)

    return best_validation_loss

def run_optuna(trial, path_params, path_data, path_data_obs, odir_base, config, n_samples):
    '''
    Run one trial for the optimizer.

    Parameters
    ----------
    trial : optuna.trial object
        Input trial.
    path_params : str
        Path to .npy file containing parameter draws.
    path_data : str
        Path to .npy file containing data vectors.
    path_data_obs : str
        Path to .npy file containing observed data vector.
    odir_base : str
        Path to output directory.
    config : dict
        Dictionary with "data", "fixed_params" and "params" keys.
    n_samples : int
        Number of posterior draws.

    Returns
    -------
    loss : float
        Validation loss.
    '''

    print(f'{comm.rank=} {trial.number=}')

    training_batch_size = trial.suggest_int("training_batch_size", 5, 10) # Powers of 2.
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    clip_max_norm = trial.suggest_float("clip_max_norm", 1, 10)
    hidden_features = trial.suggest_int("hidden_features", 10, 100)
    num_transforms = trial.suggest_int("num_transforms", 1, 10)

    # Make output directory
    odir = opj(odir_base, f'trial_{trial.number:04d}')
    imgdir = opj(odir, 'img')
    os.makedirs(imgdir, exist_ok=True)

    loss = main(path_params, path_data, path_data_obs, odir, imgdir, config, n_samples,
                learning_rate=learning_rate, training_batch_size=2 ** training_batch_size,
                clip_max_norm=clip_max_norm, hidden_features=hidden_features,
                num_transforms=num_transforms)

    # Save trial parameters.
    trial_data = {'loss' : loss, 'trial_number' : trial.number, 'params' : trial.params}
    print(f'{comm.rank=}, {trial_data=}')

    with open(opj(odir, 'trial_data.pkl'), 'wb') as handle:
        pickle.dump(trial_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--params')
    parser.add_argument('--data')
    parser.add_argument('--data-obs')
    parser.add_argument('--odir')
    parser.add_argument('--config', help="Path to config yaml file.")
    parser.add_argument('--journal')
    parser.add_argument('--n_samples', type=int, default=50000, help="samples of posterior")

    args = parser.parse_args()

    if comm.rank == 0:
        print(f'Running with arguments: {args}')
        print(f'Running with {comm.size} MPI rank(s)')
        os.makedirs(args.odir, exist_ok=True)
        with open(args.config, 'r') as yfile:
            config = yaml.safe_load(yfile)
    else:
        config = None

    config = comm.bcast(config, root=0)

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(args.journal))

    objective = lambda trial: run_optuna(trial, args.params, args.data, args.data_obs,
                                         args.odir, config, args.n_samples)
    sampler = optuna.samplers.TPESampler(multivariate=True)
    study = optuna.create_study(study_name='test_study', storage=storage, direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=10)

    comm.barrier()
    if comm.rank == 0:

        os.makedirs(opj(args.odir, 'img'), exist_ok=True)

        save_results(opj(args.odir, 'img', 'results.txt'), study)

        ax = oplt.plot_optimization_history(study)
        fig = get_figure_from_ax(ax)
        fig.set_constrained_layout(True)
        fig.savefig(opj(args.odir, 'img', 'optimization_history'), dpi=300)
        plt.close(fig)

        ax = oplt.plot_param_importances(study)
        fig = get_figure_from_ax(ax)
        fig.set_constrained_layout(True)
        fig.savefig(opj(args.odir, 'img', 'param_importance'), dpi=300)
        plt.close(fig)

        ax = oplt.plot_slice(study)
        fig = get_figure_from_ax(ax)
        fig.set_constrained_layout(True)
        fig.savefig(opj(args.odir, 'img', 'slice'), dpi=300)
        plt.close(fig)

        ax = oplt.plot_contour(study)
        fig = get_figure_from_ax(ax)
        fig.set_size_inches(12, 16)
        fig.set_constrained_layout(True)        
        fig.savefig(opj(args.odir, 'img', 'contour'), dpi=300)
        plt.close(fig)
