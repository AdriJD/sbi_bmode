import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import jax
import jax.numpy as jnp
import optax
#from jax.scipy.optimize import minimize
from pixell import enmap, reproject, curvedsky, enplot

from sbi_bmode import spectra_utils

#jax.config.update('jax_enable_x64', True)
#jax.config.update("jax_debug_nans", True)

opj = os.path.join

odir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
maskdir = opj(odir, 'masks')
imgdir = opj(odir, 'img')

def get_sed_dust(freq, beta, temp, freq_pivot):
    '''
    Compute the SED of the Planck modified blackbody dust model in RJ temp,
    see Eq. 15 in Choi et al.
    
    Parameters
    ----------
    freq : float
        Effective freq of passband in Hz.    
    beta : float
        Frequency power law index.
    temp : float
        Dust temperature.
    freq_pivot : float
        Pivot frequency in Hz.

    Returns
    -------
    out : float
        SED^2 evaluated at input freq.
    '''

    b_freq = spectra_utils.get_planck_law(freq, temp)
    b_pivot = spectra_utils.get_planck_law(freq_pivot, temp)
    
    return (freq / freq_pivot) ** (beta - 2) * b_freq / b_pivot

def get_data_model(beta, freq, amps, freq_pivot, temp):
    '''
    Parameters
    ----------
    amps : (npix)
        Reference amplitude at freq_pivot.

    Returns
    -------
    model : (npix)
    '''

    sed_fact = get_sed_dust(freq, beta, temp, freq_pivot) * spectra_utils.get_g_fact(freq) / \
        spectra_utils.get_g_fact(freq_pivot)
    model = amps * sed_fact

    return model

get_data_model_batched = jax.vmap(get_data_model, in_axes=(None, 0, None, None, None),
                                  out_axes=0)

def get_loss(beta, freqs, amps, freq_pivot, data, temp):

    model = get_data_model_batched(beta, freqs, amps, freq_pivot, temp)
    diff = data - model

    loss = jnp.sum(diff ** 2)
    #jax.debug.print('{x}', x=diff[3,:10])
    return loss
    
#temp = 19.6
temp = 22.
freq_pivot = 340e9
freqs = ['wK', 'f030', 'f040', 'f090', 'f150', 'f230', 'f290', 'p353']
central_freq_dict = {'wK' : 25.e9, 'f030' : 27e9, 'f040' : 39e9, 'f090' : 93e9,
                     'f150' : 145e9, 'f230' : 225e9, 'f290' : 280e9, 'p353' : 340e9}

central_freqs = jnp.asarray([central_freq_dict[freq] for freq in freqs])

nfreq = len(freqs)
dust_models = ['d12']
sync_models = []

nside = 128
lmax = 200
ainfo = curvedsky.alm_info(lmax)
oversample = 2
shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax),
                                        2 * np.pi / (2 * oversample * lmax + 1)])
omap = enmap.zeros(shape, wcs)

mask = hp.read_map(opj(maskdir, 'mask.fits'))
#mask[:] = 1

for model in (dust_models + sync_models):

    data = jnp.zeros((nfreq, 12 * nside ** 2))
    
    for fidx, freq in enumerate(freqs):

        ## Load B-mode map. Units are uK_CMB.
        #data = data.at[fidx].set(hp.alm2map(hp.read_alm(opj(odir, f'pysm_{model}_{freq}.fits')), nside))

        # Read T map.
        #data = data.at[fidx].set(hp.read_map(opj(odir, f'pysm_{model}_{freq}_I_map_unmasked.fits')).astype(jnp.float64))
        data = data.at[fidx].set(hp.read_map(opj(odir, f'pysm_{model}_{freq}_I_map_unmasked.fits')).astype(jnp.float32))        

        #hp.mollview(np.asarray(data[fidx]))
        #plt.savefig(opj(imgdir, 'debug'))
        #plt.close()

        #exit()
        
    # Init beta
    beta_init = jnp.ones((12 * 128 ** 2)) * 1.6
    temp_init = jnp.ones((12 * 128 ** 2)) * 20.    
    #beta_init = jnp.ones((12 * 128 ** 2)) * 3.
    amps = data[-1]

    # NOTE
    #beta_init = beta_init[:5001]
    #amps = amps[:5001]
    #data = data[:,:5001]

    beta_init = beta_init[mask > 0.1]
    temp_init = temp_init[mask > 0.1]    
    amps = amps[mask > 0.1]
    data = data[:, mask > 0.1]
    
    # fit
    #loss = lambda beta : get_loss(beta, central_freqs, amps, freq_pivot, data, temp)
    loss = lambda x : get_loss(x[0], central_freqs, amps, freq_pivot, data, x[1])    
    loss_jit = jax.jit(loss)

    #print(loss_jit(beta_init))

    #solver = optax.adam(learning_rate=0.003)
    solver = optax.adam(learning_rate=0.001)    
    #solver = optax.adam(learning_rate=0.01)
    #solver = optax.adam(learning_rate=0.05)
    #solver = optax.lbfgs()
    x_init = jnp.zeros((2, beta_init.size))
    x_init = x_init.at[0].set(beta_init)
    x_init = x_init.at[1].set(temp_init)    
    
    #opt_state = solver.init(beta_init)
    opt_state = solver.init(x_init)    
    #params = beta_init.copy()
    params = x_init.copy()    
    nsteps = 800
    #nsteps = 1000

    #value_and_grad = optax.value_and_grad_from_state(loss_jit)
    for idx in range(nsteps):
        grad = jax.grad(loss_jit)(params)
        #value, grad = value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(grad, opt_state, params)
        #updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=loss_jit)
        params = optax.apply_updates(params, updates)
        print(f'{loss_jit(params):.5E}, {params}')
        #betas[idx] = params[0]
        #print()    
    #print(minimize(loss_jit, beta_init, method='BFGS', tol=1e-6))

    # fig, ax = plt.subplots(dpi=300)
    # for idx in range(nsteps):
    #     ax.plot(central_freqs * 1e-9, get_data_model_batched(betas[idx], central_freqs, amps, freq_pivot, temp))
    # ax.plot(central_freqs * 1e-9, get_data_model_batched(params, central_freqs, amps, freq_pivot, temp), color='black',
    #         zorder=2.01) 
    # ax.scatter(central_freqs * 1e-9, data, zorder=2.1)        
    # fig.savefig(opj(imgdir, 'debug_beta_fit'))
    # plt.close(fig)


    np.random.seed(1)
    fig, axs = plt.subplots(dpi=300, nrows=10, constrained_layout=True, figsize=(3, 20), sharex=True)
    for aidx, idx in enumerate(np.random.choice(params.size, 10, replace=False)):
        print(idx)
        axs[aidx].plot(central_freqs * 1e-9,
                       get_data_model_batched(params[0,idx], central_freqs, amps[idx], freq_pivot, params[1,idx]),
                       color='black',
            zorder=2.01) 
        axs[aidx].scatter(central_freqs * 1e-9, data[:,idx], zorder=2.1)
    fig.savefig(opj(imgdir, 'debug_beta_fit'))
    plt.close(fig)
    
    beta_map = np.zeros((12 * 128 ** 2))
    beta_map[mask > 0.1] = np.asarray(params[0])

    print(np.std(beta_map[mask > 0.1]))
    
    temp_map = np.zeros((12 * 128 ** 2))    
    temp_map[mask > 0.1] = np.asarray(params[1])
    
    hp.mollview(beta_map, min=1.3, max=1.9)
    plt.savefig(opj(imgdir, f'beta_estimate_mollview_{model}'))
    plt.close()
    
    #beta_map[:] = np.asarray(params)    
    alm_beta = hp.map2alm(beta_map * mask, lmax)
    curvedsky.alm2map(alm_beta, omap, ainfo=ainfo)
    plot = enplot.plot(omap, colorbar=True, ticks=30, quantile=0.05, min=1.3, max=1.9)
    enplot.write(opj(imgdir, f'beta_estimate_{model}'), plot)

    hp.mollview(temp_map, min=18, max=21)
    plt.savefig(opj(imgdir, f'temp_estimate_mollview_{model}'))
    plt.close()
    
    #temp_map[:] = np.asarray(params)    
    alm_temp = hp.map2alm(temp_map * mask, lmax)
    curvedsky.alm2map(alm_temp, omap, ainfo=ainfo)
    plot = enplot.plot(omap, colorbar=True, ticks=30, quantile=0.05, min=18, max=21)
    enplot.write(opj(imgdir, f'temp_estimate_{model}'), plot)
    
    # store

    # Only consider pixels for which m > 0.1



