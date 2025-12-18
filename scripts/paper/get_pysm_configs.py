import os
import yaml

opj = os.path.join

freqs = ['wK', 'f030', 'f040', 'f090', 'f150', 'f230', 'f290', 'p353']

dust_models = ['d1', 'd4', 'd10', 'd10alt', 'd12']
sync_models = ['s5', 's7', ]

pysmdir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
odir = opj(pysmdir, 'configs')
os.makedirs(odir, exist_ok=True)

for dust_model in dust_models:
    for sync_model in sync_models:

        out = {}
        out['dust'] = {}
        out['sync'] = {}

        for fidx, freq in enumerate(freqs):
            
            #alm_dust = hp.read_alm(opj(pysmdir, f'pysm_{dust_model}_{freq}.fits'))
            #alm_sync = hp.read_alm(opj(pysmdir, f'pysm_{sync_model}_{freq}.fits'))
            #alm = alm_dust + alm_sync + alm_cmb
            out['dust'][freq] = opj(pysmdir, f'pysm_{dust_model}_{freq}.fits')
            out['sync'][freq] = opj(pysmdir, f'pysm_{sync_model}_{freq}.fits')

            with open(opj(odir, f'pysm_{dust_model}_{sync_model}.yaml'), 'w') as f:
                yaml.dump(out, f)
