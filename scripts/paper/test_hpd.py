import os

import numpy as np
import matplotlib.pyplot as plt

opj = os.path.join

imgdir = '/u/adriaand/project/so/20240521_sbi_bmode/test_hpd'
os.makedirs(imgdir, exist_ok=True)

np.random.seed(0)

samples = np.random.randn(10000)

fig, ax = plt.subplots(dpi=300)
ax.plot(np.sort(samples))
fig.savefig(opj(imgdir, 'ecdf'))
plt.close(fig)

