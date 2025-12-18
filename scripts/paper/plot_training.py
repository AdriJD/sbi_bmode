import os

import numpy as np
import matplotlib.pyplot as plt

opj = os.path.join

basedir = '/u/adriaand/project/so/20240521_sbi_bmode'

subdir = 'run50'
idir = opj(basedir, subdir)
imgdir = opj(idir, 'img')
os.makedirs(imgdir, exist_ok=True)

training_loss = np.load(opj(idir, 'training_loss.npy'))
validation_loss = np.load(opj(idir, 'validation_loss.npy'))

fig, ax = plt.subplots(dpi=300)
ax.plot(training_loss, label='training_loss')
ax.plot(validation_loss, label='validation_loss')
ax.legend(frameon=False)
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
fig.savefig(opj(imgdir, 'loss'))
plt.close(fig)

