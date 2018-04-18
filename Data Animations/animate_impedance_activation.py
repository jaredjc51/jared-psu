# -*- coding: utf-8 -*-
"""
summary

Note: to animate with FFMPEG, use the following command:
ffmpeg -framerate 2 -f image2 -s 1920x1080 -i test_%02d.png -r 30 \
-vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4

Author: Jared Carter, Clive Randall Group
Last update 2018-04-04
version 1.0
"""


#%% CHANGE THESE VALUES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

area = 0.335 * 0.328  # cm^2
thickness = 0.05  # cm

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Import modules
import numpy as np # Efficient array handling
import pandas as pd
import glob # Get all files in folder
from lmfit.models import LinearModel, GaussianModel # Fitting
import xarray as xr # Labeled, multi-dimensional arrays
import matplotlib.pyplot as plt # Plotting
from matplotlib import cm # Auto-generated colors for plots
from matplotlib import style # Add custom style

filenames = glob.glob('*.csv')

#%% Plotting
style.use('video') # Custom style
# Get custom colors
cm_subsection = np.linspace(0, 1, len(filenames))
colors = [cm.viridis(a) for a in cm_subsection]
# Initialize figure
plt.figure()
# ax1 is admittance vs frequency
ax1 = plt.subplot(121)
ax1.axis([10, 1e5, 1e-10, 5e-7])
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Real Admittance (S/cm)')
# ax2 is arrhenius
ax2 = plt.subplot(122)
ax2.axis([1.87, 2.71, -9.9, -6.6])
ax2.set_xlabel('1000/T (1/K)')
ax2.set_ylabel(r'$\log(\sigma$ [S/cm])')

sigmas = np.zeros(len(filenames))
invT = np.zeros_like(sigmas)

for i, fname in enumerate(filenames):
    # Find temperature
    split_name = fname.split('_')
    for part in split_name:
        try:
            c_temp = int(part)
        except ValueError:
            pass
    invT[i] = 1000.0 / (c_temp + 273.15)
    yprime = pd.read_csv(fname, header=0, index_col='# f', usecols=['# f', 'realY'])
    yprime = yprime * thickness / area
    ax1.loglog(yprime.index, yprime, '.', c=colors[i])
    sigmas[i] = np.log10(yprime[yprime.index < 50].median())
    ax1.axhline(10**sigmas[i], c='r')
    ax2.plot(invT[i], sigmas[i], 'o', c=colors[i])
    plt.savefig('test_{:02d}.png'.format(i), dpi=96)

m2 = LinearModel()
# Initialize parameters for linear model
p2 = m2.make_params()
# Fit Arrhenius
out2 = m2.fit(sigmas, p2, x=invT)
EA = out2.values['slope'] * 8.6173303e-5 * -1000 / np.log10(np.e)
ax2.plot(invT, out2.best_fit, 'r-', label=r'E$_A$ = {:.3f} eV'.format(EA))
ax2.legend()
plt.show()
plt.savefig('impedance_activation.png'.format(i), dpi=96)
