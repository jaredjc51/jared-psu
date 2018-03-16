# -*- coding: utf-8 -*-
"""
Python script for reading impedance measurements as a function of temperature
and converting them to all immittance formalisms
Author: Jared Carter, Clive Randall Group
Version 1.2
Last updated: 2018-02-12

v 1.2
  - Works for impedance as a function of temperature instead of time
  - Improved documentation

v 1.1
  - Added ability to normalize Z and Y to geometry of sample
"""

#%%############################################################################
### Change these values #######################################################

filename = 'Thomas_example.diel'
area = 1.0 # cm^2
thickness = 1.0 # cm
# Save impedance in Z-view compatible format?
save_zview = True
# Area correction? True = Ohm/cm and S/cm for Z and Y
per_cm = False

#%%############################################################################
###############################################################################

# Import modules and define functions

import sys, os # Make folders on Mac and PC without breaking
sys.path.append(r'/Users/mac/Box/AFOSR_Randall/Python Scripts')
sys.path.append('C:\\Users\\Jared\\Box Sync\\AFOSR_Randall\\Python Scripts')
import diel
import numpy as np # Efficient array handling
e0 = 8.854188e-14 # F/cm (vacuum permittivity)
# Header for exporting immittance data
h = 'f,1/f,realY,imagY,realZ,imagZ,realM,imagM,realE,imagE,tandelta'

#%% Run the script

# Get path to current folder
my_path = sys.path[0]
# Make immittance file path if it doesn't exist
directory = os.path.join(my_path, 'immittance')
if not os.path.exists(directory):
    os.makedirs(directory)

# Make zview file path if it doesn't exist and we are exporting zview data
if save_zview is True:
    dir2 = os.path.join(my_path, 'zview')
    if not os.path.exists(dir2):
        os.makedirs(dir2)

# Import data
s,m = diel.getdata(filename)
# Get set temperatures
temps = np.array(s['A TEMPERATURE_set'])
# How many sweeps do we expect
num_sweeps = temps.size

# For every measurement in the Temperature sweep
for i in range(num_sweeps):
    goodfile = True
    try:
        # First deg measurement ends with 2
        data = m['LIST_REAL_CMPLX SWEEPFREQ  RX SWEEPDATA {}'.format(i+1)]
    # Don't crash if the measurement is in progress
    except KeyError:
        goodfile = False
    if goodfile is True:
        # Frequency
        f = data[:, 0]
        # Real impedance
        r = data[:, 1]
        # Imaginary impedance
        x = data[:, 2]
        # Get other immittance formalisms
        tup = diel.convert_z(f, r, x, area, thickness, per_cm)
        # Stack into one array
        out = np.column_stack(tup)
        # Export immittance
        export_name = os.path.join(directory, filename[:-5]+'_{0:03d}C_all.csv'.format(int(np.around(temps[i]))))
        np.savetxt(export_name, out, delimiter=',',header=h)
        # Export zview
        if save_zview is True:
            np.savetxt(os.path.join(dir2, filename[:-5]+'_{0:03d}C.csv'.format(int(temps[i]))), data, delimiter=',')
