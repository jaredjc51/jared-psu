# -*- coding: utf-8 -*-
"""
Python script for reading TSDC poling and extracting time,
leakage current, current density, and conductivity.

Author: Jared Carter, Clive Randall Group
Version 1.0
Last updated: 2017-11-08
"""

#%%############################################################################
### Change these values #######################################################
filename = 'poling_file.diel'
area = 0.012593 # cm^2
thickness = 0.203 # cm
field = 140.0 # V/cm
#%%############################################################################
###############################################################################

import sys
import numpy as np
# Add your own path to the .diel module here.
sys.path.append(r'C:\Users\sulat\Box Sync\AFOSR_Randall\Python Scripts')
sys.path.append(r'C:\Users\Jared\Box Sync\AFOSR_Randall\Python Scripts')
import diel

#%% Read and calculate data

# Read data
s,m = diel.getdata(filename)
# Extract time
poling_times = np.array(s['A TIME_set'])
# Extract current
poling_pa = np.array(m['REAL PA0'])
# Calculate current density
poling_j = poling_pa / area
# Calculate conductivity
poling_sigma = poling_j / field

#%% Export data
# Stack columns for exporting
poling_data = np.column_stack((poling_times, poling_pa, poling_j,
                               poling_sigma))

# Define column names
hp = 'time(s),pa(A),j(A/cm^2),sigma(S/cm)'
# Export data
np.savetxt(filename[:-5]+'_poling.csv', poling_data, delimiter=',', header=hp)
