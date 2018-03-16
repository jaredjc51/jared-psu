# -*- coding: utf-8 -*-
"""
Python script for reading TSDC ramp and extracting time, temperature,
depolarization current, and depolarization current density.

Author: Jared Carter, Clive Randall Group
Version 1.0
Last updated: 2017-11-07
"""

#%%############################################################################
### Change these values #######################################################
filename = 'tsdc_file.diel'
area = 0.012593 # cm^2
thickness = 0.203 # cm
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
# Extract temperature
tsdc_temps = np.array(m['REAL TEMPERATURE0'])
# Extract time (force time=0 at beginning of measurement)
tsdc_times = np.array(m['REAL TIME0']) - m['REAL TIME0'][0]
# Extract depolarization current
tsdc_pa = np.array(m['REAL PA0'])
# Calculate depolarization current density
tsdc_j = tsdc_pa / area

#%% Export data
# Stack columns for exporting
tsdc_data = np.column_stack((tsdc_times, tsdc_temps, tsdc_pa, tsdc_j))

# Define column names
ht = 'time(s),temp(C),pa(A),j(A/cm^2)'
# Export data
np.savetxt(filename[:-5]+'_tsdc.csv', tsdc_data, delimiter=',', header=ht)
