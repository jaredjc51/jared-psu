# -*- coding: utf-8 -*-
"""
Python script for TSDC
Author: Jared Carter, Clive Randall Group
Version 1.1
Last updated: 2017-09-14
"""

#%% CHANGE THIS INFORMATION ###################################################
###############################################################################

filename = 'STO005.diel'
area = 0.25  # cm^2
thickness = 0.047  # cm
plot = True
export = True

#%%############################################################################
###############################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
# Add your own path to the .diel module here.
sys.path.append(r'/Users/mac/Box/AFOSR_Randall/Python Scripts')
sys.path.append('C:\\Users\\Jared\\Box Sync\\AFOSR_Randall\\Python Scripts')
sys.path.append('D:\\Eigene Dateien\\uni\\Box\\b-afsor-project Shared\\AFOSR_Randall\\Python Scripts')
import diel

s, m = diel.getdata(filename)
# What are the fields?
fields = np.around(np.array(s['A DC_set']) / thickness)

#%% Get poling data

# Values from sweep time
poling_times = np.array(s['B TIME_set'])
# dimensions of poling array
shape_poling = (fields.size, poling_times.size)
# Poling measurement data
pa0 = np.array(m['REAL PA0'])
# Make poling measurement a 2-dimensional array
pa0.resize(shape_poling)
# Make poling measurement a data array, with dimensions of field and time
poling_I = xr.DataArray(pa0, coords=[('field', fields), ('time', poling_times)])
# Calculate current density J = I / A
poling_J = poling_I / area
# Calculate conductivity sigma = J / E
poling_sigma = poling_J / xr.DataArray(fields, dims='field')

# Data array with poling data
poling_x = xr.concat((poling_I, poling_J, poling_sigma),
                     pd.Index(['I', 'J', 'sigma'], name='id'))

#%% Get TSDC data

# Values from sweep temperature
tsdc_set_temps = np.array(s['C TEMP_set'])
# dimensions of tsdc array
shape_tsdc = (fields.size, tsdc_set_temps.size)
# TSDC current measurement data
pa1 = np.array(m['REAL PA1'])
# Make TSDC cuurent measurement a 2-dimensional array
pa1.resize(shape_tsdc)
# Make TSDC current measurement a data array, with dimensions of
# field and set_temperature
tsdc_I = xr.DataArray(pa1, coords=[('field', fields),
                                   ('set_temp', tsdc_set_temps)])

# Calculate current density J = I / A
tsdc_J = tsdc_I / area

# TSDC temperature measurement data
temp1 = np.array(m['REAL TEMPERATURE1'])
# make TSDC temperature measurement a 2-dimensional array
temp1.resize(shape_tsdc)
# Make TSDC temperature measurement a data array, with dimensions of
# field and set_temperature
tsdc_temp = xr.DataArray(temp1, coords=[('field', fields),
                                        ('set_temp', tsdc_set_temps)])

# TSDC time check data
time1 = np.array(m['REAL TIME1'])
# make TSDC time check a 2-dimensional array
time1.resize(shape_tsdc)
# Make TSDC time check a data array, with dimensions of field and 
# set_temperature
tsdc_time = xr.DataArray(time1, coords=[('field', fields),
                                        ('set_temp', tsdc_set_temps)])
# Make the first time of each TSDC time check at each field 0
tsdc_time = tsdc_time - tsdc_time.sel(set_temp=25, method='nearest')

# Data array with TSDC data
tsdc_x = xr.concat((tsdc_temp, tsdc_time, tsdc_I, tsdc_J),
                   pd.Index(['temp', 'time', 'I', 'J'], name='id'))

#%% Export data to .csv files
if plot is True or export is True:
    if plot is True:
        plt.figure(1)
        plt.figure(2)
    for i, poling_field in enumerate(fields):
        if export is True:
            # Define filename for poling data
            poling_name = filename[:-5] + '_poling_{:04d}E.csv'.format(int(poling_field))
            # Define filename for TSDC data
            tsdc_name = filename[:-5] + '_tsdc_{:04d}E.csv'.format(int(poling_field))
            # Select poling data at each poling field
            poling_data = poling_x.sel(field=poling_field).T.to_pandas()
            # Select TSDC data at each poling field
            tsdc_data = tsdc_x.sel(field=poling_field).T.to_pandas()
            # Export poling data
            poling_data.to_csv(poling_name)
            # Export TSDC data
            tsdc_data.to_csv(tsdc_name)
        if plot is True:
            plt.figure(1)
            plt.plot(poling_x.time, poling_x.sel(field=poling_field, id='sigma'))
