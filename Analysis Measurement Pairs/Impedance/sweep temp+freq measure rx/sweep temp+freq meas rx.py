# -*- coding: utf-8 -*-
"""
Python script for reading impedance measurements at various temperatures and
converting them to all immittance formalisms and use in Zview. This script also
determines the activation energy using the low frequency admittance data
Author: Jared Carter, Clive Randall Group
Version 1.0
Last updated: 2017-09-18
"""

#%%############################################################################
### Change these values #######################################################
filename = 'STO005.diel'
area = 0.25 # cm^2
thickness = 0.05 # cm
export_zview = True      # True if you want to export zview data
export_immittance = True # True if you want to export immittance data
find_activation = True # Attempt to determine activation energy
min_temp = 200.0 # Do not attempt to fit temperatures lower than this
max_temp = 410.0 # Do not attempt to fit temperatures higher than this
#%%############################################################################
###############################################################################

# Import modules
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r'/Users/mac/Box/AFOSR_Randall/Python Scripts')
sys.path.append('C:\\Users\\Jared\\Box Sync\\AFOSR_Randall\\Python Scripts')
sys.path.append('D:\\Eigene Dateien\\uni\\Box\\b-afsor-project Shared\\AFOSR_Randall\\Python Scripts')
import diel
k = 8.6173303e-5


#%% Make new folders for zview data and immittance data
# Get current path
my_path = sys.path[0]
# Are we exporting zview data?
if export_zview is True:
    # Make the new directory name correctly on Mac and Windows
    z_directory = os.path.join(my_path, 'zview')
    # If the directory doesn't currently exist, then make it
    if not os.path.exists(z_directory):
        os.makedirs(z_directory)

if export_immittance is True:
    imm_directory = os.path.join(my_path, 'immittance')
    if not os.path.exists(imm_directory):
        os.makedirs(imm_directory)

#%% Read data
s,m = diel.getdata(filename)
# Get the coordinates for the temperature and frequency dimensions
my_coords = {'set_temp': np.array(s['A TEMPERATURE_set']),
             'freq' : np.array(s['B FREQ_set'])}

# Get the shape that the rx data should be in
my_shape = (my_coords['set_temp'].size, my_coords['freq'].size)

# Get and reshape the real part of the impedance
rx0_r = np.real(m['CMPLX RX0'])
rx0_r = rx0_r.reshape(my_shape)
# Create a data array for the real part of the impedance with rows labeled with
# the set temperature, and columns labeled with frequency
r = xr.DataArray(rx0_r, coords=my_coords, dims=['set_temp', 'freq'])

# Do the same for the imaginary part of the impedance
rx0_x = np.imag(m['CMPLX RX0'])
rx0_x = rx0_x.reshape(my_shape)
x = xr.DataArray(rx0_x, coords=my_coords, dims=['set_temp', 'freq'])

# Read the measured temperature before the impedance measurement
temp1 = np.array(m['REAL TEMPERATURE1'])
t1 = xr.DataArray(temp1, coords=[('set_temp', my_coords['set_temp'])])
# Read the measured temperature after the impedance measurement
temp2 = np.array(m['REAL TEMPERATURE2'])
t2 = xr.DataArray(temp2, coords=[('set_temp', my_coords['set_temp'])])
# What is the difference between the set temperature and the average measured
# temperature?
t3 = my_coords['set_temp'] - (t1+t2)/2.0

# Create a new data array with all of the temperature information
# The .T transposes the data array so the set temperatures are the rows and
# the labels are the columns
t_out = xr.concat((t1, t2, t3), dim='id').T
# Label the different temperatures
t_out['id'] = ['Temperature 1', 'Temperature 2', 'Avg. Difference from set']
# xarray is not very good at exporting data to .csv files, so the .to_pandas()
# method is called to convert the data array to a pandas dataframe.
# Immediately after that, the .to_csv() method is called ont the newly created
# dataframe and is exported to a .csv file
t_out.to_pandas().to_csv('temp_summary.csv')

# Initialize arrays for fitting information
fit_temp = np.zeros_like(my_coords['set_temp']) * np.nan
y_low = np.zeros_like(my_coords['set_temp']) * np.nan

# Export impedance data at each temperature. The enumerate function returns
# the number of the loop and the current value of the list or array. For
# example i=0 and temp=140.0 at the start, then i=1 and temp=160.0 and so on.
for i, temp in enumerate(my_coords['set_temp']):
    # Average temperature of measurement
    fit_temp[i] = (t1.sel(set_temp=temp) + t2.sel(set_temp=temp))/2.0
    # Filename for current temperature
    current_name = filename[:-5]+'_{}'.format(int(temp))
    # Real part of the impedance for current temperature
    current_r = r.sel(set_temp=temp).data
    # Imiginary part of the impedance for current temperature
    current_x = x.sel(set_temp=temp).data
    # Get all immittance formalisms
    imm = diel.convert_z(f=my_coords['freq'], r=current_r, x=current_x,
                         A=area, t=thickness)

    # Median low frequency admittance (to ignore outliers)
    y_low[i] = np.median(imm[2][-10:])
    # Should we export zview data?
    if export_zview is True:
        # Stack the data for export
        zview = np.column_stack((my_coords['freq'], current_r, current_x))
        # Make the file path correctly on Mac or Windows
        z_fpath = os.path.join(z_directory, current_name+'_zview.csv')
        # Save
        np.savetxt(z_fpath, zview, delimiter=',')
    # Should we export immittance data
    if export_immittance is True:
        # Make the file path correctly on mac or windows
        imm_path =  os.path.join(imm_directory, current_name+'_all.csv')
        # Header for .csv data
        h = 'f,1/f,realY,imagY,realZ,imagZ,realM,imagM,realE,imagE,tandelta'
        # Save
        np.savetxt(imm_path, np.column_stack(imm), delimiter=',', header=h)
#%% Find activation energy
if find_activation is True:
    # Ignore points outside of temperature range
    mask = np.where((fit_temp >= min_temp) & (fit_temp <= max_temp))
    # 1000 / K
    x = 1000.0/(fit_temp[mask] + 273.15)
    # log10(sigma)
    y = np.log10(y_low[mask]*thickness/area)
    # log10(sigma * T) = log10(sigma) + log10(T)
    y2 = y + np.log10(fit_temp[mask] + 273.15)
    # Arrhenius fit
    res = linregress(x, y)
    # Arrhenius fit with temperature prefactor
    res2 = linregress(x, y2)
    # Find activation energy from slope of plot
    EA = (-1e3*res.slope * k)/(np.log10(np.e))
    EA2 = (-1e3*res2.slope * k)/(np.log10(np.e))
    
    # Plot results
    # Initialize figure
    fig = plt.figure()
    # Define axes to plot
    ax1 = plt.subplot(111)
    # Plot conductivity vs inverse temperature
    ax1.plot(x, y, 'o', label='low freq. Y, E$_A$={:.2f} eV'.format(EA))
    # Plot the fit
    ax1.plot(x, res.slope*x + res.intercept, 'r--')
    # Label the axes
    ax1.set_xlabel('1000/T (1/K)')
    ax1.set_ylabel(u'log(σ [S/cm])')
    ax1.legend()
    # Make an x-axis on the top of the plot to show temp. in °C
    ax2 = ax1.twiny()
    # Temperatures we would like to see
    tick_temps = np.array([200.0, 300.0, 400.0])
    # Where are these tick labels going to be?
    ax2.set_xticks(1000/(273.15+tick_temps))
    # Display the tick labels
    ax2.set_xticklabels(tick_temps)
    ax2.set_xlabel(u'Temperature (°C)')
    # Force the x-axis bounds to be the same as the bottom x-axis
    ax2.set_xbound(ax1.get_xbound())
    # Show the plot
    plt.show()
    
    fig = plt.figure()
    # Define axes to plot
    ax1 = plt.subplot(111)
    # Plot conductivity vs inverse temperature
    ax1.plot(x, y2, 'o', label='low freq. Y, E$_A$={:.2f} eV'.format(EA2))
    # Plot the fit
    ax1.plot(x, res2.slope*x + res2.intercept, 'r--')
    # Put the activation energy on the plot
    # Label the axes
    ax1.set_xlabel('1000/T (1/K)')
    ax1.set_ylabel(u'log(σT [S-K/cm])')
    ax1.legend()
    # Make an x-axis on the top of the plot to show temp. in °C
    ax2 = ax1.twiny()
    # Temperatures we would like to see
    tick_temps = np.array([200.0, 300.0, 400.0])
    # Where are these tick labels going to be?
    ax2.set_xticks(1000/(273.15+tick_temps))
    # Display the tick labels
    ax2.set_xticklabels(tick_temps)
    ax2.set_xlabel(u'Temperature (°C)')
    # Force the x-axis bounds to be the same as the bottom x-axis
    ax2.set_xbound(ax1.get_xbound())
    # Show the plot
    plt.show()
    