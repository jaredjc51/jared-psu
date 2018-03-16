# -*- coding: utf-8 -*-
"""
Python-GADD module for reading .diel files
Author: Jared Carter, Clive Randall Group
Version 2.1.1
Last updated: 2017-04-11

Changelog:
v2.1.2
    Corrected the calculation of tandelta in def z_to_tand(r,x): added multiplication with -1; TBayer
v2.1.1
    Added `detect_peaks` function from demotu's github
v2.1.0
    Added `deg_rec` program for in-situ degradation and recovery.
    Added loss tangent to `convert_z` function
v2.0.0
    Cleaned `get_data` function, started to depreciate unused functions
    Added `get_pO2` Nernst equation for oxygen pressure
v1.1.5
    Added convenience functions for find nearest and FWHM
------
v1.1.4
    Modified getdata function to read sweepdevice prompt from new LCR capability.
------
v1.1.3
    Added 'Sample' category to possible sweeps in getdata function.
------
v1.1.2
    Added convert_eps function and optimized the supporting functions.
------
v1.1.1
    Improved getdata function to handle .diel files where GADD was force quit.
------
v1.1.0
    Added convert_z function and its supporting functions.
------
v1.0.9
    Added support for electric field in newivloop function. Also improved
    flexibility of impedancetemp function
------
v1.0.8
    Improved documentation.
------
v1.0.7
    Optimized newivloop function.
------
v1.0.6
    Added newivloop function.
------
v1.0.5
    Added permittivityandloss function.
------
v1.0.4
    impedancetemp function modified now calculates 1000/T(K) column.
------
v1.0.3
    Updated functions ivloop and ageprogression to calculate current density
    instead of current. Note that both functions will default to an area of
    1 m^2 if no area is entered.
------
v1.0.2
    Added impedancetemp function
------
v1.0.1
    Changed impedance function to add a leading zero to two digit temperatures
    in the keys of the returned dictionary
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import os
import time
from scipy.stats import linregress
from collections import namedtuple
e0 = 8.854188e-14 # F/cm (vacuum permittivity)

def getdata(filename):
    '''
    This function reads the .diel file and returns a dictionary with the values
    of each sweep as well as a dictionary with a list of each measurement.

    Parameters
    ----------
    filename : (string)
        Name of .diel file to be imported

    Returns
    -------
    s : (dict)
        A dictionary where the keys are the sweeps and their values are lists
        of the swept parameter
    m : (dict)
        A dictionary where the keys are the thing being measured and the values
        are lists of the measured values

    Examples
    --------
    >>> s,m = diel.getdata('myfile.diel')
    dict s, dict m
    '''
    # initialize values
    az = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter = 0
    s = {}
    m = {}
    H = True
    # open the file
    with open(filename, 'r+b') as f:
        # iterate over every line in the file
        for line in f:
            # remove extra line breaks
            line = line.replace('\r\n', '')
            # Check if we are in the header
            if H is True:
                # Define a new sweep
                if '_set' in line or 'COUNTER' in line or 'SAMPLE' in line:
                    # Name of the sweep
                    sline = line.split('\t')
                    xname = az[letter]+' '+sline[0]
                    # Values that the sweep uses
                    line = f.next().replace('+', '').replace(' ', '')
                    sline = line.split('\t')
                    del sline[-1]   # Last value in list is empty
                    # Make the strings into numbers
                    xdata = [float(x) for x in sline]
                    # Add to the sweep dictionary
                    s[xname] = xdata
                    letter += 1
                # Make a key in the measurement dictionary
                elif 'REAL' in line or 'CMPLX' in line:
                    xname = line.replace('  ', ' ')
                    m[xname] = []
            # What to do when outside of the header
            elif H is False:
                # This line starts a measurement
                if line in m:
                    key = line
                    try:
                        line = f.next()
                    except StopIteration:
                        break
                    # complex
                    if '\t' in line:
                        sline = line.split('\t')
                        realline = float(sline[0])
                        imagline = float(sline[1])
                        xdata = complex(realline, imagline)
                        m[key].append(xdata)
                    # real
                    else:
                        m[key].append(float(line))
                elif 'LIST_REAL_CMPLX SWEEPFREQ' in line:
                    xname = line
                    rows = int(f.next())
                    m[xname] = np.zeros((rows, 3))
                    for i in range(rows):
                        m[xname][i, :] = f.next().split()
            # Are we in the header?
            if '*' in line:
                H = False
    return s, m

def find_nearest(array, value):
    '''
    Find the number in the array closest to the given value.

    Parameters
    ----------
    array : 1d array
        Array to check.

    value : float
        Value to compare.

    Returns
    -------
    closest value to given value in array
    '''
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def find_nearest_index(array, value):
    '''
    Find the index of the array value closest to the given value

    Parameters
    ----------
    array : 1d array
        Array to check.

    value : float
        Value to compare.

    Returns
    -------
    index of closest value to given value in array
    '''
    return (np.abs(array-value)).argmin()

def get_pO2(V, T=900.0):
    '''
    Calculate oxygen partial pressure from sensor voltage. Assumes oxygen is
    measured with a zirconia ceramic electrolyte with a reference p02 of 0.209.
    See http://www.cof.com.au/nernst.shtml for more info.

    Parameters
    ----------
    V : 1d array or float
        Voltage output (in V)

    T : float
        Temperature (in deg. C)

    Returns
    -------
    Partial pressure of oxygen in the furnace (in atm)
    '''
    return 0.209 * np.exp(-46.421*1000*V/(T+273.15))

def rc_to_m(f, sigma, eps, dx=1.0):
    '''
    Calculate real and imaginary modulus spectra from a given conductivity,
    permittivity, and frequency range

    Parameters
    ----------
    f : 1d array
        array of frequency values

    sigma : 1d array or float
        conductivity (S/cm)

    eps : 1d array or float
        permittivity

    dx : float (default 1.0)
        fractional thickness. Should be 1.0 unless simulating multiple
        conductivities in the same sample

    Returns
    -------
    real_m : 1d array
        array of real modulus data that is the same length as the frequency
        array.

    imag_m : 1d array
        array of imaginary modulus data that is the same length as the
        frequency array.
    '''
    omega = np.array(2.0*np.pi*e0*f)[:, np.newaxis]
    denom = (omega * eps)**2 + sigma**2
    real_m = (omega**2 * eps * dx) / denom
    imag_m = (omega * sigma * dx) / denom
    return real_m, imag_m

def m_to_all(real_m, imag_m, f, A, t):
    '''
    Convert real and imaginary modulus data to all other immittance formalisms.

    Parameters
    ----------
    real_m : 1d array
        Voltage output (in V)

    imag_m : 1d array
        Temperature (in deg. C)

    f : 1d array
        Temperature (in deg. C)

    A : float
        Area (cm^2)

    t : float
        thickness (cm)

    Returns
    -------
    real_z : 1d array
        real part of the impedance

    imag_z : 1d array
        imaginary part of the impedance

    real_y : 1d array
        real part of the admittance

    imag_y : 1d array
        imaginary part of the admittance

    real_eps : 1d array
        real part of the permittivity

    imag_eps : 1d array
        imaginary part of the permittivity

    '''
    omega = np.array(2.0*np.pi*f)[:, np.newaxis]
    C0 = e0 * A / t
    real_z = imag_m / (omega * C0)
    imag_z = real_m / (omega * C0)
    denom_y = real_z**2 + imag_z**2
    real_y = real_z / denom_y
    imag_y = imag_z / denom_y
    denom_eps = real_m**2 + imag_m**2
    real_eps = real_m / denom_eps
    imag_eps = imag_m / denom_eps
    return (real_z, imag_z, real_y, imag_y, real_eps, imag_eps)

def z_to_y(r, x):
    r'''
    Convert real and imaginary impedance to complex admittance, according to
    the following formula:

    .. math:: Y^\prime &=\frac{R}{R^2+X^2} \\
              Y^{\prime\prime} &=\frac{X}{R^2+X^2}

    where :math:`Y^\prime` is the real admittance, :math:`Y^{\prime\prime}` is
    the imaginary admittance, :math:`R` is the real part
    of the impedance, and :math:`X` is the imaginary part of the impedance.

    Parameters
    ----------
    r : (1d array)
        Real part of the impedance.
    x : (1d array)
        Imaginary part of the impedance.

    Returns
    -------
    y1 : (1d array)
        Real part of the admittance.
    y2 : (1d array)
        Imaginary part of the admittance.
    '''
    den = np.square(r) + np.square(x)
    y1 = np.divide(r, den) 
    y2 = np.divide(x, den) *-1.0
    return y1, y2

def z_to_m(r, x, f, A, t):
    r'''
    Convert real and imaginary impedance to complex modulus, according to
    the following formula:

    .. math::
      M^\prime &=2\pi f\epsilon_0\frac{A}{t} X \\
      M^{\prime\prime} &=2\pi f\epsilon_0\frac{A}{t} R

    where :math:`M^\prime` is the real modulus, :math:`M^{\prime\prime}` is the
    imaginary modulus, :math:`R` is the real part
    of the impedance, :math:`X` is the imaginary part of the impedance,
    :math:`\epsilon_0` is the vacuum permittivity, :math:`A` is the area
    (in cm^2), and :math:`t` is the thickness (in cm).

    Parameters
    ----------
    r : (1d array)
        Real part of the impedance.
    x : (1d array)
        Imaginary part of the impedance.
    f : (1d array)
        Frequency (in Hz).
    A : (float)
        Area (in cm^2).
    t : (float)
        Thickness (in cm).

    Returns
    -------
    m1 : (1d array)
        Real part of the modulus.
    m2 : (1d array)
        Imaginary part of the modulus.
    '''
    mu = np.multiply(2*np.pi*e0*A/t, f)
    m1 = np.multiply(mu, x) *-1.0
    m2 = np.multiply(mu, r)
    return m1, m2

def z_to_eps(r, x, f, A, t):
    r'''
    Convert real and imaginary impedance to complex permittivity, according to
    the following formula:

    .. math::
      \epsilon^\prime &=\frac{X}{R^2+X^2}\frac{t}{2\pi f\epsilon_0 A} \\
      \epsilon^{\prime\prime} &=\frac{R}{R^2+X^2}\frac{t}{2\pi f\epsilon_0 A}

    where :math:`\epsilon^\prime` is the real permittivity,
    :math:`\epsilon^{\prime\prime}` is the imaginary permittivity,
    :math:`R` is the real part
    of the impedance, :math:`X` is the imaginary part of the impedance,
    :math:`\epsilon_0` is the vacuum permittivity, :math:`A` is the area
    (in cm^2), and :math:`t` is the thickness (in cm).

    Parameters
    ----------
    r : (1d array)
        Real part of the impedance.
    x : (1d array)
        Imaginary part of the impedance.
    f : (1d array)
        Frequency (in Hz).
    A : (float)
        Area (in cm^2).
    t : (float)
        Thickness (in cm).

    Returns
    -------
    eps1 : (1d array)
        Real part of the permittivity.
    eps2 : (1d array)
        Imaginary part of the permittivy.
    '''
    area = A
    thickness = t
    mu = np.divide(
        thickness/(np.add(np.square(r), np.square(x))*2*np.pi*e0*area), f)
    eps1 = np.multiply(mu, x) *-1.0
    eps2 = np.multiply(mu, r)
    return eps1, eps2

def z_to_tand(r,x):
    '''
    Calculate the loss given the complex impedance
    '''
    return -r / x

def convert_z(f, r, x, A, t, per_cm=False):
    '''
    Converts complex impedance data (R, X or Z', Z'') and returns a the complex
    admittance, modulus, and permittivity, as
    well as the inverse frequency for plotting in the time domain.

    Parameters
    ----------
    f : 1d array
        array of frequencies

    r : 1d array
        real part of the impedance

    x : 1d array
        imaginary part of the impedance

    A : float
        Area of sample (in cm^2)

    t : float
        Thickness of sample (in cm)

    per_cm : bool, default False
        If True, report impedance in units of Ohm/cm and admittance in units of
        S/cm

    Returns
    -------
    f : 1d array
        array of frequencies

    inv_f : 1d array
        array of inverse frequencies

    y1 : 1d array
        real part of the admittance

    y2 : 1d array
        imaginary part of the admittance

    z1 : 1d array
        real part of the impedance

    z2 : 1d array
        imaginary part of the impedance

    m1 : 1d array
        real part of the modulus

    m2 : 1d array
        imaginary part of the modulus

    eps1 : 1d array
        real part of the permittivity

    eps2 : 1d array
        imaginary part of the permittivity
    
    tand : 1d array
        Ratio of real to imaginary impedance (tan delta or loss)
    '''
    if per_cm is True:
        geo = (1.0*t)/A
    else:
        geo = 1.0

    inv_f = 1.0 / f
    y1, y2 = z_to_y(r, x)
    y1 *= geo
    y2 *= geo
    z1, z2 = (r, x)
    z1 *= 1.0/geo
    z2 *= 1.0/geo
    m1, m2 = z_to_m(r, x, f, A, t)
    eps1, eps2 = z_to_eps(r, x, f, A, t)
    tand = z_to_tand(r,x)
    return f, inv_f, y1, y2, z1, z2, m1, m2, eps1, eps2, tand

def zview_to_immittance(filenames, A, t, per_cm=False):
    '''
    Converts complex impedance data (R, X or Z', Z'') and returns a the complex
    admittance, modulus, and permittivity, as
    well as the inverse frequency for plotting in the time domain.

    Parameters
    ----------
    filenames : list
        list of filenames to convert. Should have cols of f, R, X.

    A : float
        Area of sample (in cm^2)

    t : float
        Thickness of sample (in cm)

    per_cm : bool, default False
        If True, report impedance in units of Ohm/cm and admittance in units of
        S/cm

    Returns
    -------
    csv files with immittance data from impedance.
    '''
    # Create \Immittance\ directory if it's not there already
    if not os.path.exists(os.path.join(os.getcwd(),'immittance')):
        os.makedirs(os.path.join(os.getcwd(),'immittance'))
    for filename in filenames:
            # Read space-delmited data
            try:
                data = np.loadtxt(filename)
            # Read comma-delimited data
            except ValueError:
                data = np.loadtxt(filename, delimiter=',')
            # Frequency
            f = data[:, 0]
            # Real impedance
            r = data[:, 1]
            # Imaginary impedance
            x = data[:, 2]
            # Get other immittance formalisms
            tup = convert_z(f, r, x, A, t, per_cm)
            # Stack into one array
            out = np.column_stack(tup)
            # Export
            np.savetxt(
                os.getcwd()+'\\immittance\\{}_all.csv'.format(filename[:-4]),
                out, delimiter=',',
                header='f,1/f,realY,imagY,realZ,imagZ,realM,imagM,realE,imagE,tandelta')
            # Write metadata so you can remember the area and thickness
            with open('metadata.dat', 'w') as mf:
                mf.write('Metadata for this data manipulation:\n')
                mf.write('Time: ' + time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
                mf.write('Area = {:f} cm^2\n'.format(A))
                mf.write('Thickness = {:f} cm\n'.format(t))
                mf.write('Z and Y geometry correction: {}\n'.format(per_cm))
                mf.write('File(s) read:\n')
                for item in filenames:
                    mf.write('{}\n'.format(item))

def deg_rec(filename, t, A, per_cm, init_padding, deg_padding, rec_padding,
                leakage, plot_leakage, immittance, plot_m, export_temps,
                plot_temps, init_entry, deg_entry, rec_entry, leakage_entry):
    '''
    Reads degradation and recovery impedance `.diel` file and exports Zview
    files, immittance files, leakage current, and temperature information

    Parameters
    ----------
    filename : string
        .diel file to read

    t : float
        Thickness of sample (in cm)

    A : float
        Area of sample (in cm^2)

    per_cm : bool, default False
        If True, report impedance in units of Ohm/cm and admittance in units of
        S/cm

    init_padding : int
        Padding of exported files.

    deg_padding : int
        Padding of exported files.

    rec_padding : int
        Padding of exported files.

    leakage : bool
        If True, then calculate leakage current at selected frequencies

    plot_leakage : bool
        If True, then plot the leakage data

    immittance : bool
        If True, then calculate immittance data

    plot_m : bool
        If True, then plot imaginary modulus

    export_temps : bool
        If True, then export temperature data

    plot_temps : bool
        If True, then plot the temperature data

    init_entry : int
        Measurement number for initial sweep. `init_entry=0` if scrpt command
        is `check time0`

    deg_entry : string
        Measurement numbers for degradatino sweeps, separated by a space

    rec_entry : string
        Measurement numbers for recovery sweeps, separated by a space

    leakage_entry : string
        Frequencies for leakage measurements, separated by a space. Scientific
        notation (`2e6`) is okay

    Returns
    -------
    Various exported spreadsheets and plots, depending on inputs.
    '''
    s, m = getdata(filename)
    deg_suffix = [int(x) for x in deg_entry.split()]
    rec_suffix = [int(x) for x in rec_entry.split()]
    try:
        init_time = m['REAL TIME{}'.format(init_entry)][0]
    except KeyError:
        init_time = 0.0
    try:
        init_temp = m['REAL TEMPERATURE{}'.format(init_entry)][0]
    except KeyError:
        init_temp = -999
    init_sweep = m['LIST_REAL_CMPLX SWEEPFREQ  RX SWEEPDATA 1']
    deg_time_keys = ['REAL TIME{}'.format(x) for x in deg_suffix]
    deg_temp_keys = ['REAL TEMPERATURE{}'.format(x) for x in deg_suffix]
    rec_time_keys = ['REAL TIME{}'.format(x) for x in rec_suffix]
    rec_temp_keys = ['REAL TEMPERATURE{}'.format(x) for x in rec_suffix]
    leakage_freqs = [float(x) for x in leakage_entry.split()]
    
    deg = {'time_keys' : deg_time_keys,
           'temp_keys' : deg_temp_keys,
           'times' : [],
           'temps' : [],
           'sweeps' : []}
    rec = {'time_keys' : rec_time_keys,
           'temp_keys' : rec_temp_keys,
           'times' : [],
           'temps' : [],
           'sweeps' : []}
    
    for d in [deg, rec]:
        for i in xrange(len(d['time_keys'])):
            try:
                d['times'] += m[d['time_keys'][i]]
            except KeyError:
                pass
            try:
                d['temps'] += m[d['temp_keys'][i]]
            except KeyError:
                pass
    if init_time > 0:
        c = 2
    else:
        c = 1
    deg['times'] = np.array(deg['times']) - init_time
    try:
        rec['times'] = np.array(rec['times']) - init_time - deg['times'][-1]
    except IndexError:
        rec['times'] = np.array(rec['times']) - init_time
    
    for i in xrange(deg['times'].size):
        try:
            deg['sweeps'].append(
                    m['LIST_REAL_CMPLX SWEEPFREQ  RX SWEEPDATA {}'.format(c)])
        except KeyError:
            break
        c += 1
    for i in xrange(rec['times'].size):
        try:
            rec['sweeps'].append(
                    m['LIST_REAL_CMPLX SWEEPFREQ  RX SWEEPDATA {}'.format(c)])
        except KeyError:
            break
        c += 1
    # Save all sweep data
    np.savetxt(filename[:-5]+'_'+'0'.zfill(init_padding)+'_initial.csv',
               init_sweep, delimiter=',')
    for i, data in enumerate(deg['sweeps']):
        np.savetxt(filename[:-5]+'_deg_'+str(int(deg['times'][i])).zfill(deg_padding)+'.csv',
               data, delimiter=',')
    for i, data in enumerate(rec['sweeps']):
        np.savetxt(filename[:-5]+'_rec_'+str(int(rec['times'][i])).zfill(rec_padding)+'.csv',
               data, delimiter=',')
    # Save all immittance data
    if immittance is True:
        # make a list of all .csv files in folder
        rawfile = glob.glob('.\*.csv')
        filenames = [x[2:] for x in rawfile]
        zview_to_immittance(filenames, A, t, per_cm)
    # Plot modulus data
    if plot_m is True:
        # Read modulus, frequency, and time
        deg_immittance = {}
        rec_immittance = {}
        init_immittance = np.array([])
        i_files = glob.glob('immittance\{}*.csv'.format(filename[:-5]))
        for name in i_files:
            for p in name.split('_'):
                try:
                    ti = int(p)
                except ValueError:
                    pass
            if name.find('_deg_') >= 0:
                deg_immittance[ti] = np.loadtxt(name, delimiter=',', skiprows=1,
                                                  usecols=(0, 7))
            elif name.find('_rec_') >= 0:
                rec_immittance[ti] = np.loadtxt(name, delimiter=',', skiprows=1,
                                                  usecols=(0, 7))
            elif name.find('initial') > 0:
                init_immittance = np.loadtxt(name, delimiter=',', skiprows=1,
                                                  usecols=(0, 7))
        
        if len(deg_immittance) > 0:
            deg_cm = cm.viridis_r(np.linspace(0,1,len(deg_immittance)))
            plt.figure()
            plt.title('Degradation')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('M\"')
            if init_immittance.size > 0:
                plt.loglog(init_immittance[:,0], init_immittance[:,1], 'r:', label='initial')
            for i,k in enumerate(sorted(deg_immittance)):
                if i is 0 or i is len(deg_immittance)-1:
                    plt.loglog(deg_immittance[k][:, 0], deg_immittance[k][:, 1],
                           c=deg_cm[i], label=str(k))
                else:
                    plt.loglog(deg_immittance[k][:, 0], deg_immittance[k][:, 1],
                           c=deg_cm[i])
            plt.legend()
            plt.show()
        
        if len(rec_immittance) > 0:
            rec_cm = cm.viridis_r(np.linspace(0,1,len(rec_immittance)))
            plt.figure()
            plt.title('Recovery')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('M\"')
            if init_immittance.size > 0:
                plt.loglog(init_immittance[:,0], init_immittance[:,1], 'r:', label='initial')
            for i,k in enumerate(sorted(rec_immittance)):
                if i is 0 or i is len(rec_immittance)-1:
                    plt.loglog(rec_immittance[k][:, 0], rec_immittance[k][:, 1],
                           c=rec_cm[i], label=str(k))
                else:
                    plt.loglog(rec_immittance[k][:, 0], rec_immittance[k][:, 1],
                           c=rec_cm[i])
            plt.legend()
            plt.show()
    # Save leakage current
    if leakage is True:
        close_freq = [find_nearest_index(data[:,0], x) for x in leakage_freqs]
        if per_cm is True:
            geo = 1.0
        else:
            geo = (1.0*t)/A
        deg_leakage = {}
        rec_leakage = {}
        l_files = glob.glob('immittance\{}*.csv'.format(filename[:-5]))
        for name in l_files:
            for p in name.split('_'):
                try:
                    ti = int(p)
                except ValueError:
                    pass
            leak_data = np.loadtxt(name, delimiter=',', skiprows=1,
                                                  usecols=(2,))[close_freq] * geo
            if name.find('_deg_') >= 0:
                deg_leakage[ti] = leak_data
            elif name.find('_rec_') >= 0:
                rec_leakage[ti] = leak_data
            elif name.find('initial') > 0:
                deg_leakage[-1] = leak_data
                rec_leakage[-1] = leak_data
        if len(deg_leakage) > 0:
            deg_leak_df = pd.DataFrame.from_dict(deg_leakage, orient='index')
            deg_leak_df.sort_index(inplace=True)
            deg_leak_df.columns = data[close_freq, 0]
            deg_leak_df.to_csv(filename[:-4]+'leakage_deg.dat')
        if len(rec_leakage) > 0:
            rec_leak_df = pd.DataFrame.from_dict(rec_leakage, orient='index')
            rec_leak_df.sort_index(inplace=True)
            rec_leak_df.columns = data[close_freq, 0]
            rec_leak_df.to_csv(filename[:-4]+'leakage_rec.dat')
    # Plot leakage current
    if plot_leakage is True:
        if len(deg_leakage) > 0:
            plt.figure()
            plt.title('Degradation Leakage')
            plt.xlabel('Time (s)')
            plt.ylabel('Admittance (S/cm)')
            for col in deg_leak_df.columns:
                plt.semilogy(deg_leak_df.index, deg_leak_df[col], label=col)
            plt.legend()
            plt.show()
        if len(rec_leakage) > 0:
            plt.figure()
            plt.title('Recovery Leakage')
            plt.xlabel('Time (s)')
            plt.ylabel('Admittance (S/cm)')
            for col in rec_leak_df.columns:
                plt.semilogy(rec_leak_df.index, rec_leak_df[col], label=col)
            plt.legend()
            plt.show()
    # Export temperature data
    if export_temps is True:
        if len(deg['temps']) > 0:
            if init_temp > -999:
                deg_temp_data = np.append(init_temp, deg['temps'])
                deg_temp_times = np.append(init_time, deg['times'])
            else:
                deg_temp_data = deg['temps']
                deg_temp_times = deg['times']
            deg_temp_stack = np.column_stack((deg_temp_times, deg_temp_data))
            np.savetxt(filename[:-4]+'_deg_temps.dat',deg_temp_stack, delimiter=',',header='Time(s),Temp(C)')
        if len(rec['temps']) > 0:
            if init_temp > -999:
                rec_temp_data = np.append(init_temp, rec['temps'])
                rec_temp_times = np.append(init_time, rec['times'])
            else:
                rec_temp_data = rec['temps']
                rec_temp_times = rec['times']
            rec_temp_stack = np.column_stack((rec_temp_times, rec_temp_data))
            np.savetxt(filename[:-4]+'_rec_temps.dat',deg_temp_stack, delimiter=',',header='Time(s),Temp(C)')
    
    if plot_temps is True:
        if len(deg['temps']) > 0:
            plt.figure()
            plt.title('Degradation Temperature')
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (deg. C)')
            plt.plot(deg_temp_stack[:,0], deg_temp_stack[:,1])
        if len(rec['temps']) > 0:
            plt.figure()
            plt.title('Recovery Temperature')
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (deg. C)')
            plt.plot(rec_temp_stack[:,0], rec_temp_stack[:,1])

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    return ind

if __name__ == '__main__':
    pass