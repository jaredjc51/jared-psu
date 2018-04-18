# -*- coding: utf-8 -*-
"""
Functions to accompany the Complex modulus fit notebook
Author: Jared Carter, Clive Randall Group
Last Updated 2018-04-18

v2.2
----
 - Modified appearance of residual plots
 - Fit permittivity as well as conductivities

v2.1
----
 - Modified `read_modulus_data` function to read excel files
 - Updated function documentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
e0 = 8.854188e-14 #F/cm (vacuum permittivity)

def read_modulus_data(p):
    '''
    Return frequency and modulus.
    '''
    if p['data_format'] is 'PSU':
        data = np.loadtxt(p['filename'], delimiter=',', skiprows=1,
                          usecols=[0, 6, 7])

    elif p['data_format'] is 'NCSU':
        raw_data = pd.read_excel(p['filename'], skiprows=4, parse_cols=[0,8,9])
        data = raw_data.values

    return data

def plot_modulus(p):
    '''Plot experimental modulus data'''
    fig = plt.figure(figsize=(6,8))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.loglog(p['inv_data'][:,0], p['inv_data'][:,1], c='#BBBBBB', marker='.', ls='none')
    ax1.loglog(p['sel_data'][:,0], p['sel_data'][:,1], c='#000000', marker='.', ls='none')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('real M')
    ax2.loglog(p['inv_data'][:,0], p['inv_data'][:,2], c='#BBBBBB', marker='.', ls='none')
    ax2.loglog(p['sel_data'][:,0], p['sel_data'][:,2], c='#000000', marker='.', ls='none')
    ax2.loglog(np.array([p['sigma_lo'], p['sigma_hi']])/(2*np.pi*e0*p['permittivity']), [1e-3,1e-3], c='blue', marker='o', ls='none')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('imag M')
    return fig

def plot_result(p):
    '''plot fit result'''
    fig = plt.figure(figsize=(10,6))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(222)
    ax4 = plt.subplot(224)
    # Real modulus
    ax1.loglog(p['sel_data'][:,0], p['sel_data'][:,1], 'k.', alpha=0.5)
    ax1.loglog(p['best_fit'][:,0], p['best_fit'][:,1], 'r')
    ax1.set_title(r'$M^\prime$ fit')
    # Imaginary Modulus
    ax2.loglog(p['sel_data'][:,0], p['sel_data'][:,2], 'k.', alpha=0.5)
    ax2.loglog(p['best_fit'][:,0], p['best_fit'][:,2],'b')
    ax2.loglog(10**p['log_sigmas']/(2*np.pi*e0*p['permittivity']),2.5e-3*np.ones(p['slices']),'bo', alpha=p['slices']/(10.0*p['slices']))
    ax2.set_title(r'$M^{\prime\prime}$ fit')
    # Distribution of conductivities
    ax3.hist(p['log_sigmas'], p['bins'], color='b', alpha=0.7)
    ax3.set_title('Distribution of Conductivities')
    ax3.set_xlabel(r'$\log(\sigma)$')
    ax3.set_ylabel('Count')
    # Residuals
    ax4.semilogx(p['sel_data'][:,0],p['real_residual'], 'r')
    ax4.semilogx(p['sel_data'][:,0],p['imag_residual'], 'b')
    ax4.axhline(color='black') # y=0 line
    ax4.legend(['Real','Imag.'], loc='lower right')
    ax4.set_title(r'(Exp$-$Calc)/Exp')
    return fig

def real_to_bounded(real, lo_bound, hi_bound):
    r'''
    The numbers this function returns can be changed freely by an
    unbounded solver. When they are converted back to the real value by the
    inverse of this function (:code:`bounded_to_real`), they will be in between
    the bounds specified.
    
    .. math::
        P_{\rm bounded} = \arcsin\big(\frac{2 (P_{\rm real} -
        {\rm min})}{({\rm max} - {\rm min})} - 1\big)
     
    for more information on bounds implementation, see 
    https://lmfit.github.io/lmfit-py/bounds.html
    '''
    return np.arcsin((2*(real - lo_bound))/(hi_bound-lo_bound)-1)

def bounded_to_real(bounded, lo_bound, hi_bound):
    r'''
    Convert value optimizer sees back to the actual value.
    
    .. math::
        P_{\rm real}  = {\rm min} + \big(\sin(P_{\rm bounded}) +
        1\big) \frac{({\rm max} - {\rm min})}{2}
     
    for more information on bounds implementation, see 
    https://lmfit.github.io/lmfit-py/bounds.html
    '''
    return lo_bound+(np.sin(bounded)+1)*(hi_bound-lo_bound)/2



def fit_eps_sigmas_residual(bound_inputs, inputs_lo, inputs_hi, data):
    '''
    Determine residual of calcualted modulus data
    '''
    eps = np.power(10, bounded_to_real(bound_inputs[0], np.log10(inputs_lo[0]), np.log10(inputs_hi[0])))
    sigmas = np.power(10, bounded_to_real(bound_inputs[1:], np.log10(inputs_lo[1:]), np.log10(inputs_hi[1:])))
    freq = data[:,0]
    calc_data = complex_modulus(sigmas, freq, eps)
    real_residual = np.log10(data[:,1]) - np.log10(calc_data[:,1])
    imag_residual = np.log10(data[:,2]) - np.log10(calc_data[:,2])
    return np.append(real_residual, imag_residual)


def complex_modulus(sigmas, freq, eps):
    '''
    Given conductivities, frequencies, and permittivity, calculate modulus
    data
    '''
    dx = 1.0/sigmas.size
    omega = 2*np.pi*e0*freq
    prefactor = np.multiply(1.0/(np.array(np.square(omega*eps))[:, np.newaxis] + np.square(sigmas)), np.array(omega*dx)[:, np.newaxis])
    realM = np.sum(prefactor*eps*omega[:,np.newaxis], axis=1)
    imagM = np.sum(prefactor*sigmas[np.newaxis,:], axis=1)
    return np.column_stack((freq, realM, imagM))

def complex_modulus_residual(bound_sigmas, sigma_lo, sigma_hi, data, eps):
    '''
    Determine residual of calcualted modulus data
    '''
    sigmas = np.power(10, bounded_to_real(bound_sigmas, np.log10(sigma_lo), np.log10(sigma_hi)))
    freq = data[:,0]
    calc_data = complex_modulus(sigmas, freq, eps)
    real_residual = np.log10(data[:,1]) - np.log10(calc_data[:,1])
    imag_residual = np.log10(data[:,2]) - np.log10(calc_data[:,2])
    return np.append(real_residual, imag_residual)