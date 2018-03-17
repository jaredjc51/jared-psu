# -*- coding: utf-8 -*-
"""
Calculate the time required for oxygen re-equilibration at a given temperature,
assuming the process is either diffusion limited, surface limited (bare STO),
or surface limited (porous Pt)
"""
import numpy as np
from scipy.stats import linregress
import xarray as xr
import matplotlib.pyplot as plt
plt.style.use('thesis')
k = 8.617e-5 # eV/K
thickness = 0.05 # cm
# Selected temperatures
Cs = np.arange(1125, 20, -25)
Ks = Cs + 273.15
# Selectecd times
times = np.logspace(-2.5 ,12, 1e4)

def surface_reaction(times, rate_constant, L):
    '''
    Percentage of oxygen that has diffused into the sample at time t and the
    temperature of the rate constant for the surface limited case
    '''
    t = xr.DataArray(times, coords=[('time', times)])
    k_eff = xr.DataArray(rate_constant, coords=[('temp', Cs)])
    return 1 - np.exp((-k_eff*t)/(2*L))

def oxygen_diffusion(times, Ds, L, n_max):
    '''
    Percentage of oxygen that has diffused into the sample at time t and the
    temperature of the diffusion coefficient for the diffusion limited case
    '''
    n = xr.DataArray(np.arange(1, n_max+1, 1), dims='n')
    t = xr.DataArray(times, coords=[('time', times)])
    D = xr.DataArray(Ds, coords=[('temp', Cs)])
    summation = (8.0/((2.0*n-1)**2*np.pi**2)
                 * np.exp((-D*(2.0*n-1)**2*np.pi**2*t)/(4*L**2)))
    return 1 - summation.sum(dim='n')

def oxygen_mobility(T):
    '''Mobility of oxygen vacancy at given temperature'''
    return 1.0e4/T*np.exp(-0.86/(k*T))


#%% Diffusion limited case
# Oxygen vacancy mobility
mobilty_VO = oxygen_mobility(Ks)
# Diffusion constant of oxygen vacancy at each temperature (Einstein relation)
D_VO = mobilty_VO * k * Ks * 0.5
# Diffusion limited solution
res_diff = oxygen_diffusion(times, D_VO, thickness/2.0, 50)

#%% Surface reaction limited case
# Calculate effective rate constant by extrapolating this data
# Read file
data = np.genfromtxt('Merkle2006_keff.txt', delimiter='\t')
# Columns of 1000/K
invK = [data[:, b]/1e3 for b in (0, 2, 4, 6, 8, 10)]
# remove empty values
invK = [a[~np.isnan(a)] for a in invK]
# Columns of k_eff
lnkeff = [np.log(data[:, b]) for b in (1, 3, 5, 7, 9, 11)]
# Remove empty values
lnkeff = [a[~np.isnan(a)] for a in lnkeff]
# Labels for list items
label = ['bare', 'UV', 'Ag', 'Pt', 'STO:30Fe', 'YBCO']
# initialize lists
result = []
EA = []
# For every item in the list,
for i in range(len(invK)):
    # Fit the data
    res = linregress(invK[i], lnkeff[i])
    # Add result to list
    result.append(res)
    # Add activation energy to list
    EA.append(res.slope * k)
# Use fit to extrapolate effective rate constants for selected temperatures
# bare STO
k_eff_sto = np.exp(result[0].slope*1/Ks + result[0].intercept)
# Porous Pt
k_eff_pt = np.exp(result[3].slope*1/Ks + result[3].intercept)
# Bare STO solution
res_surf_bare = surface_reaction(times, k_eff_sto, thickness/2.0)
# Porous Pt solution
res_surf_pt = surface_reaction(times, k_eff_pt, thickness/2.0)

#%% Determine when sample will be 5% re-equilibrated
# Initialize result arrays
t_05_diff = np.ones_like(Cs) * np.nan
t_05_surf_bare = np.copy(t_05_diff)
t_05_surf_pt = np.copy(t_05_diff)
# At each temperature,
for i in range(Cs.size):
    idx = [0, 0, 0]
    # Find the index of the value closest to 0.05 for the:
    # Diffusion limited case
    idx[0] = np.abs(res_diff.sel(temp=Cs[i]).data-0.05).argmin()
    # Bare STO case
    idx[1] = np.abs(res_surf_bare.sel(temp=Cs[i]).data-0.05).argmin()
    # Porous Pt case
    idx[2] = np.abs(res_surf_pt.sel(temp=Cs[i]).data-0.05).argmin()
    # Don't plot when the result is the lowest or highest selected time
    if idx[0] > 0 and idx[0] < times.size-1:
        t_05_diff[i] = np.log10(res_diff.time[idx[0]])
    if idx[1] > 0 and idx[1] < times.size-125:
        t_05_surf_bare[i] = np.log10(res_surf_bare.time[idx[1]])
    if idx[2] > 0 and idx[2] < times.size-1:
        t_05_surf_pt[i] = np.log10(res_surf_pt.time[idx[2]])

#%% Plot the results
plt.figure()
plt.plot(Cs, t_05_diff, 'o', label='Diffusion')
plt.plot(Cs, t_05_surf_bare, 's', label=r'Surf. SrTiO$_3$')
plt.plot(Cs, t_05_surf_pt, 'd', label=r'Surf. SrTiO$_3$:Pt')
minute = np.log10(60)
plt.axhline(minute, c='k', ls=':')
plt.annotate('1 minute', xy=(1e3, minute+0.15), xycoords='data')
hour = np.log10(10**minute * 60)
plt.axhline(hour, c='k', ls=':')
plt.annotate('1 hour', xy=(1e3, hour+0.15), xycoords='data')
day = np.log10(10**hour * 24)
plt.axhline(day, c='k', ls=':')
plt.annotate('1 day', xy=(1e3, day+0.15), xycoords='data')
year = np.log10(10**day * 365.25)
plt.axhline(year, c='k', ls=':')
plt.annotate('1 year', xy=(1e3, year+0.15), xycoords='data')
plt.axvline(400, c='k')
plt.axvline(950, c='k')
plt.xlabel(u'Temp. (°C)')
plt.ylabel('log(time to 5% re-equilibration [s])')
plt.legend()
#plt.savefig('re-equilibration time vs temp.png', dpi=300)
plt.show()

#%% Plot re-equilibration percent at 900 C
plt.figure()
plt.plot(times, res_surf_bare.sel(temp=900), label=r'Surf. SrTiO$_3$')
plt.plot(times, res_surf_pt.sel(temp=900), label=r'Surf. SrTiO$_3$:Pt')
plt.plot(times, res_diff.sel(temp=900), label=r'Diffusion')
plt.xlim(0,100)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Fraction re-equlibrated')
#plt.savefig('re-equilibration time at 900.png', dpi=300)
plt.show()