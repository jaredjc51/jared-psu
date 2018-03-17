# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('thesis')
save_figures = True
e0 = 8.85418782e-12 # F/m
c_cgs = 29979245800.0

def tunability_bto(E, T, eps_low):
    xi_11 = 4.69986e-15 * (T-273.15) - 8.2597e-13
    return ((-3*eps_low**4*xi_11) / (4*np.pi)**3) * E**2
def curie_weiss(T, c, theta):
    '''Curie-Weiss permittivity relation'''
    return c/(T-theta)
def G_srtio3(Px, Py, Pz, qx, qy, qz, T):
    '''Landau free energy coefficients for STO'''
    alpha_1 = 7.06e5*(T-35.5)
    alpha_11 = 1.70e9
    alpha_12 = 1.37e9
    beta_1 = 1.32e29*((1.0/np.tanh(145.0/T))-(1.0/np.tanh(145.0/105.0)))
    beta_11 = 0.996e50
    beta_12 = 2.73e50
    t_11 = -2.10e29
    t_12 = -0.845e29
    t_44 = 5.85e29
    return (alpha_1*(Px**2+Py**2+Pz**2)
            + alpha_11*(Px**4+Py**4+Pz**4)
            + alpha_12*(Px**2*Py**2+Py**2*Pz**2+Px**2*Pz**2)
            + beta_1*(qx**2+qy**2+qz**2)
            + beta_11*(qx**4+qy**4+qz**4)
            + beta_12*(qx**2*qy**2+qy**2*qz**2+qx**2*qz**2)
            - t_11*(Px**2*qx**2+Py**2*qy**4+Pz**2*qz**2)
            -t_12*(Px**2*(qy**2+qz**2)+Py**2*(qx**2+qz**2)+Pz**2*(qx**2+qy**2))
            -t_44*(Px*Py*qx*qy+Py*Pz*qy*qz+Pz*Px*qz*qx))

def G_batio3(Px, Py, Pz,T):
    '''Landau free energy coefficients for BTO'''
    alpha_1 = 4.124e5*(T-388)
    alpha_11 = -209.7e6
    alpha_12 = 7.974e8
    alpha_111 = 129.4e7
    alpha_112 = -1.950e9
    alpha_123 = -2.500e9
    alpha_1111 = 3.863e10
    alpha_1112 = 2.529e10
    alpha_1122 = 1.637e10
    alpha_1123 = 1.367e10
    return (alpha_1*(Px**2+Py**2+Pz**2)
            + alpha_11*(Px**4+Py**4+Pz**4)
            + alpha_12*(Px**2*Py**2+Py**2*Pz**2+Px**2*Pz**2)
            + alpha_111*(Px**6+Py**6+Pz**6)
            + alpha_112*(Px**2*(Py**4+Pz**4)+Py**2*(Px**4+Pz**4)+Pz**2*(Px**4+Py**4))
            + alpha_123*Px**2*Py**2*Pz**2
            + alpha_1111*(Px**8+Py**8+Pz**8)
            + alpha_1112*(Px**6*(Py**2+Pz**2)+Py**6*(Px**2+Pz**2)+Pz**6*(Px**2+Py**2))
            + alpha_1122*(Px**4*Py**4+Py**4*Pz**4+Px**4*Pz**4)
            + alpha_1123*(Px**4*Py**2*Pz**2+Py**4*Pz**2*Px**2+Pz**4*Px**2*Py**2))

# Selected Temperatures
Ts = np.array([26.85, 114.85, 210.0, 275.0]) + 273.15
# Polarization range
num_Ps = 10000
Ps = np.linspace(-0.4,0.4,num_Ps)
# Differential polarization
dP = Ps[1] - Ps[0]
# Initialize array for free energy
Gs_sto = np.zeros((Ts.size,Ps.size))
Gs_bto = np.zeros_like(Gs_sto)
for k in range(Ts.size):
    Gs_sto[k,:] = G_srtio3(Ps, 0, 0, 0, 0, 0, Ts[k])
    Gs_bto[k,:] = G_batio3(Ps, 0, 0, Ts[k])
# E = dG/dP
Es_sto = np.gradient(Gs_sto, dP, axis=1)
Es_bto = np.gradient(Gs_bto, dP, axis=1)
#Es_bto_old = np.copy(Es_bto)
#Es_bto_na = np.copy(Es_bto)
#for i in range(2):
#    unstable_bto = np.where(np.diff(np.sign(np.diff(Es_bto[i,2:-2]))))[0] + 2
#    middle = int(Es_bto.shape[1]/2)
#    jump_bto = (np.abs(Es_bto[i,middle:]-Es_bto[i,unstable_bto[0]]).argmin() + middle)
#    Es_bto[i,:] = np.concatenate((Es_bto[i,0:unstable_bto[0]],
#                          np.ones(jump_bto-unstable_bto[0])*Es_bto[i,unstable_bto[0]],
#                          Es_bto[i,jump_bto:]))
#    Es_bto_na[i,:] = np.concatenate((Es_bto_old[i,0:unstable_bto[0]],
#                                     np.ones(unstable_bto[1]-unstable_bto[0])*np.nan,\
#                                     Es_bto_old[i,unstable_bto[1]:]))
# 1/eps = d^2G/dP^2
inv_eps_sto = e0 * np.gradient(Es_sto, dP, axis=1)
inv_eps_bto = e0 * np.gradient(Es_bto, dP, axis=1)
#%% Plot G vs P and Polarization vs. E
fig = plt.figure(figsize=(6,3))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for k in range(Ts.size):
    ax1.plot(Ps*1e-4*1e6,Gs_sto[k,:]*1e-6, label=u'{:03d}°C'.format(int(Ts[k]-273.15)))
    ax2.plot(Es_sto[k,2:-2]*1e-2*1e-3, Ps[2:-2]*1e-4*1e6, label=u'{:03d}°C'.format(int(Ts[k]-273.15)))
ax1.set_xlabel(u'Polarization (µC/cm$^2$)')
ax1.set_ylabel('G (MJ)')
ax1.annotate('a)', xy=(0.02, 0.85),
             xycoords='figure fraction')
ax2.set_xlabel('Field (kV/cm)')
ax2.set_ylabel(u'Polarization (µC/cm$^2$)')
ax2.annotate('b)', xy=(0.52, 0.85),
             xycoords='figure fraction')
ax1.legend()
if save_figures is True:
    plt.savefig('STO-energy-polarization-field.pdf')
plt.show()
#%% Plot permittivity vs E
plt.figure()
for k in range(Ts.size):
    plt.plot(Es_sto[k,2:-2]*1e-2*1e-3,1.0/inv_eps_sto[k,2:-2], label=u'{:03d}°C'.format(int(Ts[k]-273.15)))
plt.plot(np.zeros_like(Ts), curie_weiss(Ts, 8.5e4, 17), 'ko', label='Linz et al.')
plt.xlabel('E field (kV/cm)')
plt.ylabel('Relative Permittivity')
plt.axis([-0.05,1,125,325])
plt.legend()
if save_figures is True:
    plt.savefig('STO-permittivity-field-zoom.pdf')
plt.show()


#%%#################################################
# Plot G vs P BARIUM TITANATE and Polarization vs. E
####################################################
fig = plt.figure(figsize=(6,3))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for k in range(Ts.size):
    ax1.plot(Ps*1e-4*1e6,Gs_bto[k,:]*1e-6, label=u'{:03d}°C'.format(int(Ts[k]-273.15)))
    ax2.plot(Es_bto[k,2:-2]*1e-2*1e-3, Ps[2:-2]*1e-4*1e6, label=u'{:03d}°C'.format(int(Ts[k]-273.15)))
ax1.set_xlabel(u'Polarization (µC/cm$^2$)')
ax1.set_ylabel('G (MJ)')
ax1.annotate('a)', xy=(0.02, 0.85),
             xycoords='figure fraction')
ax2.set_xlabel('Field (kV/cm)')
ax2.set_ylabel(u'Polarization (µC/cm$^2$)')
ax2.annotate('b)', xy=(0.52, 0.85),
             xycoords='figure fraction')
ax1.legend()
if save_figures is True:
    plt.savefig('BTO-energy-polarization-field.pdf')
plt.show()
#%% Plot permittivity vs E
plt.figure()
for k in range(Ts.size):
    plt.plot(Es_bto[k,int(Es_bto.shape[1]/2)-100:-2]*1e-2*1e-3,1.0/inv_eps_bto[k,int(Es_bto.shape[1]/2)-100:-2], label=u'{:03d}°C'.format(int(Ts[k]-273.15)))
plt.xlabel('E field (kV/cm)')
plt.ylabel('Relative Permittivity')
plt.axis([-0.05,1,0,1500])
plt.legend()
if save_figures is True:
    plt.savefig('BTO-permittivity-field-zoom.pdf')
plt.show()

#%% Tunability of Barium Titanate
plt.figure()
x = np.linspace(0,60,1000)
for i in range(Ts.size):
    eps_no_field = 1.0 / inv_eps_bto[i, np.argmin(Gs_bto[i,:])]
    plt.plot(x*c_cgs/1e11, tunability_bto(x, Ts[i], eps_no_field)/eps_no_field*100,label=u'{:03d}°C'.format(int(Ts[i]-273.15)))
plt.legend()
plt.xlabel('Field (kV/cm)')
plt.ylabel(r'Change in Permittivity (%)')
if save_figures is True:
    plt.savefig('BTO-tunability.pdf')
plt.show()