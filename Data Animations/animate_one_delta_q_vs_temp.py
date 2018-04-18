# -*- coding: utf-8 -*-
"""
Python script for reading DLTS data. All charge vs. time traces are
animated and the change in charge vs. temperature is animated for the selected
time window for all temperatures.

Author: Jared Carter, Clive Randall Group
Last update 2018-04-04
version 1.0
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation, style
style.use('video')
framerate = 10  # frames / sec (int)
# Import data
raw = np.loadtxt('dlts.txt')
raw_data = raw[1:, 1:]

my_coords = {'time': (raw[1:, 0] - 200e-6) * 1e6,
             'temp': raw[0, 1:] + 273.15}

dlts = xr.DataArray(raw_data * 1e9, coords=my_coords, dims=['time', 'temp'])
x = dlts.sel(time=slice(-250.0,200.0))
diff = x.isel(time=52) - x.isel(time=70)
# Get number of frames
num_frames = x['temp'].size

# Initialize figure
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.axis([-240, 240, -0.5, 15])
ax1.set_xlabel(u'Time (µs)')
ax1.set_ylabel('Charge (nC)')
charge, = ax1.plot([], [], lw=2, label='current temp.')
b1, = ax1.plot([], [], color=(0.25, 0.25, 0.25), label='5 points cooler')
b2, = ax1.plot([], [], color=(0.75, 0.75, 0.75), label='10 points cooler')
ax1.legend(loc='upper right')
v1, = ax1.plot([], [], color='k', linestyle='--')
v2, = ax1.plot([], [], color='k', linestyle='--')
ax2 = fig.add_subplot(122)
ax2.axis([114, 314, 0, 1.75])
ax2.set_xlabel('Temperature (K)')
ax2.set_ylabel(r'$\Delta$Q (nC)')
delta_c, = ax2.plot([], [], 'o')


def init():
    '''
    Initialization function so the animation module knows what to update
    with each frame
    '''
    b1.set_data([], [])
    b2.set_data([], [])
    charge.set_data([], [])
    v1.set_data([], [])
    v2.set_data([], [])
    delta_c.set_data([], [])
    return (b1, b2, charge, v1, v2, delta_c)


def data_gen():
    '''Data generator updates the data each time a frame is drawn'''
    # These variables are not directly passed into this function
    global x, diff, num_frames
    i = 0
    while i <= num_frames:  # Returns here
        print str(i)+' of '+str(num_frames)
        data = {}
        data['charge_x'] = x['time']
        data['charge_y'] = x.isel(temp=i)
        data['v1_x'] = [8.0, 8.0]
        data['v1_y'] = [0, x.isel(temp=i, time=52)]
        data['v2_x'] = [80.0, 80.0]
        data['v2_y'] = [0, x.isel(temp=i, time=70)]
        data['delta_c_x'] = diff['temp'][:i+1]
        data['delta_c_y'] = diff[:i+1]
        if i <= 5:
            data['b1_x'] = np.array([])
            data['b1_y'] = np.array([])
            data['b2_x'] = np.array([])
            data['b2_y'] = np.array([])
        elif i <= 10:
            data['b1_x'] = x['time']
            data['b1_y'] = x.isel(temp=i-5)
            data['b2_x'] = np.array([])
            data['b2_y'] = np.array([])
        else:
            data['b1_x'] = x['time']
            data['b1_y'] = x.isel(temp=i-5)
            data['b2_x'] = x['time']
            data['b2_y'] = x.isel(temp=i-10)
        yield data  # Pauses here until the function is called again
        i += 1


def animate(data):
    '''Set the values in the plot'''
    b1.set_data(data['b1_x'], data['b1_y'])
    b2.set_data(data['b2_x'], data['b2_y'])
    charge.set_data(data['charge_x'], data['charge_y'])
    v1.set_data(data['v1_x'], data['v1_y'])
    v2.set_data(data['v2_x'], data['v2_y'])
    delta_c.set_data(data['delta_c_x'], data['delta_c_y'])
    return (b1, b2, charge, v1, v2, delta_c)


# Make the animation
anim = animation.FuncAnimation(fig, animate, data_gen, init_func=init,
                               blit=True, save_count=num_frames, interval=1000/framerate)
# Save the animation
anim.save('dlts_one_delta_q.mp4',
          extra_args=['-framerate', str(framerate), '-vcodec', 'libx264',
                      '-r', '30'])
