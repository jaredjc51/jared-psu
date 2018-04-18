# -*- coding: utf-8 -*-
"""
Python script for reading DLTS data. An example charge vs. time trace is
animated.

Author: Jared Carter, Clive Randall Group
Last update 2018-04-04
version 1.0
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation, style
style.use('video')
framerate = 30  # frames / sec (int)
# Import data
raw = np.loadtxt('dlts.txt')
raw_data = raw[1:, 1:]

my_coords = {'time': (raw[1:, 0] - 200e-6) * 1e6,
             'temp': raw[0, 1:] + 273.15}

dlts = xr.DataArray(raw_data * 1e9, coords=my_coords, dims=['time', 'temp'])
dlts = dlts.isel(temp=40)
x = dlts.sel(time=slice(-250.0,200.0))
# Get number of frames
num_frames = x['time'].size

# Initialize figure
fig = plt.figure()
ax = fig.add_subplot(121)
ax.axis([-240, 240, -0.5, 15])
ax.set_xlabel(u'Time (µs)')
ax.set_ylabel('Charge (nC)')
curr_plot, = ax.plot([], [], lw=2)


def init():
    '''
    Initialization function so the animation module knows what to update
    with each frame
    '''
    curr_plot.set_data([], [])
    return curr_plot,


def data_gen():
    '''Data generator updates the data each time a frame is drawn'''
    # These variables are not directly passed into this function
    global x, num_frames
    i = 0
    while i <= num_frames:  # Returns here
        print str(i)+' of '+str(num_frames)
        data = {}
        data['curr_x'] = x['time'][slice(0,i+1)].values
        data['curr_y'] = x[slice(0,i+1)].values
        yield data  # Pauses here until the function is called again
        i += 1


def animate(data):
    '''Set the values in the plot'''
    curr_plot.set_data(data['curr_x'], data['curr_y'])  # Current meas.
    return curr_plot,


# Make the animation
anim = animation.FuncAnimation(fig, animate, data_gen, init_func=init,
                               blit=True, save_count=num_frames, interval=1000/framerate)
# Save the animation
anim.save('dlts_one_sweep.mp4',
          extra_args=['-framerate', str(framerate), '-vcodec', 'libx264',
                      '-r', '30'])
