# -*- coding: utf-8 -*-
"""
Python script for making a video from degradation/recovery measurements
Author: Jared Carter (jjc407@psu.edu)
Last updated 2016-10-31
version 1.1
  *  FFMPEG is invoked directly, so individual frames are not saved.
  *  Animation module used
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation, style
style.use('video')
framerate = 30  # frames / sec (int)
points_per_frame = 5

# Import data
raw_500 = np.loadtxt('B-252_TSDC_poling_tsdc_0500E.csv', delimiter=',', skiprows=1, usecols=[1,3])
raw_1000 = np.loadtxt('B-252_TSDC_poling_tsdc_1000E.csv', delimiter=',', skiprows=1, usecols=[1,3])
raw_1500 = np.loadtxt('B-252_TSDC_poling_tsdc_1500E.csv', delimiter=',', skiprows=1, usecols=[1,3])
raw_2000 = np.loadtxt('B-252_TSDC_poling_tsdc_2000E.csv', delimiter=',', skiprows=1, usecols=[1,3])
raw_temp = np.loadtxt('B-252_TSDC_poling_tsdc_0500E.csv', delimiter=',', skiprows=1, usecols=[0])
raw = np.dstack((raw_500, raw_1000, raw_1500, raw_2000))

my_coords = {'temp': raw_temp,
             'id': ['T', 'J'],
             'field': [500.0, 1000.0, 1500.0, 2000.0]}

tsdc = xr.DataArray(raw, coords=my_coords, dims=['temp', 'id', 'field'])
x = tsdc.sel(id='J') * 1e9
tsdc.loc[dict(id='J')] = x
# Get number of frames
num_points = tsdc['temp'].size * tsdc['field'].size
num_frames = num_points // points_per_frame + num_points % points_per_frame
#%%
# Initialize figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis([30, 400, 1, -12])
ax.set_xlabel(u'Temperature (°C)')
ax.set_ylabel(r'Current Density (nA/cm$^2$)')
E500, = ax.plot([], [], 'o', label='0.5 kV/mm')
E1000, = ax.plot([], [], 's', label='1 kV/mm')
E1500, = ax.plot([], [], '^', label='1.5 kV/mm')
E2000, = ax.plot([], [], 'd', label='2 kV/mm')
ax.legend(loc='upper right')


def init():
    '''
    Initialization function so the animation module knows what to update
    with each frame
    '''
    E500.set_data([], [])
    E1000.set_data([], [])
    E1500.set_data([], [])
    E2000.set_data([], [])
    return (E500, E1000, E1500, E2000)


def data_gen():
    '''Data generator updates the data each time a frame is drawn'''
    # These variables are not directly passed into this function
    global tsdc, num_frames
    i = 0
    while i <= num_frames:  # Returns here
        print str(i)+' of '+str(num_frames)
        data = {}
        if i <= num_frames / 4:
            data['E500_x'] = tsdc.sel(id='T', field=500.0)[:points_per_frame*(i+1)]
            data['E500_y'] = tsdc.sel(id='J', field=500.0)[:points_per_frame*(i+1)]
            data['E1000_x'] = np.array([])
            data['E1000_y'] = np.array([])
            data['E1500_x'] = np.array([])
            data['E1500_y'] = np.array([])
            data['E2000_x'] = np.array([])
            data['E2000_y'] = np.array([])
        elif i <= num_frames / 2:
            data['E500_x'] = tsdc.sel(id='T', field=500.0)
            data['E500_y'] = tsdc.sel(id='J', field=500.0)
            data['E1000_x'] = tsdc.sel(id='T', field=1000.0)[:points_per_frame*(i+1-(num_frames / 4))]
            data['E1000_y'] = tsdc.sel(id='J', field=1000.0)[:points_per_frame*(i+1-(num_frames / 4))]
            data['E1500_x'] = np.array([])
            data['E1500_y'] = np.array([])
            data['E2000_x'] = np.array([])
            data['E2000_y'] = np.array([])
        elif i <= num_frames / 2 + num_frames / 4:
            data['E500_x'] = tsdc.sel(id='T', field=500.0)
            data['E500_y'] = tsdc.sel(id='J', field=500.0)
            data['E1000_x'] = tsdc.sel(id='T', field=1000.0)
            data['E1000_y'] = tsdc.sel(id='J', field=1000.0)
            data['E1500_x'] = tsdc.sel(id='T', field=1500.0)[:points_per_frame*(i+1-(num_frames / 2))]
            data['E1500_y'] = tsdc.sel(id='J', field=1500.0)[:points_per_frame*(i+1-(num_frames / 2))]
            data['E2000_x'] = np.array([])
            data['E2000_y'] = np.array([])
        else:
            data['E500_x'] = tsdc.sel(id='T', field=500.0)
            data['E500_y'] = tsdc.sel(id='J', field=500.0)
            data['E1000_x'] = tsdc.sel(id='T', field=1000.0)
            data['E1000_y'] = tsdc.sel(id='J', field=1000.0)
            data['E1500_x'] = tsdc.sel(id='T', field=1500.0)
            data['E1500_y'] = tsdc.sel(id='J', field=1500.0)
            data['E2000_x'] = tsdc.sel(id='T', field=2000.0)[:points_per_frame*(i+1-(num_frames / 2 + num_frames / 4))]
            data['E2000_y'] = tsdc.sel(id='J', field=2000.0)[:points_per_frame*(i+1-(num_frames / 2 + num_frames / 4))]
        yield data  # Pauses here until the function is called again
        i += 1


def animate(data):
    '''Set the values in the plot'''
    
    E500.set_data(data['E500_x'], data['E500_y'])
    E1000.set_data(data['E1000_x'], data['E1000_y'])
    E1500.set_data(data['E1500_x'], data['E1500_y'])
    E2000.set_data(data['E2000_x'], data['E2000_y'])
    return (E500, E1000, E1500, E2000)


# Make the animation
anim = animation.FuncAnimation(fig, animate, data_gen, init_func=init,
                               blit=True, save_count=num_frames, interval=1000/framerate)
# Save the animation
anim.save('video.mp4',
          extra_args=['-framerate', str(framerate), '-vcodec', 'libx264',
                      '-r', '30'])
