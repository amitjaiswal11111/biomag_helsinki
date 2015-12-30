# -*- coding: utf-8 -*-
"""
Combine gradiometer data from evoked responses.
Plot combined responses in topographic plot (XPlotter style)
Find peak amplitudes.

@author: jussi
"""


from __future__ import print_function


import matplotlib

import matplotlib.pyplot as plt
import sys
import mne
import mne.viz
import numpy as np
from scipy import signal
from mne.channels.layout import _merge_grad_data, _pair_grad_sensors

USAGE = """
Usage: gradi_vectorplot.py evoked_file condition [lowpass]

 evoked_file  name of fiff evoked file
condition    name of category, e.g. "Auditory right"
lowpass      (optional) lowpass corner frequency
"""

#
#if len(sys.argv) not in [3,4]:
#    sys.exit(USAGE)
#
#print(sys.argv[0])
#
#evoked_file = sys.argv[1]
#condition = sys.argv[2]
#if len(sys.argv) == 4:
#    lowpass = float(sys.argv[3])
#else:
#    lowpass = None
#        

BUTTORD = 5  # order of Butterworth IIR lowpass filter

#
## for testing
evoked_file = 'ekmultimodal02_raw_trans_tsss09_MEGrejoff_EOGrejoff_ave.fif'
condition = 'Auditory right'
lowpass = 60

# read evoked data
evoked = mne.read_evokeds(evoked_file, condition=condition)

# filter and replace data in-place
if lowpass:
    data = evoked.data
    sfreq = evoked.info['sfreq']
    lpfreqn = 2 * np.array(lowpass) / sfreq
    b, a = signal.butter(BUTTORD, lpfreqn)
    evoked.data = signal.filtfilt(b, a, data)

# compute gradiometer RMS data
picks_gradc = _pair_grad_sensors(evoked.info, topomap_coords=False)
gradc_data = _merge_grad_data(evoked.data[picks_gradc])
# create list of combined channel names (e.g. MEG 204x)
ch_names_gradc = list()
for ind in picks_gradc:
    ch = evoked.ch_names[ind]
    if ch[-1] == '3':
        ch_names_gradc.append(ch[:-1]+'x')
    
# create evoked set with magnetometer channels only
evoked_mag = evoked.pick_types(meg='mag', copy=True)

# change channel names of evoked set to represent gradiometer pairs (e.g. MEG 2041 -> MEG 204x)
for i,nm in enumerate(evoked_mag.ch_names):
    evoked_mag.ch_names[i] = nm[:-1]+'x'
for i,ch in enumerate(evoked_mag.info['chs']):
    evoked_mag.info['chs'][i]['ch_name'] = evoked_mag.info['chs'][i]['ch_name'][:-1]+'x'
    evoked_mag.info['chs'][i]['unit'] = 201  # change unit to T/m
   
# get magnetometer layout for platting, replace magnetometer names as above
laym = mne.channels.read_layout('Vectorview-mag')
for i,nm in enumerate(laym.names):
    laym.names[i] = nm[:-1]+'x'
    
# replace magnetometer data array with combined gradiometer data
evoked_mag.data = np.zeros(evoked_mag.data.shape)
for j,ch in enumerate(ch_names_gradc):
    for i,ch1 in enumerate(evoked_mag.ch_names):
        if ch1 == ch:
            #print(ch1,'matches',ch,'at index',i)
            evoked_mag.data[i,:] = gradc_data[j,:]

# get peak
pch, plat = evoked_mag.get_peak()
print('Peak amplitude: channel pair',pch,'at latency',plat*1e3,'ms')

# interactive plot    
#plt.ion()

mne.viz.plot_evoked_topo(evoked_mag, layout=laym, title=evoked_file)


