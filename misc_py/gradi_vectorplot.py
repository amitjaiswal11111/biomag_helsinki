# -*- coding: utf-8 -*-
"""
Combine gradiometer data using MNE-Python.

@author: jussi
"""


import mne
import mne.viz
import numpy as np
from scipy import signal
from mne.channels.layout import _merge_grad_data, _pair_grad_sensors



fname = 'ekmultimodal02_raw_trans_tsss09_MEGrejoff_EOGrejoff_ave.fif'
LPFREQ = 60  # lowpass (Hz)
BUTTORD = 5  # order of Butterworth IIR lowpass filter

condition = 'Auditory right'
evoked = mne.read_evokeds(fname, condition=condition)

# filter and replace data in-place
data = evoked.data
sfreq = evoked.info['sfreq']
lpfreqn = 2 * np.array(LPFREQ) / sfreq
b, a = signal.butter(BUTTORD, lpfreqn)
evoked.data = signal.filtfilt(b, a, data)

# compute gradiometer RMS data
picks_gradc = _pair_grad_sensors(evoked.info, topomap_coords=False)
gradc_data = _merge_grad_data(evoked.data[picks_gradc])
# create list of combined channel names (e.g. MEG 204x)
ch_names_gradc = list()
# change data in-place
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
            print(ch1,'matches',ch,'at index',i)
            evoked_mag.data[i,:] = gradc_data[j,:]
    
# interactive plot    
mne.viz.plot_evoked_topo(evoked_mag, layout=laym)

#evoked.savgol_filter(40)
#evoked.plot(ch_type='grad')

