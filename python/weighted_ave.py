# -*- coding: utf-8 -*-

"""
Weighted averager of epochs.

input:
raw filename
category id (trigger)
weights (n_epochs)

output:
weighted average (matrix?)



-compute chpi SNR at each epoch
    -mean/median/? of coils
    -relative to SNR_0
    -time average across epoch
    -for grads/mags separately
    -plot at each epoch?
-weighted average of epochs (by relative SNR)


situation where chpi snr is good but brain snr bad?
get brain snr more directly (by e.g. function of distance?)


@author: jussi
"""

import numpy as np
import mne
from mne import io
from mne.datasets import sample
import sys

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
#event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5

#raw_fname = '/home/jussi/BLA_loppp01R.fif'

# Select events to extract epochs from.
event_id = {'Auditory/Left': 1} # , 'Auditory/Right': 2}

#   Setup for reading the raw data
raw = io.Raw(raw_fname)
#events = mne.read_events(event_fname)
events = mne.find_events(raw)

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # set bads

# pick EEG and MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=True,
                       include=include, exclude='bads')
# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(eeg=80e-6, eog=150e-6),
                    preload=True)

# manual average
ea = epochs.get_data()
eav = np.mean(ea,0)

# almost the same result (detrending etc.?)
evar = epochs.average().data


# wip : create data matrix from ea, then call:
epochs._evoked_from_epoch_data()
# to get evoked object w/ weighted average


sys.exit()









# Plot epochs.
epochs.plot(title='Auditory left/right')

# Look at channels that caused dropped events, showing that the subject's
# blinks were likely to blame for most epochs being dropped
epochs.drop_bad_epochs()
epochs.plot_drop_log(subject='sample')

# Average epochs and get evoked data corresponding to the left stimulation
evoked = epochs['Left'].average()

evoked.save('sample_audvis_eeg-ave.fif')  # save evoked data to disk



