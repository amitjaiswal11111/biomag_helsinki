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
from mne.io.constants import FIFF
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

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), reject=dict(eeg=80e-6, eog=150e-6),
                    preload=True)





def chpi_snr_epochs(epochs, linefreqs):
    """ Return estimated continuous HPI SNR for each epoch in epochs
    (mne.Epochs object). SNR estimation is done by fitting a GLM to the data. """

    alldata = epochs.get_data()    
    buflen = data.shape[2]
    nepochs = len(epochs)
    sfreq = epochs.info['sfreq']
    t = np.linspace(0, buflen/sfreq, endpoint=False, num=buflen)
    cfreqs = []    
    if len(raw.info['hpi_meas']) > 0 and 'coil_freq' in raw.info['hpi_meas'][0]['hpi_coils'][0]:
        for coil in epochs.info['hpi_meas'][0]['hpi_coils']:
            cfreqs.append(coil['coil_freq'][0])
    else:
        raise Exception('Cannot determine cHPI frequencies from epoch data info')
    ncoils = len(cfreqs)

    # create linear model    
    model = np.c_[t, np.ones(t.shape)]  # model slope and DC
    for f in list(linefreqs)+cfreqs:  # add sine and cosine term for each freq
        model = np.c_[model, np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)]
    inv_model = np.linalg.pinv(model)

    snr_avg_grad = np.zeros([ncoils, nepochs])
    snr_avg_mag = np.zeros([ncoils, nepochs])
    resid_vars = np.zeros([306, nepochs])
    ind = 0
        
    for ep in range(nepochs):
        epdata = alldata[ep, :, :].transpose()
        coeffs = np.dot(inv_model, epdata)
        coeffs_hpi = coeffs[2+2*len(linefreqs):]
        resid_vars[:,ind] = np.var(epdata - np.dot(model, coeffs), 0)
        # get total hpi amplitudes by combining sine and cosine terms
        hpi_amps = np.sqrt(coeffs_hpi[0::2,:]**2 + coeffs_hpi[1::2,:]**2)
        # divide average HPI power by average variance
        snr_avg_grad[:,ind] = np.divide((hpi_amps**2/2)[:,grad_ind].mean(1),resid_vars[grad_ind,ind].mean())
        snr_avg_mag[:,ind] = np.divide((hpi_amps**2/2)[:,mag_ind].mean(1),resid_vars[mag_ind,ind].mean())
        ind += 1

        
        
    


    
def weighted_ave(epochs, weights):
    """  Compute weighted average of epochs. epochs is a mne.Epochs object.
    weights is a list or 1-d numpy array with leading dim of n_epochs. """
    weights = np.array(weights)
    n_epochs = len(epochs)
    if not len(weights) == n_epochs:
        raise Exception('Need as many weights as epochs')
    w_ = weights.squeeze()[:,np.newaxis,np.newaxis]  # reshape for broadcasting
    epw = epochs.get_data() * w_  # / np.sum(w_) # normalize?
    epw_av = np.mean(epw, axis=0)
    return epochs._evoked_from_epoch_data(epw_av, epochs.info, None, n_epochs, FIFF.FIFFV_ASPECT_AVERAGE)

    
    
    
w = np.random.rand(52)    
w = np.ones(52)
av_w = weighted_ave(epochs, w)
av_mne = epochs.average()



    
    
    
sys.exit()
    
    




# manual average
data = epochs.get_data()  # epochs x channels x times
# weighting matrix needs dims of n_epochs x 1 x 1 according to NumPy broadcasting rules
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



