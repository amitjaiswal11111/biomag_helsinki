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
import matplotlib.pyplot as plt




def chpi_freqs(info):
    """ Get chpi frequencies from info dict (e.g. raw.info) """
    for coil in info['hpi_meas'][0]['hpi_coils']:
        yield coil['coil_freq'][0]


def chpi_snr_epochs(epochs, n_lineharm=2, channels='grad', hpi_coil='median'):
    """ Return estimated continuous HPI SNR for each epoch in epochs
    (mne.Epochs object). SNR estimation is done by fitting a GLM to the data,
    comparing to residual (unexplained) variance and averaging the resulting
    SNR across channels.
    Parameters:
    epochs: mne.epochs instance with a single condition
    n_lineharm: number of line frequency harmonics to use in the GLM
    channels: 'grad', 'mag', or 'all': which channels to consider in SNR
    averaging; 'all' returns a weighted average as (2/3)*snr_grad + (1/3)*snr/mag
    hpi_coil: 'best', 'worst', 'median': which hpi coil to track SNR for.
    'best' selects the overall best coil (one with highest average SNR),
    'median' selects the
    TODO:
    handling of bad channels
    avg snr according to N of channel types
    don't model slope, dc (epochs detrended)
    """

    if len(epochs.event_id) > 1:
        raise ValueError('Epochs object should contain one category only')

    alldata = epochs.pick_types(meg=True).get_data()    
    nepochs, nchan, buflen = alldata.shape
    sfreq = epochs.info['sfreq']
    t = np.linspace(0, buflen/sfreq, endpoint=False, num=buflen)
    cfreqs = list(chpi_freqs(epochs.info))
    ncoils = len(cfreqs)
    linefreq = raw.info['line_freq']
    linefreqs = (np.arange(n_lineharm+1)+1) * linefreq
    
    # gradiometer and magmeter indices
    pick_meg = mne.pick_types(epochs.info, meg=True)
    pick_mag = mne.pick_types(epochs.info, meg='mag')
    pick_grad = mne.pick_types(epochs.info, meg='grad')

    # create linear model    
    model = np.c_[t, np.ones(t.shape)]  # model slope and DC
    for f in list(linefreqs)+cfreqs:  # add sine and cosine term for each freq
        model = np.c_[model, np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)]
    inv_model = np.linalg.pinv(model)

    # loop through epochs
    snr_avg_grad = np.zeros([ncoils, nepochs])
    snr_avg_mag = np.zeros([ncoils, nepochs])
    resid_vars = np.zeros([nchan, nepochs])

    for epn in range(nepochs):
        epdata = alldata[epn,:,:].transpose()
        coeffs = np.dot(inv_model, epdata)
        coeffs_hpi = coeffs[2 + 2*len(linefreqs):]
        resid_vars[:,epn] = np.var(epdata - np.dot(model, coeffs), 0)
        # get total hpi amplitudes by combining sine and cosine terms
        hpi_amps = np.sqrt(coeffs_hpi[0::2,:]**2 + coeffs_hpi[1::2,:]**2)
        # divide average HPI power by average variance
        snr_avg_grad[:,epn] = np.divide((hpi_amps**2/2)[:,pick_grad].mean(1),resid_vars[pick_grad,epn].mean())
        snr_avg_mag[:,epn] = np.divide((hpi_amps**2/2)[:,pick_mag].mean(1),resid_vars[pick_mag,epn].mean())

    snr_grad, snr_mag = np.median(snr_avg_grad,axis=0), np.median(snr_avg_mag,axis=0)
    return snr_grad, snr_mag


def weighted_ave(epochs, weights):
    """  Compute weighted average of epochs. epochs is a mne.Epochs object.
    weights is a list or 1-d numpy array with leading dim of n_epochs. """
    weights = np.array(weights)
    n_epochs = len(epochs)
    if not len(weights) == n_epochs:
        raise Exception('Need as many weights as epochs')
    w_ = weights.squeeze()[:,np.newaxis,np.newaxis]  # reshape for broadcasting
    epw = epochs.get_data() * w_ / np.sum(w_) # normalize
    epw_av = np.sum(epw, axis=0)
    return epochs._evoked_from_epoch_data(epw_av, epochs.info, None, n_epochs, FIFF.FIFFV_ASPECT_AVERAGE)
   



chpi_raw_fname = '/home/jussi/Dropbox/bad_203_am_raw.fif'

#   Setup for reading the raw data
raw = io.Raw(chpi_raw_fname, allow_maxshield=True)
events = mne.find_events(raw, stim_channel='STI101')

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
#include = []  # or stim channels ['STI 014']
#raw.info['bads'] = ['MEG 2443', 'EEG 053']  # set bads

picks = mne.pick_types(raw.info, meg=True)
event_id={'Eka': 1}
tmin, tmax = -0.2, 0.8
chpi_epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), picks=picks, preload=True)




snr_grad, snr_mag = chpi_snr_epochs(chpi_epochs)



sss_raw_fname = '/home/jussi/Dropbox/bad_203_am_raw_sss.fif'
raw = io.Raw(sss_raw_fname, allow_maxshield=True)
events = mne.find_events(raw, stim_channel='STI101')

picks = mne.pick_types(raw.info, meg=True)
event_id={'Eka': 1}
tmin, tmax = -0.2, 0.8
sss_epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), picks=picks, preload=True)

weights = snr_grad
ev = weighted_ave(sss_epochs, weights)
ev.savgol_filter(60)
ev.plot()
plt.figure()
ev0 = sss_epochs.average()
ev0.savgol_filter(60)
ev0.plot()







