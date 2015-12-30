# -*- coding: utf-8 -*-
"""
cHPI SNR.

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


buflen = 10000 #samples

filepath = '/home/jussi/Dropbox/megdata/'
meg_fn = '/net/tera2/data/neuro-data/epilepsia/case_4532/151214/LA_loppp01R.fif'

raw = mne.io.Raw(meg_fn, allow_maxshield=True)
sfreq = raw.info['sfreq']

cfreqs = []    
if len(raw.info['hpi_meas']) > 0 and 'coil_freq' in raw.info['hpi_meas'][0]['hpi_coils'][0]:
    for coil in raw.info['hpi_meas'][0]['hpi_coils']:
        cfreqs.append(coil['coil_freq'][0])
else:
    raise Exception('Cannot determine cHPI frequencies from raw data info')

print('\nNominal cHPI frequencies: ', cfreqs, ' Hz')
    

print('Sampling frequency:', sfreq,'Hz')
print('Using buffers of',buflen,'samples =',buflen/sfreq,'seconds')


pick_meg = mne.pick_types(raw.info, meg=True)
pick_mag = mne.pick_types(raw.info, meg='mag')
pick_grad = mne.pick_types(raw.info, meg='grad')
meg_chnames = [raw.ch_names[i] for i in pick_meg]
mag_chnames = [raw.ch_names[i] for i in pick_mag]
grad_chnames = [raw.ch_names[i] for i in pick_grad]
# indices into 306-dim channel matrix
mag_ind = [i for i in range(0,len(meg_chnames)) if meg_chnames[i] in mag_chnames]
grad_ind = [i for i in range(0,len(meg_chnames)) if meg_chnames[i] in grad_chnames]

# create general linear model for the data
t = np.linspace(0,buflen/sfreq,endpoint=False,num=buflen)
model = t   # model slope
model = np.c_[model, np.ones(t.shape)]  # model DC
cfreqs.append(raw.info['line_freq'])  # model also line frequency
for f in cfreqs:  # add sine and cosine term for each freq
    model = np.c_[model, np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)]
inv_model = np.linalg.pinv(model)

# loop thru MEG data
stop = raw.n_times
#stop = 1e5
bufs = range(0, int(stop), buflen)[:-1]
snr_grad = np.zeros([len(cfreqs)-1, len(bufs)])
snr_mag = np.zeros([len(cfreqs)-1, len(bufs)])
amp_grad = np.zeros([len(cfreqs)-1, len(bufs)])
amp_mag = np.zeros([len(cfreqs)-1, len(bufs)])
ind = 0
for buf0 in bufs:  # drop last buffer to avoid overrun
    megbuf = raw[pick_meg, buf0:buf0+buflen][0].transpose()
    coeffs = np.dot(inv_model, megbuf)
    coeffs_hpi = coeffs[2:-2]  # drop slope, dc and power line freq
    resid_var = np.var(megbuf-np.dot(model,coeffs), 0)
    # hpi amps from sine and cosine terms
    hpi_amps = np.sqrt(coeffs_hpi[0::2,:]**2 + coeffs_hpi[1::2,:]**2)
    snr = np.divide(hpi_amps**2/2., resid_var)  # channelwise power snr
    snr_grad[:,ind] = np.mean(snr[:,grad_ind],axis=1)
    snr_mag[:,ind] = np.mean(snr[:,mag_ind],axis=1)
    # RMS amplitudes over grads and mags separately
    amp_mag[:,ind] = np.sqrt(np.sum(hpi_amps[:,mag_ind]**2, 1))
    amp_grad[:,ind] = np.sqrt(np.sum(hpi_amps[:,grad_ind]**2, 1))
    ind += 1

tvec = np.array(bufs)/sfreq
plt.figure()
plt.plot(tvec, 10*np.log10(snr_grad.transpose()))
plt.title('Gradiometer mean power SNR')
plt.legend(cfreqs)

plt.figure()
plt.plot(tvec, amp_grad.transpose())
plt.title('Gradiometer RMS amplitude')
plt.legend(cfreqs)

sys.exit()


plt.figure()
plt.plot(tvec, 10*np.log10(snr_mag.transpose()))
plt.title('Magnetometer mean power SNR')
plt.legend(cfreqs)


plt.figure()
plt.plot(tvec, amp_mag.transpose())
plt.title('Magnetometer RMS amplitude')
plt.legend(cfreqs)



sys.exit()


    





"""
test signals


f1 = 205
f2 = 230
t = np.arange(0,2,1/sfreq)
k = 10  # slope
A1 = 3.5
A2 = 5.5
phi1 = np.pi/4
phi2 = np.pi/3
An = 0
testsig = A1 * np.sin(2*np.pi*f1*t+phi1) + A2 * np.sin(2*np.pi*f2*t+phi2) + An * np.random.randn(len(t)) + k * t + 100

# model
freqs = [f1,f2]
model = t
model = np.c_[model, np.ones(t.shape)]
for f in freqs:
    model = np.c_[model, np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)]
#testsig = testsig - testsig.mean()
amps = np.dot(np.linalg.pinv(model), testsig)
np.sqrt(amps[2]**2+amps[3]**2)


amps = amps[1:-1]
A1_est = np.sqrt(amps[0]**2+amps[1]**2)




for f in freqs:
    mod.append(np.cos(2*np.pi*f1*t))
    mod.append(np.sin(2*np.pi*f1*t))
mod.append(t)







        
            sfreq = raw.info['sfreq']

picks_meg = mne.pick_types(raw.info, meg='mag')



dt_meg = raw[picks_meg,:][0]


dt_ch = dt_meg[1,:]
f, Pxx_den = signal.welch(dt_ch, sfreq, nperseg=1024)
plt.semilogy(f, Pxx_den)



"""


