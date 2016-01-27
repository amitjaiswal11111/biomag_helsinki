# -*- coding: utf-8 -*-
"""
chpi_snr.py

Plot signal-to-noise of continuous HPI coils as a function of time.
Works by fitting a general linear model (HPI freqs, line freqs, DC, slope) to
the data, and comparing estimated HPI powers with the residual (=variance
unexplained by the model).
Window length for SNR estimates can be specified on the command line.
Longer windows will by nature include more low frequencies and thus have 
larger residual variance (lower SNR).

Tested with Python 2.7, MNE 0.11.0

@author: jussi (jnu@iki.fi)
"""


from __future__ import print_function
import matplotlib.pyplot as plt
import mne
import numpy as np
import argparse

# parameters
default_winlen = 1  # window length, seconds
default_nharm = 2  # number of line harmonics to include
# plotting parameters
legend_fontsize = 12
legend_hspace = 30  # % of horizontal space to reserve for legend

# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('fiff_file', help='Name of raw fiff file')
parser.add_argument('--winlen', type=float, default=default_winlen, help='Buffer length for SNR estimates (s)')
parser.add_argument('--nharm', type=int, default=default_nharm, choices=[0,1,2,3,4], help='Number of line frequency harmonics to include')
parser.add_argument('--stop', type=float, metavar='t', default=None, help='Process only first t seconds')
args = parser.parse_args()

# get info from fiff
raw = mne.io.Raw(args.fiff_file, allow_maxshield=True)
sfreq = raw.info['sfreq']
linefreq = raw.info['line_freq']
linefreqs = (np.arange(args.nharm+1)+1) * linefreq
buflen = int(args.winlen * sfreq)
if buflen <= 0:
    raise Exception('Window length should be >0')

cfreqs = []    
if len(raw.info['hpi_meas']) > 0 and 'coil_freq' in raw.info['hpi_meas'][0]['hpi_coils'][0]:
    for coil in raw.info['hpi_meas'][0]['hpi_coils']:
        cfreqs.append(coil['coil_freq'][0])
else:
    raise Exception('Cannot determine cHPI frequencies from raw data info')

print('\nNominal cHPI frequencies: ', cfreqs, ' Hz')    
print('Sampling frequency:', sfreq,'Hz')
print('Using line freqs:', linefreqs, ' Hz')
print('Using buffers of',buflen,'samples =',buflen/sfreq,'seconds\n')

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
model = np.c_[t, np.ones(t.shape)]  # model slope and DC
for f in list(linefreqs)+cfreqs:  # add sine and cosine term for each freq
    model = np.c_[model, np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)]
inv_model = np.linalg.pinv(model)

# loop through MEG data, fit linear model and compute SNR at each window
if args.stop:
    stop = int(args.stop * sfreq)
else:
    stop = raw.n_times
bufs = range(0, int(stop), buflen)[:-1]  # drop last buffer to avoid overrun
tvec = np.array(bufs)/sfreq
snr_avg_grad = np.zeros([len(cfreqs), len(bufs)])
snr_avg_mag = np.zeros([len(cfreqs), len(bufs)])
resid_vars = np.zeros([306, len(bufs)])
ind = 0
for buf0 in bufs:  
    megbuf = raw[pick_meg, buf0:buf0+buflen][0].transpose()
    coeffs = np.dot(inv_model, megbuf)
    coeffs_hpi = coeffs[2+2*len(linefreqs):]
    resid_vars[:,ind] = np.var(megbuf-np.dot(model,coeffs), 0)
    # get total hpi amplitudes by combining sine and cosine terms
    hpi_amps = np.sqrt(coeffs_hpi[0::2,:]**2 + coeffs_hpi[1::2,:]**2)
    # divide average HPI power by average variance
    snr_avg_grad[:,ind] = np.divide((hpi_amps**2/2)[:,grad_ind].mean(1),resid_vars[grad_ind,ind].mean())
    snr_avg_mag[:,ind] = np.divide((hpi_amps**2/2)[:,mag_ind].mean(1),resid_vars[mag_ind,ind].mean())
    ind += 1


cfreqs_legend = [str(fre)+' Hz' for fre in cfreqs]

plt.figure()
# order curve legends according to mean of data
sind = np.argsort(snr_avg_grad.mean(axis=1))[::-1]
lines1 = plt.plot(tvec, 10*np.log10(snr_avg_grad.transpose()))
plt.title('Mean cHPI power / mean residual variance (gradiometers)')
plt.legend(np.array(lines1)[sind], np.array(cfreqs_legend)[sind], prop={'size':legend_fontsize})
plt.ylabel('SNR (dB)')
plt.xlabel('Time (s)')
# create some horizontal space for legend
plt.xlim([plt.xlim()[0], plt.xlim()[1]*(1+legend_hspace/100.)])

plt.figure()
sind = np.argsort(snr_avg_mag.mean(axis=1))[::-1]
lines1 = plt.plot(tvec, 10*np.log10(snr_avg_mag.transpose()))
plt.title('Mean cHPI power / mean residual variance (magnetometers)')
plt.legend(np.array(lines1)[sind], np.array(cfreqs_legend)[sind], prop={'size':legend_fontsize})
plt.ylabel('SNR (dB)')
plt.xlabel('Time (s)')
plt.xlim([plt.xlim()[0], plt.xlim()[1]*(1+legend_hspace/100.)])

# residual (unexplained) variance as function of time
plt.figure()
plt.semilogy(tvec,resid_vars[grad_ind,:].transpose())
plt.title('Residual (unexplained) variance, gradiometers')
plt.xlabel('Time (s)')
plt.ylabel('Variance (T/m)^2')
plt.xlim([plt.xlim()[0], plt.xlim()[1]*(1+legend_hspace/100.)])


plt.show()







