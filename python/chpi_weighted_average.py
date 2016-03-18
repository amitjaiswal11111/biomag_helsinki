# -*- coding: utf-8 -*-
"""
Weighted averaging of epochs according to continuous HPI (cHPI) signal-to-noise.


@author: jussi
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
parser.add_argument('fiff_file', help='Name of raw fiff file. Epochs for averaging will be taken from this file.')
parser.add_argument('event', help='Event code or category.')
parser.add_argument('--chpi_file', type=float, default=None, help='File to compute cHPI SNR from. It must have the same number of epochs as the file to average.')
parser.add_argument('--nharm', type=int, default=default_nharm, choices=[0,1,2,3,4], help='Number of line frequency harmonics to include')
parser.add_argument('--stop', type=float, metavar='t', default=None, help='Process only first t seconds')
args = parser.parse_args()
