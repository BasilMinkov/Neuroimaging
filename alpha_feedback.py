import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import firwin, filtfilt

from pandas.plotting import scatter_matrix

from data import parse_channels_locations_from_mat
from data.load_results import load_data
from processing.artifacts import ica_rejection
from graphs.power import welsh_comparison
from algorythms.test_new import test_new

# Variables

sub = [
    'df_0_05-09_12-01-47',
    'df_1_05-09_15-44-03',
    'df_2_05-09_19-04-31',
    'df_3_05-10_11-09-55',
    'df_4_05-10_16-12-03',
    'df_5_05-10_18-43-12',
    'df_6_05-11_15-57-07',
    'df_7_05-11_18-44-53',
    'df_8_05-11_21-12-48',
    'df_9_05-12_15-33-15',
    'df_10_05-12_19-08-24',
    'df_11_05-13_17-08-06',
    'df_12_05-15_12-43-13',
       ]

load_path = "/Users/basilminkov/Neuroscience/Data/discrete_feedback/{}/experiment_data.h5".format(sub[0])
# load_path = "/Users/basilminkov/Neuroscience/Data/alpha-delay-subj-14_05-16_17-17-22/experiment_data.h5"
# load_path = "/Users/basilminkov/Desktop/test/VasyaTest1_04-10_17-50-12/experiment_data.h5"
save_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/results/eye_test/{}"
colour_map = "magma"
channels_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/chanlocs_mod.mat"
fs = 500

# Load data

# clear_data = pd.read_csv("/Users/basilminkov/Scripts/python3/Neuroimaging/results/discrete_feedback/df_1_05-09_15-44-03/clear_eeg")
# clear_data = pd.read_csv("/Users/basilminkov/Scripts/python3/Neuroimaging/results/eye_test/clear_eeg")

considered_protocols = ['Real', 'Mock']
# considered_protocols = ['FB0', 'FBMock']

df, fs, channels = load_data(load_path)
# df = pd.concat([df.loc[df['block_name'] == considered_protocols[0]],
#                 df.loc[df['block_name'] == considered_protocols[1]]])

# Dealing with channels

# a = df.loc[df['block_name'] == 'Real']
# b = df.loc[df['block_name'] == 'Mock']

ch3, channels_in_list, ind_in_list = parse_channels_locations_from_mat(channels_path, channels)
ch2 = np.delete(ch3, 2, 1)
#
# sp = abs(np.fft.fft(clear_data['Fp1']))
# freq = np.fft.fftfreq(clear_data['Fp1'].shape[0], 1/500)
# plt.plot(freq, sp.real)
# plt.show()

# df['Fp1'].plot()
# clear_data['Fp1'].plot()
# plt.plot(df['Fp1']-clear_data['Fp1'])
# plt.show()

# df[df["block_name"] == "Rest"] = 100
# fig, ax = plt.subplots()
# f, t, Sxx = signal.spectrogram(df['Fc2'], fs, nperseg=500, noverlap=250)
# a = ax.pcolormesh(t, f, np.log10(Sxx**0.5), cmap="nipy_spectral", vmin=-6.6, vmax=-5)
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.81, 0.05, .15])
# cb = fig.colorbar(a, cax=cbar_ax)
# ax.set_ylabel('Frequency [Hz]')
# ax.set_xlabel('Time [sec]')
# ax.set_ylim([0, 30])
# plt.show()

# Run new test

# plt.show()

# test_new(df, fs, channels_in_list, considered_protocols)

# considered_protocols = ['Real', 'M']
#

test_new(df, fs, channels_in_list, considered_protocols)

# srate = fs
# order = 400  # filter order
# b = firwin(order, np.array([7, 14]) * 2 / srate, width=None, window='hamming', pass_zero=False)  # design filter
# a = 1
# df[channels] = df[channels].apply(lambda x: filtfilt(b, a, x))
#
# print(df['P4'][df['block_name'] == 'FB0'].std())
#  print(df['P4'][df['block_name'] == 'FBMock'].std())