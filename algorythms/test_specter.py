import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
import mne

from pandas.plotting import scatter_matrix

from data import parse_channels_locations_from_mat
from data.load_results import load_data
from processing.artifacts import ica_rejection
from graphs.power import welsh_comparison
from algorythms.test_new import test_new
from data import montage

sub = [
    'df_0_05-09_12-01-47',
    'df_1_05-09_15-44-03',
    'df_2_05-09_19-04-31',
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

load_path = "/Users/basilminkov/Neuroscience/Data/discrete_feedback/{}/experiment_data.h5"
channels_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/chanlocs_mod.mat"

considered_protocols = ['Real', 'Mock']

real_list = []
mock_list = []

for i in range(1):

    df, fs, channels = load_data(load_path.format(sub[i]))
    try:
        channels.remove("T7")
    except ValueError:
        pass
    ch3, channels_in_list, ind_in_list = parse_channels_locations_from_mat(channels_path, channels)
    ch2 = np.delete(ch3, 2, 1)

    df1 = df[df['block_name'] == considered_protocols[0]][channels_in_list].T.as_matrix()
    df2 = df[df['block_name'] == considered_protocols[1]][channels_in_list].T.as_matrix()
    #
    window = signal.get_window(window="hamming", Nx=fs)
    F, Pxx_1 = signal.welch(x=df1, window=window, noverlap=0.5*fs, nfft=5*fs, fs=fs)
    # F, Pxx_2 = signal.welch(x=df2, window=window, noverlap=0.5*fs, nfft=5*fs, fs=fs)
    # F, Pxx_1 = signal.welch(x=df1, fs=fs)
    # # F, Pxx_2 = signal.welch(x=df2, fs=fs)
    # real_list.append(Pxx_1)
    # mock_list.append(Pxx_2)
    #
    print(len(real_list))
    # del df, df1, df2, Pxx_1, Pxx_2

# real_mx = np.stack(real_list, axis=2)
# mock_mx = np.stack(mock_list, axis=2)
#
# ttest_mx = np.zeros([real_mx.shape[0], real_mx.shape[1]])
# mannwhitneyu_mx = np.zeros([real_mx.shape[0], real_mx.shape[1]])
# kruskal_mx = np.zeros([real_mx.shape[0], real_mx.shape[1]])

# print(real_mx.shape, mock_mx.shape)

# for i in range(real_mx.shape[0]):
#     for j in range(real_mx.shape[1]):
#         s, ttest_mx[i, j] = stats.ttest_ind(real_mx[i, j, :], mock_mx[i, j, :])
#         s, mannwhitneyu_mx[i, j] = stats.mannwhitneyu(real_mx[i, j, :], mock_mx[i, j, :])
#         s, kruskal_mx[i, j] = stats.kruskal(real_mx[i, j, :], mock_mx[i, j, :])
#
# np.save("ttest_mx5.npy", ttest_mx)
# np.save("mannwhitneyu_mx5.npy", mannwhitneyu_mx)
# np.save("kruskal_mx5.npy", kruskal_mx)

# print(len(F))

ttest_mx = np.load("ttest_mx5.npy")
mannwhitneyu_mx = np.load("mannwhitneyu_mx5.npy")
kruskal_mx = np.load("ttest_mx5.npy")

# n = 15
#
# plt.subplot(1, 3, 1)
# plt.imshow(ttest_mx[:, :n], aspect='auto', cmap="RdBu_r", vmax=1, vmin=0)
# plt.yticks(np.arange(len(channels_in_list)), channels_in_list)
# plt.xticks(np.arange(0, n, 5), map(int, F[:n:5]))
# plt.xlim([0, 100])
# plt.ylabel("EEG Channels")
# plt.xlabel("Frequency")
# plt.title("Two Sided T-Test")
#
# plt.subplot(1, 3, 2)
# plt.imshow(mannwhitneyu_mx[:, :n], aspect='auto', cmap="RdBu_r", vmax=1, vmin=0)
# plt.yticks(np.arange(len(channels_in_list)), channels_in_list)
# plt.xticks(np.arange(0, n, 5), map(int, F[:n:5]))
# plt.xlim([0, 100])
# plt.ylabel("EEG Channels")
# plt.xlabel("Frequency")
# plt.title("Mann–Whitney U test")
#
# plt.subplot(1, 3, 3)
# plt.imshow(kruskal_mx[:, :n], aspect='auto', cmap="RdBu_r", vmax=1, vmin=0)
# plt.yticks(np.arange(len(channels_in_list)), channels_in_list)
# plt.xticks(np.arange(0, n, 5), map(int, F[:n:5]))
# plt.xlim([0, 100])
# plt.ylabel("EEG Channels")
# plt.xlabel("Frequency")
# plt.title("Kruskal–Wallis one-way analysis")
# plt.colorbar()
# plt.show()

n = 100

plt.subplot(1, 3, 1)
plt.imshow(ttest_mx, aspect='auto', cmap="RdBu_r", vmax=1, vmin=0)
plt.yticks(np.arange(len(channels_in_list)), channels_in_list)
plt.xticks(np.arange(0, n, 5), map(int, F[:n:5]))
plt.xlim([0, 100])
plt.ylabel("EEG Channels")
plt.xlabel("Frequency")
plt.title("Two Sided T-Test")

plt.subplot(1, 3, 2)
plt.imshow(mannwhitneyu_mx, aspect='auto', cmap="RdBu_r", vmax=1, vmin=0)
plt.yticks(np.arange(len(channels_in_list)), channels_in_list)
plt.xticks(np.arange(0, n, 5), map(int, F[:n:5]))
plt.xlim([0, 100])
plt.ylabel("EEG Channels")
plt.xlabel("Frequency")
plt.title("Mann–Whitney U test")

plt.subplot(1, 3, 3)
plt.imshow(kruskal_mx, aspect='auto', cmap="RdBu_r", vmax=1, vmin=0)
plt.yticks(np.arange(len(channels_in_list)), channels_in_list)
plt.xticks(np.arange(0, n, 5), map(int, F[:n:5]))
plt.xlim([0, 100])
plt.ylabel("EEG Channels")
plt.xlabel("Frequency")
plt.title("Kruskal–Wallis one-way analysis")
plt.colorbar()
plt.show()

colour_map = "RdBu_r"
m = montage.Montage(channels_in_list)

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 6))
counter = 0
for i in range(3):
    for j in range(3):
        ax1, _ = mne.viz.plot_topomap(ttest_mx[:, n-counter], m.get_pos(),
                             names=channels_in_list,
                             show_names=True,
                             axes=ax[i][j], cmap=colour_map,
                             show=False, contours=0, vmin=0, vmax=1)
        ax[i][j].set_title(str(F[n-counter])+' Hz')
        counter += 1
# divider = make_axes_locatable(ax[i][j])
plt.colorbar(ax1)
plt.show()
