import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
import mne
import h5py
from sklearn.preprocessing import normalize

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

load_path = "/Users/basilminkov/Neuroscience/Data/discrete_feedback/{}/experiment_data.h5"
channels_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/chanlocs_mod.mat"

considered_protocols = ['Real', 'Mock']

real_list = []
mock_list = []

# for i in range(1):
for i in range(1):

    df, fs, channels = load_data(load_path.format(sub[i]))

    with h5py.File(load_path.format(sub[i])) as f:
        rej = f['protocol3/signals_stats/P4/rejections/rejection1'][:]
    df[channels] = np.dot(df[channels], rej)

    try:
        channels.remove("T7")
    except ValueError:
        pass
    ch3, channels_in_list, ind_in_list = parse_channels_locations_from_mat(channels_path, channels)
    ch2 = np.delete(ch3, 2, 1)

    df1 = df[df['block_name'] == considered_protocols[0]][channels_in_list].T.as_matrix()
    # df2 = df[df['block_name'] == considered_protocols[1]][channels_in_list].T.as_matrix()

    # window = signal.get_window(window="hamming", Nx=fs)
    # F, Pxx_1 = signal.welch(x=df1, noverlap=0.5*fs, nfft=5*fs, fs=fs, nperseg=5*fs)
    # F, Pxx_2 = signal.welch(x=df2, noverlap=0.5*fs, nfft=5*fs, fs=fs, nperseg=5*fs)
    F, Pxx_1 = signal.welch(x=df1, nfft=fs, fs=fs)
    # F, Pxx_2 = signal.welch(x=df2, nfft=fs, fs=fs)
    # real_list.append(Pxx_1)
    # mock_list.append(Pxx_2)
    #
    # plt.plot(F, Pxx_1.T-Pxx_2.T)
    # plt.show()
    #
    print(len(real_list))
    # del df, df1, df2, Pxx_1, Pxx_2

m = montage.Montage(channels_in_list)

# real_mx = np.stack(real_list, axis=2)
# mock_mx = np.stack(mock_list, axis=2)

# np.save("real_mx.npy", real_mx)
# np.save("mock_mx.npy", mock_mx)

real_mx = np.load("real_mx_eye.npy")*10e11
mock_mx = np.load("mock_mx_eye.npy")*10e11

alpha_mask = (F >= 7) & (F <= 14)
beta_mask = (F >= 11) & (F <= 30)
theta_mask = (F >= 3) & (F <= 7)
delta_mask = (F >= 0) & (F <= 4)

alpha_real = np.mean(real_mx[:, alpha_mask, :], 1)
beta_real = np.mean(real_mx[:, beta_mask, :], 1)
theta_real = np.mean(real_mx[:, theta_mask, :], 1)
delta_real = np.mean(real_mx[:, delta_mask, :], 1)
real = np.stack([alpha_real, beta_real, theta_real, delta_real], 2)

alpha_mock = np.mean(mock_mx[:, alpha_mask, :], 1)
beta_mock = np.mean(mock_mx[:, beta_mask, :], 1)
theta_mock = np.mean(mock_mx[:, theta_mask, :], 1)
delta_mock = np.mean(mock_mx[:, delta_mask, :], 1)
mock = np.stack([alpha_mock, beta_mock, theta_mock, delta_mock], 2)

ttest_mx = np.zeros([real_mx.shape[0], 4])
mannwhitneyu_mx = np.zeros([real_mx.shape[0], 4])
kruskal_mx = np.zeros([real_mx.shape[0], 4])

for i in range(real_mx.shape[0]):  # electrode
    for j in range(4):  # waves
        s, ttest_mx[i, j] = stats.ttest_ind(real[i, :, j], mock[i, :, j])
        s, mannwhitneyu_mx[i, j] = stats.mannwhitneyu(real[i, :, j], mock[i, :, j])
        s, kruskal_mx[i, j] = stats.kruskal(real[i, :, j], mock[i, :, j])

ds = np.zeros([real_mx.shape[0], 4])
for i in range(4):
    ds[:, i] = (np.mean(real[:, :, i], 1) - np.mean(mock[:, :, i], 1))/np.mean(mock[:, :, i], 1)

dsg = np.zeros([real_mx.shape[0], len(sub), 4])
for i in range(4):
    dsg[:, :, i] = (real[:, :, i] - (mock[:, :, i]))/mock[:, :, i]

names = ['Alpha', 'Beta', 'Theta', 'Delta']

# plt.subplot(1, 3, 1)
# plt.imshow(ttest_mx, aspect='auto', cmap="RdBu_r", vmax=1, vmin=0)
# plt.yticks(np.arange(len(channels_in_list)), channels_in_list)
# plt.xticks(np.arange(4), names)
# plt.ylabel("EEG Channels")
# plt.xlabel("Frequency")
# plt.title("Two Sided T-Test")
# plt.subplot(1, 3, 2)
# plt.imshow(mannwhitneyu_mx, aspect='auto', cmap="RdBu_r", vmax=1, vmin=0)
# plt.yticks(np.arange(len(channels_in_list)), channels_in_list)
# plt.xticks(np.arange(4), names)
# plt.ylabel("EEG Channels")
# plt.xlabel("Frequency")
# plt.title("Mann–Whitney U test")
# plt.subplot(1, 3, 3)
# plt.imshow(kruskal_mx, aspect='auto', cmap="RdBu_r", vmax=1, vmin=0)
# plt.yticks(np.arange(len(channels_in_list)), channels_in_list)
# plt.xticks(np.arange(4), names)
# plt.ylabel("EEG Channels")
# plt.xlabel("Frequency")
# plt.title("Kruskal–Wallis one-way analysis")
# plt.colorbar()
# plt.show()

colour_map = "RdBu_r"
names = ['18.4 - 20.0 Hz', 'Beta', 'Theta', 'Delta']

for wave in range(len(names)):
    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(15, 6))
    counter = 0
    for i in range(6):
        for j in range(2):
            ax1, _ = mne.viz.plot_topomap(dsg[:, counter, wave],
                                          m.get_pos(),
                                          names=channels_in_list,
                                          show_names=True,
                                          axes=ax[i][j],
                                          cmap=colour_map,
                                          show=False,
                                          contours=0,
                                          vmin=5,
                                          vmax=-5
                                          )
            ax[i][j].set_title('Sub {}'.format(counter))
            counter += 1
    plt.colorbar(ax1)
    plt.show()

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 6))
counter = 0

for i in range(4):
        ax1, _ = mne.viz.plot_topomap(ds[:, i],
                                      m.get_pos(),
                                      names=channels_in_list,
                                      show_names=True,
                                      axes=ax[i],
                                      cmap=colour_map,
                                      show=False,
                                      contours=0,
                                      vmax=1)
        ax[i].set_title(names[i]+' desynchronization coefficient')
plt.colorbar(ax1)
plt.show()

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 6))
counter = 0

for i in range(4):
        ax1, _ = mne.viz.plot_topomap(kruskal_mx[:, i],
                                      m.get_pos(),
                                      names=channels_in_list,
                                      show_names=True,
                                      axes=ax[i],
                                      cmap=colour_map,
                                      show=False,
                                      contours=0,
                                      vmin=0,
                                      vmax=1)
        ax[i].set_title(names[i]+' Kruskal–Wallis p-value')
plt.colorbar(ax1)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=True)
axes.boxplot(np.stack([alpha_real.ravel(), alpha_mock.ravel()], axis=1), labels=considered_protocols)
axes.set_title('Cross-electrode Cross-subject Power Box Plot')
plt.show()