import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize


def test_new(df, fs, channels, considered_protocols):

    # order = 400
    # band = np.array([2, 50])
    # b = signal.firwin(order, band * 2 / fs, width=None, window='hamming', pass_zero=False)  # design filter
    # a = 1
    # b, a = signal.butter(3, np.array([7, 14])/(fs/2))
    # df[channels] = df[channels].apply(lambda x: signal.filtfilt(b, a, x))  # apply filter

    # welsh_comparison(df, ["Open", "Closed"], channels, fs)
    # plt.plot()
    #
    # scatter_matrix(df.iloc[::1000, :])
    # plt.plot()

    # Plot spectrum for each channel

    df1 = df[df['block_name'] == considered_protocols[0]][channels].T.as_matrix()
    df2 = df[df['block_name'] == considered_protocols[1]][channels].T.as_matrix()
    # df1 = df1[:, fs*60*7+1:]
    # df2 = df2[:, fs*60*7+1:]

    # window = signal.get_window(window="hamming", Nx=fs)
    # F, Pxx_1 = signal.welch(x=df1, window=window, noverlap=0.5*fs, nfft=5*fs, fs=fs)
    # F, Pxx_2 = signal.welch(x=df2, window=window, noverlap=0.5*fs, nfft=5*fs, fs=fs)

    # n = 100
    # plt.imshow(Pxx_1[:, :n]-Pxx_2[:, :n], aspect='auto')
    # plt.yticks(np.arange(len(channels)), channels)
    # plt.xticks(np.arange(0, n, 5), map(int, F[:n:5]))
    # plt.colorbar()
    # plt.show()

    # PSD

    range_fs = np.arange(fs)
    n_windows = int(2*np.fix(df1.shape[1]/fs) - 1)
    pg1 = np.zeros([df1.shape[0], fs, n_windows])
    freq = np.fft.fftfreq(fs, 1 / fs)

    for window in range(n_windows):
        pg1[:, :, window] = np.fft.fft(df1[:, range_fs])
        range_fs += int(np.fix(fs/2))

    range_fs = np.arange(fs)
    n_windows = int(2*np.fix(df2.shape[1]/fs) - 1)
    pg2 = np.zeros([df2.shape[0], fs, n_windows])

    for window in range(n_windows):
        pg2[:, :, window] = np.fft.fft(df2[:, range_fs])
        range_fs += int(np.fix(fs/2))

    sum_n_window = pg1.shape[2] + pg2.shape[2]
    inds = np.arange(sum_n_window)

    pg = np.concatenate([pg2, pg1], axis=2)

    steps = 100
    pg1sg = np.zeros([df1.shape[0], fs, steps])
    pg2sg = np.zeros([df1.shape[0], fs, steps])
    # pgsg = np.zeros([df1.shape[0], fs, steps])

    for step in range(steps):
        ind_mdbl_p = inds  # np.random.permutation(inds)
        pg2sg[:, :, step] = np.mean(np.abs(pg[:, :, ind_mdbl_p[:int(np.fix(sum_n_window/2))]]), 2)
        pg1sg[:, :, step] = np.mean(np.abs(pg[:, :, ind_mdbl_p[int(np.fix(sum_n_window/2) + 1):]]), 2)

    pgsg = pg1sg - pg2sg

    PSD_sg_av = np.mean(pgsg, axis=2)
    PSD_sg_std = np.std(pgsg, axis=2)
    PSD_1_av = np.mean(abs(pg1), axis=2)
    PSD_2_av = np.mean(abs(pg2), axis=2)
    PSD_mdbl_av = PSD_1_av - PSD_2_av
    zscore = np.divide(PSD_mdbl_av - PSD_sg_av, PSD_sg_std)

    tmp = np.multiply(zscore, (abs(zscore) > 3))
    tmp1 = tmp
    tmp1[1, -1] = -max(abs(tmp.flatten()))
    tmp1[-1, -1] = max(abs(tmp.flatten()))

    # plt.imshow(tmp1);
    # set(h.Parent, 'Ytick', 1: 32);
    # set(h.Parent, 'YtickLabel', hdr
    # {1}.label(1: 32));
    # colormap(hsv)
    # xlabel('Frequency');
    # title('Randomization test Z-score')
    # colorbar
    # colormap(jet)
    # figure
    # imagesc(zscore(:, 1: 300))

    type = tmp1
    # n = 100
    # plt.subplot(2, 1, 2)
    plt.imshow(type[:, :31], aspect='auto', cmap="jet", vmin=-np.max(np.abs(type[:, :31])), vmax=np.max(np.abs(type[:, :31])))
    plt.yticks(np.arange(len(channels)), channels)
    # plt.xticks(np.arange(0, 250, )
    plt.ylabel("EEG Channels")
    plt.xlabel("Frequency")
    plt.title("Randomisation Test Z-Score")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    pass
