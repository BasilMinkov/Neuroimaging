import numpy as np
import scipy
from scipy.signal import firwin, filtfilt
from data.load_results import load_data
from sklearn.decomposition import FastICA

from utilities import ProgressBar
from processing.utils import csp
from graphs.topoplots import plot_topomaps
from graphs.power import welsh_comparison
from graphs.time_series import plot_time_series
from processing.artifacts import eye_cleaner


def test_rolling_variance_maximisation(df, fs, channels, min_frequency=None, max_frequency=None,
                                       order=None, steps=None):
    """
    test_rolling_variance_maximisation(df, channels, n_channels=None, min_frequency=None, max_frequency=None,
    srate=None, order=None, steps=None)

    :param df:
    :param channels:
    :param n_channels:
    :param min_frequency:
    :param max_frequency:
    :param srate:
    :param order:
    :param steps:
    :return:
    """

    print("Test rolling variance maximisation")

    # settings
    n_channels = len(channels)
    print("n_channels: ", n_channels)
    min_frequency = min_frequency or 1
    max_frequency = max_frequency or 20
    frequency_range = max_frequency - min_frequency
    srate = fs
    order = order or 400  # filter order
    steps = steps or 300  # number of permutations

    # progressbar settings
    runs = frequency_range * (5 + steps * 3 + n_channels) + 1
    progressbar = ProgressBar(runs)

    vectors_list = np.zeros([frequency_range, n_channels, n_channels])
    values_list = np.zeros([frequency_range, n_channels])
    frequencies = np.zeros([frequency_range, 2])

    p_value_rightsided = np.zeros([frequency_range, n_channels])
    p_value_leftsided = np.zeros([frequency_range, n_channels])

    band = np.array([min_frequency, min_frequency+1])

    # df["P4"][:1000].plot()

    for frequency in np.arange(frequency_range):

        values_list_sampled = np.zeros([steps, n_channels])

        # filter dataset in particular band
        progressbar.update_progressbar("Started permutation test: {} – {} Hz".format(band[0], band[1]))
        b = firwin(order, band * 2 / srate, width=None, window='hamming', pass_zero=False)  # design filter
        a = 1
        df_filtered = df.copy()  # create filtered data frame
        progressbar.update_progressbar("{} – {} Hz; Applying filter...".format(band[0], band[1]))
        df_filtered[channels] = df_filtered[channels].apply(lambda x: filtfilt(b, a, x))  # apply filter
        progressbar.update_progressbar("{} – {} Hz; Filtered!".format(band[0], band[1]))
        df_filterd_copy = df_filtered.copy()
        df_filtered = df_filtered[channels]
        df_filtered_shape = df_filtered.shape

        # df_filtered["P4"][:1000].plot()
        sub = np.array_split(df_filtered, 30)

        for step in np.arange(steps):

            progressbar.update_progressbar("{} – {} Hz; Step: {}".format(band[0], band[1], step))

            ind = np.random.permutation(30)
            ind1 = ind[:int(ind.shape[0] / 2)]
            ind2 = ind[int(ind.shape[0] / 2):]

            real_sampled = sub[ind1[0]]
            mock_sampled = sub[ind2[0]]

            for i in ind1[1:]:
                real_sampled = np.append(real_sampled, sub[i], axis=0)

            for i in ind2[1:]:
                mock_sampled = np.append(mock_sampled, sub[i], axis=0)

            real_sampled = pd.DataFrame(real_sampled)
            mock_sampled = pd.DataFrame(mock_sampled)

            # ind = np.random.permutation(df_filtered_shape[0])
            # ind1 = ind[int(ind.shape[0]/2):]
            # ind2 = ind[:int(ind.shape[0]/2)]
            # real_sampled = df_filtered.iloc[ind1, :]  # real sampled data
            # mock_sampled = df_filtered.iloc[ind2, :]  # mock sampled data

            progressbar.update_progressbar("{} – {} Hz; Step: {}; Applying CSP...".format(band[0], band[1], step))

            values, vectors = csp(real_sampled, mock_sampled)

            del real_sampled, mock_sampled

            progressbar.update_progressbar("{} – {} Hz; Step: {}; CSP is computed.".format(band[0], band[1], step))

            # Save values for future visualisation
            values_list_sampled[step, :] = values
            # print(values)

        # print(values_list_sampled)

        real = df_filterd_copy[df_filterd_copy['block_name'] == "Open"][channels]  # real data
        mock = df_filterd_copy[df_filterd_copy['block_name'] == "Closed"][channels]  # mock data
        # real["P4"][:1000].plot()
        # mock["P4"][:1000].plot()

        values, vectors = csp(real, mock)

        vectors_list[frequency, :, :] = np.linalg.inv(vectors).transpose()
        values_list[frequency, :] = values
        frequencies[frequency, :] = band

        del values, vectors

        progressbar.update_progressbar("{} – {} Hz; Calculating statistics...".format(band[0], band[1]))
        for component in range(n_channels):

            permutated_eigs = np.zeros(steps)

            for step in range(steps):
                permutated_eigs[step] = values_list_sampled[step, component]
            eigenvalue = values_list[frequency, component]
            p_value_rightsided[frequency, component] = (sum(permutated_eigs > eigenvalue) / len(permutated_eigs))
            p_value_leftsided[frequency, component] = (sum(permutated_eigs < eigenvalue) / len(permutated_eigs))
            progressbar.update_progressbar("{} – {} Hz; Component {} is done!".format(band[0], band[1], component))

        progressbar.update_progressbar("{} – {} Hz: frequency band is done!".format(band[0], band[1]))
        band += 2

    p_value = np.concatenate([p_value_leftsided[:, 0:1], p_value_rightsided[:, -1:]], axis=1)
    p_value[p_value <= 0.05] = -1
    p_value[p_value > 0.05] = 1

    plt.show()

    progressbar.update_progressbar("Test is done!")

    return p_value, p_value_rightsided, p_value_leftsided, vectors_list, values_list, frequencies


if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    from data import parse_channels_locations_from_mat
    import numpy as np

    print("Last year algorithm")

    # Prepossessing

    considered_protocols = ['Open', 'Closed']

    df, fs, channels = load_data("/Users/basilminkov/Desktop/test/VasyaTest1_04-10_17-50-12/experiment_data.h5")

    df = pd.concat([df.loc[df['block_name'] == considered_protocols[0]],
                    df.loc[df['block_name'] == considered_protocols[1]]])

    # An estimate of the power spectral density

    # welsh_comparison(df, considered_protocols, channels, fs)

    # Dealing with channels

    colour_map = "magma"
    channels_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/chanlocs_mod.mat"
    ch3, channels_in_list, ind_in_list = parse_channels_locations_from_mat(channels_path, channels)
    ch2 = np.delete(ch3, 2, 1)

    # Delete eyes with ICA

    channels_mu = ['Fp1', 'Fp2']
    clear_eeg, n_channels, ch_names_wo_eog, mask_ch_idx = eye_cleaner(df[channels_in_list].as_matrix().T, channels_in_list, fs, channels_mu)

    # plt.subplot(211)
    # plt.plot(df[channels_in_list[0:5]][0:1000])
    # plt.ylim([-0.0002, 0.0002])

    df[channels_in_list] = pd.DataFrame(clear_eeg.T, columns=channels_in_list)

    # plt.subplot(212)
    # plt.plot(df[channels_in_list[0:5]][0:1000])
    # plt.ylim([-0.0002, 0.0002])
    # plt.show()

    # Test

    p_value, p_value_rightsided, p_value_leftsided, vectors_list, values_list, frequencies = test_rolling_variance_maximisation(df, fs, channels_in_list)

    # Save data

    names = ['p_value',
             'p_value_rightsided',
             'p_value_leftsided',
             'vectors_list',
             'values_list',
             'frequencies',
             'ch2',
             'channels_in_list']

    vars = [p_value,
            p_value_rightsided,
            p_value_leftsided,
            vectors_list,
            values_list,
            frequencies,
            ch2,
            channels_in_list]

    for i in range(len(vars)):
        np.save("/Users/basilminkov/Scripts/python3/Neuroimaging/results/eye_test/{}".format(names[i]), vars[i])