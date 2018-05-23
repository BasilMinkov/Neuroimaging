import numpy as np
import scipy
from scipy.signal import firwin, filtfilt
from data.load_results import load_data
import pandas as pd

from utilities import ProgressBar
from algorythms import csp


def test_rolling_variance_maximisation(df, trials_r, trials_m, channels, min_frequency=None, max_frequency=None,
                                       srate=None, order=None, steps=None):
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

    # settings
    n_channels = len(channels)
    print("n_channels")
    print(n_channels)
    min_frequency = min_frequency or 1
    max_frequency = max_frequency or 20
    frequency_range = max_frequency - min_frequency
    srate = srate or 500  # sampling rate
    order = order or 400  # filter order
    steps = steps or 300  # number of permutations

    sub = np.concatenate([trials_r, trials_m], axis=0)

    # progressbar settings
    runs = frequency_range * (5 + steps * 3 + n_channels) + 1
    progressbar = ProgressBar(runs)

    vectors_list = np.zeros([frequency_range, n_channels, n_channels])
    values_list = np.zeros([frequency_range, n_channels])
    frequencies = np.zeros([frequency_range, 2])

    p_value_rightsided = np.zeros([frequency_range, n_channels])
    p_value_leftsided = np.zeros([frequency_range, n_channels])

    band = np.array([min_frequency, min_frequency + 1])

    for frequency in np.arange(frequency_range):

        values_list_sampled = np.zeros([steps, n_channels])

        # filter dataset in particular band
        progressbar.update_progressbar("Started permutation test: {} – {} Hz".format(band[0], band[1]))
        b = firwin(order, band * 2 / srate, width=None, window='hamming', pass_zero=False)  # design filter
        a = 1
        df_filtered = df.copy()  # create filtered data frame
        progressbar.update_progressbar("{} – {} Hz; Applying filter...".format(band[0], band[1]))
        df_filtered = df_filtered[channels].apply(lambda x: filtfilt(b, a, x))  # apply filter
        progressbar.update_progressbar("{} – {} Hz; Filtered!".format(band[0], band[1]))

        for step in np.arange(steps):

            progressbar.update_progressbar("{} – {} Hz; Step: {}".format(band[0], band[1], step))

            ind = np.random.permutation(len(sub))
            ind1 = ind[:int(ind.shape[0] / 2)]
            ind2 = ind[int(ind.shape[0] / 2):]

            real_sampled = np.concatenate(sub[ind1], axis=0)
            mock_sampled = np.concatenate(sub[ind2], axis=0)

            real_sampled = df_filtered.iloc[real_sampled, :]
            mock_sampled = df_filtered.iloc[mock_sampled, :]

            progressbar.update_progressbar("{} – {} Hz; Step: {}; Applying CSP...".format(band[0], band[1], step))

            values, vectors = csp(real_sampled, mock_sampled)

            del real_sampled, mock_sampled

            progressbar.update_progressbar("{} – {} Hz; Step: {}; CSP is computed.".format(band[0], band[1], step))

            # Save values for future visualisation
            values_list_sampled[step, :] = values

        real = np.concatenate(trials_r, axis=0)
        mock = np.concatenate(trials_m, axis=0)

        real = df_filtered.iloc[real, :]
        mock = df_filtered.iloc[mock, :]

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

    progressbar.update_progressbar("Test is done!")

    return p_value, p_value_rightsided, p_value_leftsided, vectors_list, values_list, frequencies


if __name__ == "__main__":

    from data import parse_channels_locations_from_mat, load_signals_data, load_data
    import numpy as np

    # Load experimental data

    data_path = "/Users/basilminkov/Neuroscience/Data/Test/20.02.17/Alpha1_02-20_17-52-50/experiment_data.h5"
    df, fs, p_names, channels = load_data(data_path)

    # Load signal data

    df_full = load_signals_data(
        "/Users/basilminkov/Neuroscience/Data/Test/20.02.17/Alpha1_02-20_17-52-50/experiment_data.h5")

    # Prepare epochs

    x_r = df_full.loc[df_full['block_name'] == 'Real', 'P4'].as_matrix()
    y_r = df_full.loc[df_full['block_name'] == 'Real', 'Composite'].as_matrix()
    df_r = df.loc[df['block_name'] == 'Real'][channels]

    x_m = df_full.loc[df_full['block_name'] == 'Mock', 'P4'].as_matrix()
    y_m = df_full.loc[df_full['block_name'] == 'Mock', 'Composite'].as_matrix()
    df_m = df.loc[df['block_name'] == 'Mock'][channels]

    fs = 500
    events = np.where(np.diff((x_r > y_r) * 1) > 0)[0]
    trials_r = []
    c = 0
    for ev in events:
        if ev - c > 3 * fs:
            trials_r.append(np.arange(ev, ev + 1000))
            c = ev

    events = np.where(np.diff((x_m > y_m) * 1) > 0)[0]
    trials_m = []
    c = 0
    for ev in events:
        if ev - c > 3 * fs:
            trials_m.append(np.arange(ev, ev + 1000))
            c = ev

    trials_r = np.array(trials_r)
    trials_m = np.array(trials_m)

    # Load Channels Positions

    channels_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/chanlocs_mod.mat"
    ch3, channels_in_list, ind_in_list = parse_channels_locations_from_mat(channels_path, df.columns.values)
    ch2 = np.delete(ch3, 2, 1)

    # Run test

    p_value, p_value_rightsided, p_value_leftsided, vectors_list, values_list, frequencies = test_rolling_variance_maximisation(df, trials_r, trials_m, channels_in_list)

    # Save data

    names = ['p_value', 'p_value_rightsided', 'p_value_leftsided', 'vectors_list', 'values_list', 'frequencies']
    vars = [p_value, p_value_rightsided, p_value_leftsided, vectors_list, values_list, frequencies]

    for i in range(len(vars)):
        np.save("/Users/basilminkov/Scripts/python3/Neuroimaging/results/aa/{}".format(names[i]), vars[i])
