import numpy as np
import mne
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import os


def ica_rejection(eeg_data, channels, fs, channels_mu):

    montage = mne.channels.read_montage(kind="standard_1020")

    print("\n\tCreate mne raw array\n")
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types="eeg", montage=montage)
    eeg_instance = mne.io.RawArray(eeg_data, info)

    print("\n\tPlot some channels\n")
    print("Plotting...")
    eeg_instance.plot()
    plt.show()

    print("\n\tProcessing ICA\n")
    ica = mne.preprocessing.ICA(method='extended-infomax', max_iter=300, verbose='INFO')
    ica.fit(eeg_instance, verbose='INFO')
    sources = ica.get_sources(eeg_instance)
    S = sources.get_data()
    X = eeg_instance.get_data()
    print()

    print("\tFind eye artifacts components\n")
    eog_channels = [eeg_instance.ch_names.index(channels_mu[0]), eeg_instance.ch_names.index(channels_mu[1])]
    mu_coefs = np.zeros((S.shape[0]))
    for eog in eog_channels:
        for i in range(S.shape[0]):
            mu_coefs[i] += mutual_information(X[eog, :].ravel(), S[i, :].ravel(), int(S.shape[1] / fs))
    del X, S
    upper_bound = outliers_iqr(mu_coefs)
    mu_coefs_sorted_idx = np.flip(np.argsort(mu_coefs), axis=0)
    # fig = plt.figure()
    # plt.plot(np.full(mu_coefs.size, upper_bound))
    # plt.plot(mu_coefs[mu_coefs_sorted_idx])
    # plt.show()
    # plt.close(fig)
    ex_comps = mu_coefs_sorted_idx[mu_coefs[mu_coefs_sorted_idx] > upper_bound].ravel().tolist()
    inc_comps = [i for i in np.arange(len(sources.ch_names)) if not i in ex_comps]
    # print('[Debug] Bad comps {0}, good comps {1} '.format(ex_comps, inc_comps))

    print("Components to exclude: ", ex_comps)
    print("Components to include: ", inc_comps)

    subject_save_path = os.path.join(save_path, subject)

    if not os.path.exists(subject_save_path):
        os.makedirs(subject_save_path)

    # Plot topoplots of components
    ica.plot_components(show=False)

    # Plot sourses
    ica.plot_sources(eeg_instance, show=False)
    plt.show()

    print("\n\tEnter components to reject.")
    while True:
        try:
            c = input().split()
            if c[0] == "end":
                print("Rejecting bad components...")
                break
            elif c[0] == "add":
                inc_comps.remove(int(c[1]))
                ex_comps.append(int(c[1]))
            elif c[0] == "drop":
                inc_comps.append(int(c[1]))
                ex_comps.remove(int(c[1]))
            else:
                print("Wrong command")
            print("Components to exclude: ", ex_comps)
            print("Components to include: ", inc_comps)
        except ValueError:
            print(c[1], "not in list")

    plt.figure(1)

    # Reject band components
    clear_eeg = eeg_instance.copy()
    clear_eeg = ica.apply(clear_eeg, inc_comps, ex_comps)

    # Potentially drop some more channels
    # clear_eeg = clear_eeg.drop_channels(channels_mu)
    ch_names_wo_eog = clear_eeg.ch_names
    clear_eeg = clear_eeg.drop_channels(eeg_instance.info['bads'])
    n_channels = len(ch_names_wo_eog)
    # print('[Debug] Channels: ', ch_names_wo_eog, n_channels)
    mask_ch_idx = [ch_names_wo_eog.index(ch) for ch in ch_names_wo_eog
                   if not ch in (eeg_instance.info['bads'])]

    print("\n\tPlot clean channels\n")
    print("Plotting...")
    eeg_instance.plot()
    clear_eeg.plot()
    plt.show()

    return clear_eeg.get_data(), n_channels, ch_names_wo_eog, mask_ch_idx


def outliers_iqr(signal):
    quartile_1, quartile_3 = np.percentile(signal, [25, 75])
    iqr = quartile_3 - quartile_1
    upper_bound = quartile_3 + (iqr * 1.5)
    return upper_bound


def mutual_information(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


if __name__ == "__main__":

    import pandas as pd

    from data import parse_channels_locations_from_mat
    from data.load_results import load_data

    # Clear data

    # Variables

    data_name = "experiment_data.h5"
    clear_data_name = "clear_eeg"
    load_path = "/Users/basilminkov/Neuroscience/Data/discrete_feedback"
    save_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/results/discrete_feedback"
    colour_map = "magma"
    channels_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/chanlocs_mod.mat"
    fs = 500

    subjects = ['VasyaTest1_04-10_17-50-12']
    # subjects.remove('.DS_Store')
    # subjects.remove('df_11_05-13_17-08-06')
    # subjects.remove('df_9_05-12_15-33-15')

    for subject in subjects:

        # Load data

        considered_protocols = ['Open', 'Closed']

        df, fs, channels = load_data(os.path.join(load_path, subject, data_name))
        df = pd.concat([df.loc[df['block_name'] == considered_protocols[0]],
                    df.loc[df['block_name'] == considered_protocols[1]]])

        # Dealing with channels

        ch3, channels_in_list, ind_in_list = parse_channels_locations_from_mat(channels_path, channels)
        ch2 = np.delete(ch3, 2, 1)

        # Delete artifacts with ICA

        channels_mu = ['Fp1', 'Fp2']
        clear_eeg, n_channels, ch_names_wo_eog, mask_ch_idx = ica_rejection(df[channels_in_list].as_matrix().T,
                                                                        channels_in_list,
                                                                        fs,
                                                                        channels_mu)

        df[channels_in_list] = pd.DataFrame(clear_eeg.T, columns=channels_in_list)

        # Save data and channels info

        names = ['ch2', 'channels_in_list']
        vars = [ch2, channels_in_list]

        subject_save_path = os.path.join(save_path, subject)

        if not os.path.exists(subject_save_path):
            os.makedirs(subject_save_path)

        for i in range(2):
            np.save(os.path.join(subject_save_path, names[i]), vars[i])
            df.to_csv(os.path.join(subject_save_path, clear_data_name))
