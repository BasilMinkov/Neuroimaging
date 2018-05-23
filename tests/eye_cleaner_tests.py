from sklearn.decomposition import FastICA
from graphs.time_series import plot_time_series
from graphs.topoplots import plot_topomaps
from processing.artifacts import eye_cleaner


def before_and_aclear_eyes(df, channels_in_list):

    # Plot ICA

    df = df[channels_in_list]

    ica = FastICA(n_components=len(channels_in_list))
    S_ = ica.fit_transform(df)  # Get the estimated sources
    A_ = ica.mixing_  # Get estimated mixing matrix
    C_ = ica.components_

    plot_topomaps(A_, channels_in_list, ch2)
    plot_time_series(S_, np.arange(0, 2000), channels_in_list)

    # Delete eyes with ICA

    channels_mu = ['Fp1', 'Fp2']
    clear_eeg, n_channels, ch_names_wo_eog, mask_ch_idx = eye_cleaner(df[channels_in_list].as_matrix().T, channels_in_list, fs, channels_mu)

    # Test cleaning

    df = pd.DataFrame(clear_eeg.T, columns=channels_in_list)

    ica = FastICA(n_components=len(channels_in_list))
    S_ = ica.fit_transform(df)  # Get the estimated sources
    A_ = ica.mixing_  # Get estimated mixing matrix
    C_ = ica.components_

    plot_topomaps(A_, channels_in_list, ch2)
    plot_time_series(S_, np.arange(0, 2000), channels_in_list)