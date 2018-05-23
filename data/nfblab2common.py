import mne
import numpy as np
import pandas as pd
import pickle as p
from sklearn import preprocessing


def save_data_and_events_as_fif(df, name, fs, channels, montage_kind="standard_1020"):
    """
    save_data_and_events_as_fif(df, name, fs, channels, montage_kind="standard_1020")

    Saves raw data and events, extracted from a photoelectric sensor and block names
    as readable for Brainstorm .fif (See https://neuroimage.usc.edu/brainstorm/Introduction).

    :param df: pandas data frame returned from nfblab data by standard util
    :param name: save path or name
    :param fs: data sampling rate
    :param channels: channels to save
    :param montage_kind: mne style montage name or array
    :return: raw data (.fif); events data (.fif); event names (.txt)
    """

    # make an appropriate data name
    if name[-1] == os.sep:
        name = name[-1]

    # delete a photoelectric sensor from raw data
    if channels.count("Photo") > 0:
        channels.remove("Photo")

    # save raw data as readable for Brainstorm .fif
    montage = mne.channels.read_montage(kind=montage_kind)
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types="eeg", montage=montage)
    raw = mne.io.RawArray(df[channels].T, info)
    raw.save("{}-raw.fif".format(name), overwrite=True)

    # make events from block names
    le = preprocessing.LabelEncoder()
    le.fit(df["block_name"].unique())
    triggers = pd.Series(le.transform(df["block_name"])).diff(1)
    events = np.array(triggers.where(abs(triggers) > 0).dropna().index)
    events_ids = le.transform(df["block_name"].iloc[events])
    events_channels = np.zeros([events.shape[0]])
    block_events = np.stack([events, events_channels, events_ids], axis=1)

    event_dict = dict(zip(df["block_name"].unique(), le.transform(df["block_name"].unique())))

    # make events from a photoelectric sensor
    triggers = df['Photo'].diff(1)
    events = np.array(triggers.where(triggers > 0.00005).dropna().index)
    id = max(event_dict.values()) + 1
    event_dict.update({"Photo": id})
    events_ids = np.ones([events.shape[0]]) * id
    events_channels = np.zeros([events.shape[0]])
    photo_events = np.stack([events, events_channels, events_ids], axis=1)

    # block_events = np.array([[i, 0, df["block_name"][i]] for i in range(df["block_name"].shape[0] - 1)
    #                          if df["block_name"][i] != df["block_name"][i + 1]])

    # concatenate events of different types
    events_list = np.concatenate([photo_events, block_events])

    # save events data as readable for Brainstorm .fif
    mne.write_events("{}-eve.fif".format(name), events_list)

    # save event names
    with open("{}-names.txt".format(name), "w") as file:
        file.write(str(event_dict.keys()))
        file.write(str(event_dict.values()))
        file.close()


if __name__ == "__main__":
    import os
    from data.load_results import load_data

    load_path = "/Users/basilminkov/Neuroscience/Data/discrete_feedback"
    subject = 'df_0_05-09_12-01-47'
    data_name = "experiment_data.h5"

    df, fs, channels = load_data(os.path.join(load_path, subject, data_name))

    save_data_and_events_as_fif(df, "df0", fs, channels)
