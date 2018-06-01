import sys
import os
import h5py
import mne
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn import preprocessing
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize


def _get_channels_and_fs(xml_str_or_file):
    root = ET.fromstring(xml_str_or_file)
    channels = [k.find('label').text for k in root.find('desc').find('channels').findall('channel')]
    fs = int(root.find('nominal_srate').text)
    return channels, fs


def _get_signals_list(xml_str):
    root = ET.fromstring(xml_str)
    derived = [s.find('sSignalName').text for s in root.find('vSignals').findall('DerivedSignal')]
    composite = [s.find('sSignalName').text for s in root.find('vSignals').findall('CompositeSignal')]
    return derived + composite


def _get_info(f):
    channels, fs = _get_channels_and_fs(f['stream_info.xml'][0])
    signals = _get_signals_list(f['settings.xml'][0])
    n_protocols = len([k for k in f.keys() if ('protocol' in k and k != 'protocol0')])
    block_names = [f['protocol{}'.format(j+1)].attrs['name'] for j in range(n_protocols)]
    return fs, channels, block_names, signals


def load_data(file_path):
    with h5py.File(file_path) as f:
        # load meta info
        fs, channels, p_names, signals = _get_info(f)

        # load raw data
        data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]
        df = pd.DataFrame(np.concatenate(data), columns=channels)

        # load signals data
        signals_data = [f['protocol{}/signals_data'.format(k + 1)][:] for k in range(len(p_names))]
        df_signals = pd.DataFrame(np.concatenate(signals_data), columns=['signal_'+s for s in signals])
        df = pd.concat([df, df_signals], axis=1)

        # load timestamps
        timestamp_data = [f['protocol{}/timestamp_data'.format(k + 1)][:] for k in range(len(p_names))]
        df['timestamps'] = np.concatenate(timestamp_data)

        # events data
        events_data = [f['protocol{}/mark_data'.format(k + 1)][:] for k in range(len(p_names))]
        df['events'] = np.concatenate(events_data)

        # set block names and numbers
        df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
        df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])
    return df, fs, channels


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

    raw.interpolate_bads()

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
        file.write("Event, Number\n")
        for name in event_dict:
            file.write("{}, {}\n".format(name, event_dict[name]))
        file.close()


class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(320, 140))
        self.setWindowTitle("NFBLab2Fif Converter")

        self.nameLabel = QLabel(self)
        self.nameLabel.setText('Path:')
        self.line = QLineEdit(self)

        self.line.move(80, 20)
        self.line.resize(200, 32)
        self.nameLabel.move(20, 20)

        pybutton = QPushButton('Process', self)
        pybutton.clicked.connect(self.click_method)
        pybutton.resize(200, 32)
        pybutton.move(80, 60)

    def click_method(self):
        try:
            data_path = self.line.text()
            data_list = data_path.split(os.sep)
            save_name = os.path.join(data_path.replace("experiment_data.h5", ""), data_list[-2])
            del data_list
            df, fs, channels = load_data(data_path)
            save_data_and_events_as_fif(df, save_name, fs, channels)
        except KeyError:
            print("Key Error: Wrong path name!")
        except OSError:
            print("OS Error: Wrong path name!")


if __name__ == "__main__":

    # With interface
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

    # Without interface
    # data_path = ""
    # data_list = data_path.split(os.sep)
    # save_name = os.path.join(data_path.replace("experiment_data.h5", ""), data_list[-2])
    # del data_list
    # df, fs, channels = load_data(data_path)
    # save_data_and_events_as_fif(df, "df0", fs, channels)
