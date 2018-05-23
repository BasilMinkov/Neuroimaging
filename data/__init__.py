import numpy as np
import pandas as pd
import h5py
import xml.etree.ElementTree as ET
from scipy.io import loadmat

ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
            'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2',
            'Fpz', 'Af7', 'Af3', 'Af4', 'Af8', 'F5', 'F1', 'F2', 'F6', 'Ft7', 'Fc3', 'Fcz', 'Fc4', 'Ft8', 'C5', 'C1',
            'C2', 'C6', 'Tp7', 'Cp3', 'Cpz', 'Cp4', 'Tp8', 'P5', 'P1', 'P2', 'P6', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8',
            'Aff1h', 'Aff2h', 'F9', 'F10', 'Ffc5h', 'Ffc1h', 'Ffc2h', 'Ffc6h', 'Ftt7h', 'Fcc3h', 'Fcc4h', 'Ftt8h',
            'Ccp5h', 'Ccp1h', 'Ccp2h', 'Ccp6h', 'Tpp7h', 'Cpp3h', 'Cpp4h', 'Tpp8h', 'P9', 'P10', 'Ppo9h', 'Ppo1h',
            'Ppo2h', 'Ppo10h', 'Po9', 'Po10', 'I1', 'Oi1h', 'Oi2h', 'I2', 'Afp1', 'Afp2', 'Aff5h', 'Aff6h', 'Fft9h',
            'Fft7h', 'Ffc3h', 'Ffc4h', 'Fft8h', 'Fft10h', 'Ftt9h', 'Fcc5h', 'Fcc1h', 'Fcc2h', 'Fcc6h', 'Ftt10h',
            'Ttp7h', 'Ccp3h', 'Ccp4h', 'Ttp8h', 'Tpp9h', 'Cpp5h', 'Cpp1h', 'Cpp2h', 'Cpp6h', 'Tpp10h', 'Ppo5h', 'Ppo6h',
            'Poo9h', 'Poo1', 'Poo2', 'Poo10h', 'Aux 1.1', 'Aux 1.2', 'Aux 2.1', 'Aux 2.2', 'Aux 3.1', 'Aux 3.2',
            'Aux 4.1', 'Aux 4.2']

ch_names32 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
              'C4', 'T8', 'Tp9',
              'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']


def get_lsl_info_from_xml(xml_str_or_file):
    try:
        root = ET.fromstring(xml_str_or_file)
    except FileNotFoundError:
        root = ET.fromstring(xml_str_or_file)
    info = {}
    channels = [k.find('label').text for k in root.find('desc').find('channels').findall('channel')]
    fs = int(root.find('nominal_srate').text)
    return channels, fs


def get_info(f, drop_channels):
    labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
    print('fs: {}\nall labels {}: {}'.format(fs, len(labels), labels))
    channels = [label for label in labels if label not in drop_channels]
    print('selected channels {}: {}'.format(len(channels), channels))
    n_protocols = len([k for k in f.keys() if ('protocol' in k and k != 'protocol0')])
    protocol_names = [f['protocol{}'.format(j+1)].attrs['name'] for j in range(n_protocols)]
    print('protocol_names:', protocol_names)
    return fs, channels, protocol_names


def load_data(file_path):
    with h5py.File(file_path) as f:
        fs, channels, p_names = get_info(f, ['A1', 'A2', 'AUX'])
        data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]

        df = pd.DataFrame(np.concatenate(data), columns=channels)
        df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
        df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])

    return df, fs, p_names, channels


def load_signals_data(file_path):
    with h5py.File(file_path) as f:
        fs, channels, p_names = get_info(f, ['A1', 'A2', 'AUX'])
        data = [f['protocol{}/signals_data'.format(k + 1)][:] for k in range(len(p_names))]
        df = pd.DataFrame(np.concatenate(data), columns=['P4', 'Signal', 'P42', 'Signal2', 'Composite', 'Composite2'])
        df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
        df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])
    return df


def parse_channels_locations_from_mat(channels_path, used_channels=ch_names32, info=False):

    """
    parse_channels_locations(self)

    Parses channels locations from .mat file.
    """

    channels_locations = []
    channels_in_list = []
    ind_in_list = []

    temp_data = loadmat(channels_path)['chanlocs'][0]
    temp_length = temp_data.shape[0]
    for i in range(len(used_channels)):
        ind = 0
        for j in range(temp_length):
            if used_channels[i].upper() == temp_data[j][0][0].upper():
                ind = j
                channels_in_list.append(used_channels[i])
                ind_in_list.append(i)
                channels_locations.append([temp_data[ind][4][0][0], temp_data[ind][5][0][0], temp_data[ind][6][0][0]])
    channels_locations = np.array(channels_locations)

    if info:
        print("\nchannels_locations ", len(channels_locations),
              "\nchannels_in_list ", len(channels_in_list),
              "\nchannels_in_list ", len(ind_in_list), end="\n\n")

    return channels_locations, channels_in_list, ind_in_list

