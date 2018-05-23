"""
Class and functions to process data from NFBLab
"""

import os
import re
import h5py
import numpy as np
from lxml import etree
from scipy.io import loadmat

encoding = "cp1251"
path_to_protocols_in_xml = "/NeurofeedbackSignalSpecs/vPSequence/s"
path_to_references_in_xml = "/NeurofeedbackSignalSpecs/sReference"
path_to_channels_in_xml = "/info/desc/channels/channel/label"

templates_of_filenames = ["experiment_data.h5",
                                  "stream_info.xml",
                                  "settings.xml",
                                  "chanlocs_mod.mat"]


def parse_nfblab_xml(path, encoding):
    """
    parse_nfblab_xml(path, encoding)
    
    Parses NFBLab software XML tree file. Returns ElementTree object.
    
    :param path: Path to XML tree file (string).
    :param encoding: encoding of XML tree file (string).
    :return: ElementTree object, determined in xml.etree package.  
    """

    try:
        with open(path, 'r', encoding=encoding) as xml_file:
            xml = xml_file.read().replace('\n', '')
        xml_tree = etree.XML(xml)
    except ValueError:
        with open(path, 'r', encoding=encoding) as xml_file:
            xml_file.readline()  # frequent error: the first line is irrelevant.
            xml = xml_file.read().replace('\n', '')
        xml_tree = etree.XML(xml)
    except ValueError:
        print("There might be some lines irrelevant to XML tree in the file. Check it carefully.")
        raise
    return xml_tree


class NFBLabData:
    """ NFBLabData   Summary of eegData
         
     EegData represents xml-files, returned by NFBLab, as a Python object.     
    
     EegData Properties:
    
        path - Path to a folder with xml files, returned by NFBLab
        path_to_protocols_in_xml - Path to protocol in xml tree
        path_to_references_in_xml - Path to reference in xml tree
        path_to_channels_in_xml - Path to channel names in xml tree
            
        h5_filename - Name of a file with EEG data returned by NFBLab
        stream_info_filename - Name of a stream info used by NFBLab
        settings_filename - Name of a settings file info used by NFBLab
        channels_loc_filename - Name of channel locations (not used yet)
        
        protocols_list - List of protocols used by NFBLab
        references_list - List of reference electrodes used by NFBLab
        channels_list - List of channels names used by NFBLab
        nominal_srate - Sampling rate of EEG data 
       
        tree_settings - XML-tree object for settings file info
        tree_stream_info - XML-tree object
       
        special_protocols - Protocols names that should be considered 
                            (should be regular expressions)
        special_protocols_encoding - Protocols classes for special
                                     protocols
        
     eegData Methods:
        makeParsing - Parse information from files, mentioned in
                      properties
        getUsefulProtocolsList - Return usefulProtocolsList ( list of 
                                 protocols that will be considered somehow
                                 in futer processing), numberProtocolList
                                 (sequence of protocol numbers),
                                 encodedProtocolList (sequence of protocol
                                 clases)
    """

    def __init__(self, path, channels_path, parse=True, get_info=True, load=True, mode="whole",
                 template_of_useful_protocols=None):

        self.path = path  # path to data folder
        self.channels_path = channels_path
        self.h5_filename = templates_of_filenames[0]  # path to data file in the folder
        self.stream_info_filename = templates_of_filenames[1]  # path to stream info file in the folder
        self.settings_filename = templates_of_filenames[2]  # path to templates file in the folder
        self.channels_loc_filename = templates_of_filenames[3]  # path to channel localization

        self.protocols_list = []
        self.channels_list = []
        self.references_list = []
        self.channels_locations = []
        self.channels_count = 0
        self.nominal_srate = 500
        self.template_of_useful_protocols = template_of_useful_protocols or [[re.compile("FBR"), 1],
                                                                             [re.compile("FBM"), 0]]

        self.useful_protocols_list = []

        self.combine_flag = True

        # Load data variables
        self.data_list = []  # Contains a list of protocols.
        # Each object's index correspond to the same index in self.useful_protocols list.
        self.data_set = np.array([])  # Contains full data set.
        self.data_set_indeces = []  # Contains protocol starting points in self.data_set.

        if parse:
            self.parse_settings()
            self.parse_stream_info()
            self.parse_channels_locations()
            self.get_useful_protocols_list()

        if get_info:
            print(self.__str__())

        if load:
            self.load_data(mode)

    def __str__(self):

        info_string = "Protocols: {}\n" \
                      "Useful Protocols: {}\n" \
                      "Channels: {}\n" \
                      "References: {}\n" \
                      "Channels Locations: {}\n" \
                      "Channels Count: {}\n" \
                      "Sampling Rate: {}\n".format(self.protocols_list, self.useful_protocols_list, self.channels_list,
                                                   self.references_list, self.channels_locations, self.channels_count,
                                                   self.nominal_srate, self.channels_locations)

        return info_string

    def parse_settings(self):

        """
        parse_settings(self):

        Parses settings from XML file.  
        """

        xml_tree = parse_nfblab_xml(os.path.join(self.path, self.settings_filename), encoding=encoding)
        nodes = xml_tree.xpath(path_to_protocols_in_xml)
        self.protocols_list = [i.text for i in nodes]  # list contains protocol names of "str" type.

        try:
            self.references_list = xml_tree.xpath(path_to_references_in_xml)[0].text.split(", ")
        except AttributeError:
            print("Empty reference list!")

    def parse_stream_info(self):

        """
        def parse_stream_info(self):

        Parses stream info from XML tree. 
        """

        xml_tree = parse_nfblab_xml(os.path.join(self.path, self.stream_info_filename), encoding)
        nodes = xml_tree.xpath(path_to_channels_in_xml)
        self.channels_list = [i.text for i in nodes if i.text not in self.references_list]
        self.channels_count = len(self.channels_list)
        self.nominal_srate = int(xml_tree.xpath("/info/nominal_srate")[0].text)

    def parse_channels_locations(self):

        """
        parse_channels_locations(self)
        
        Parses channels locations from .mat file.
        """

        temp_data = loadmat(self.channels_path)['chanlocs'][0]
        temp_length = temp_data.shape[0]
        for i in range(self.channels_count):
            ind = 0
            for j in range(temp_length):
                if self.channels_list[i].upper() == temp_data[j][0][0].upper():
                    ind = j
                    break
            self.channels_locations.append([temp_data[ind][4][0][0], temp_data[ind][5][0][0], temp_data[ind][6][0][0]])
        self.channels_locations = np.array(self.channels_locations)
        self.combine_flag = False

    def get_useful_protocols_list(self):

        """
        get_useful_protocols_list(self)
        
        Returns list of useful protocols, that will be considered somehow in future processing.
        First value – name of protocol.
        Second value – protocol number.
        Third value - number of useful protocol.
        """

        for i in range(len(self.protocols_list)):
            for template in self.template_of_useful_protocols:
                if re.search(template[0], self.protocols_list[i]):
                    self.useful_protocols_list.append([self.protocols_list[i], i + 1, template[1]])

    def load_data(self, mode="whole", inner_path="raw_data"):

        """
        load_data(self, mode="merged", inner_path="raw_data")

        :param mode: Merged mode will import full data set. Split mode will import data splited into protocols.
                     Each object's index correspond to the same index in self.useful_protocols list.
        :param inner_path: Path to the specific data in xml file (raw_data, etc.)
        :return: 
        """

        h5_file = h5py.File(os.path.join(self.path, self.h5_filename), 'r')

        if mode is "useful":
            for i in range(len(self.useful_protocols_list)):
                temp_str = "/protocol{}/".format(str(self.useful_protocols_list[i][1]))
                self.data_list.append(np.array(h5_file.get(temp_str + inner_path)))

        elif mode is "useful_merged":
            self.data_set = h5_file.get("/protocol{}/".format(self.useful_protocols_list[0][1]) + inner_path)
            for i in range(len(self.useful_protocols_list[1:])):
                self.data_set = np.append(self.data_set,
                                          h5_file.get("/protocol{}/".format(
                                              str(self.useful_protocols_list[i][1])) + inner_path))

        elif mode is "whole":
            self.data_set = h5_file.get("/protocol1/" + inner_path)
            for i in range(1, len(self.protocols_list)):
                self.data_set = np.append(self.data_set, h5_file.get("/protocol{}/".format(i) + inner_path), axis=0)

        h5_file.close()
        del h5_file

    def save_data(self, inner_path, new_filename=None, compression_type="gzip", compression_level=9,
                  rewrite_original=True):
        """
        save_data(self, inner_path, new_filename=None, compression_type="gzip", compression_level=9,
                  rewrite_original=True)

        :param inner_path:
        :param new_filename:
        :param compression_type:
        :param compression_level:
        :param rewrite_original:
        :return:
        """

        if new_filename is None:
            new_filename = self.h5_filename
        upl = self.useful_protocols_list
        temp_str = ""
        if rewrite_original:
            os.remove(os.path.join(self.path, new_filename))

        h5_file = h5py.File(os.path.join(self.path, new_filename), 'a')

        channels_string = ','.join(self.channels_list)
        print(self.data_list, self.channels_list)  ###############
        if self.data_list[0].shape[1] != len(self.channels_list):
            multiplier = 1
            if self.data_list[0].shape[1] % 2 == 0:
                inc = 0
                for i in range(self.data_list[0].shape[1] % 2, self.data_list[0].shape[1] - 1):
                    if np.array_equal(self.data_list[0][:, i], self.data_list[0][:, i + 1]):
                        inc += 1
                    if inc > 2:
                        multiplier = 2
                        break
            channels_string = ','.join(["Comp " + str(i + 1) for i in range(int(self.data_list[0].shape[1] / \
                                                                                multiplier))])

        for j in range(len(self.useful_protocols_list)):
            group = h5_file.require_group("/protocol" + str(upl[j][1]) + "/" + temp_str)
            group.attrs.create("channels_count", np.string_(channels_string))
            group.attrs.create("class_mark", upl[j][2])
            for i in range(sum(self.data_spaces_in_each_group[:j]),
                           sum(self.data_spaces_in_each_group[:j]) + self.data_spaces_in_each_group[j]):
                group.create_dataset(inner_path + str(i),
                                     data=self.data_list[i],
                                     compression=compression_type,
                                     compression_opts=compression_level)
        h5_file.flush()
        h5_file.close()
