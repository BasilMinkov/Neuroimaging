import os 
import sys
import mne 
import pandas as pd

sys.path.insert(0, '/Users/wassilyminkow/Scripts/Python3/')
from Neuroinformatics.math.pandas.statistics import *


def get_files_from_subfolders_as_dict(path: str) -> dict:
	paths = {}
	for folder in os.walk(path):
	    key = folder[0].split(os.path.sep)[-1]
	    li = []
	    for file in folder[2:][0]:
	        if file.split('.')[-1] == "bdf":
	            li.append(os.path.join(folder[0], file))
	    if len(li) > 0:
	        paths[key] = li
	return paths


def plot_raw_data_iteratively_from_dict(paths: dict) -> None: 
	for key, value in paths.items():
		for name in value:
			plot_eeg_acc(name)

			
def plot_eeg_acc(name):
	raw_data = mne.io.read_raw_edf(name)
	info = mne.create_info(ch_names=raw_data.info["ch_names"], sfreq=raw_data.info["sfreq"], ch_types=["eeg", "eeg", "bio"])
	raw = mne.io.RawArray(raw_data.get_data(), info)
	scalings = {'eeg': 0.001, 'bio': 1000}
	# scalings = 'auto'
	raw.plot(n_channels=3, scalings=scalings, title=name, show=True, block=True)


if __name__ == "__main__":
	paths = get_files_from_subfolders_as_dict('/Users/wassilyminkow/Data/Hamster_Data/hamster_lactate/1_D-L/')
	print(paths)
	plot_raw_data_iteratively_from_dict(paths)
	# plot_eeg_acc("/Users/wassilyminkow/Data/Hamster_Data/hamster_lactate/1_D-L/RecV3_1_ang-/01-02-2019_14-21.bdf")