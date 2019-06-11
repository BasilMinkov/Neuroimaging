import os 
import sys
import mne 
import pandas as pd

sys.path.insert(0, '/Users/wassilyminkow/Scripts/Python3/')
from Neuroinformatics.math.pandas.statistics import *


def get_subfolder_file_dict(path: str) -> dict:
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


def plot_raw_data_iteratively(paths: dict) -> None:
	for key, value in paths.items():
	    for name in value:
	        raw_data = mne.io.read_raw_edf(name)    
	        raw_data.plot()


if __name__ == "__main__":
	# path = "/Users/wassilyminkow/Data/Hamster_Data/hamster_lactate/1_D-L"
	# plot_raw_data_iteratively(get_subfolder_file_dict(path))
	raw_data = mne.io.read_raw_edf('/Users/wassilyminkow/Data/Hamster_Data/hamster_lactate/1_D-L/RecV3_2_ang+/01-02-2019_14-22.bdf')
	df = pd.DataFrame(raw_data.get_data(), columns=raw_data.info["ch_names"])
	info = mne.create_info(ch_names=list(df.index[:2]), sfreq=250)
	raw = mne.io.RawArray(df.iloc[:2, :], info)
	raw.plot(block=True)