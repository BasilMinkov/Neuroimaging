import numpy as np
from sklearn.decomposition import FastICA
from scipy.io import wavfile

fs1, data1 = wavfile.read('/Users/basilminkov/Desktop/techno/.wav')
fs2, data2 = wavfile.read('rss_mB.wav')

df = np.stack((data1, data2), axis=0)

ica = FastICA()

S_ = ica.fit_transform(df.T)  # Get the estimated sources

wavfile.write("first.wav", fs1, S_[:, 0])
wavfile.write("second.wav", fs2, S_[:, 1])