import numpy as np

def sine_wave(t, A, w, p, c):  
	return A * np.sin(w*t + p) + c

def square_wave(t, A, w, p, c):
	return np.sign(sine_wave(t, A, w, p, c))
	