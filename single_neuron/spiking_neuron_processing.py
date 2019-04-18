import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from scipy.signal import find_peaks


def highlight_local_max(x): 
    
    thr = x["Trace #1"].diff().abs() > 1
    color = np.zeros(x.shape[0])

    for i, logical in enumerate(thr):
            if logical:
                color[i] = i 
            else:
                color[i] = NAN 
                
    plt.fill_between(color[o:n], np.ones(color.shape[0])[o:n]*40, np.ones(color.shape[0])[o:n]*-40, facecolor="red")
    plt.plot(x["Trace #1"])
    plt.show()

def find_spikes(x, peaks):

    enter = []
    exit = []

    en_b = 0
    ex_b = 0

    for peak in peaks:
        en_b, ex_b = peak, peak
        while x["Trace #1"][en_b-6:en_b].diff().abs().mean() > 1:
            en_b -= 1
        while x["Trace #1"][ex_b:ex_b+6].diff().abs().mean() > 1:
            ex_b += 1
        enter.append(en_b)
        exit.append(ex_b)

    return np.array(enter), np.array(exit)

def find_spikes_poor_version(x, peaks):

    thr = x["Trace #1"].diff().abs() > 1
    enter = []
    exit = []

    en_b = 0
    ex_b = 0
    peak_id = 0
    
    for i in np.arange(1, x.shape[0]-1):
        if not thr[i-1] and thr[i] and thr[i+1] and x["Trace #1"][i] < -15:
            en_b = i
        if thr[i-1] and thr[i] and not thr[i+1] and x["Trace #1"][i] < -15:
            ex_b = i
            if en_b < peaks[peak_id] < ex_b:
                enter.append(en_b)
                exit.append(ex_b)
                peak_id += 1
                if peak_id >= peaks.shape[0]:
                    return np.array(enter), np.array(exit)
                
    return np.array(enter), np.array(exit)

def highlight_peaks(x, peaks):
    plt.plot(x["Trace #1"])
    plt.plot(peaks, x["Trace #1"][peaks], "x")
    plt.plot(np.zeros_like(x["Trace #1"]), "--", color="gray")
    plt.show()

def highlight_spikes(x, enter, exit):
    for i, _ in enumerate(enter):
        plt.axvline(x=enter[i], color="red")
        plt.axvline(x=exit[i], color="red")    
    x["Trace #1"].plot()
    plt.show()

def spike_stats(x, enter, peaks, exit):
    
    spike_frequency =        1/(x["Time (ms)"][peaks].diff().mean()*1e-3)
    spike_lengths =          x["Time (ms)"][exit].values-x["Time (ms)"][enter].values
    depolarisation_lengths = x["Time (ms)"][peaks].values-x["Time (ms)"][enter].values
    repolarization_lengths = x["Time (ms)"][exit].values-x["Time (ms)"][peaks].values
    amplitude_changes =      x["Trace #1"][peaks].values-x["Trace #1"][enter].values
    spike_powers =           np.array([(x["Trace #1"][enter[i]:exit[i]]**2).sum() for i in range(len(enter))])
    
    stat_text = """
        Average spike frequency:                        {} Hz
        Average spike length:                           {} ms
        Average depolarisation length:                  {} ms
        Average repolarization length:                  {} ms
        Average amplitude change during depolarisation: {} mV
        Average spike power:                            {} mW
        """.format(
        round(spike_frequency, 3),
        round(spike_lengths.mean(), 3),
        round(depolarisation_lengths.mean(), 3),
        round(repolarization_lengths.mean(), 3),
        round(amplitude_changes.mean(), 3),
        round(spike_powers.mean()*1e-3, 3),
    )
    
    print(stat_text)
    
    return (
        spike_frequency, 
        spike_lengths, 
        depolarisation_lengths, 
        repolarization_lengths, 
        amplitude_changes, 
        spike_powers
    )