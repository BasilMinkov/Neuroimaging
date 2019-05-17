from scipy import signal
import matplotlib.pyplot as plt


def welsh_comparison(df, considered_protocols, channels, fs):

    for channel in range(len(channels)):
        f, Pxx_den_open = signal.welch(df.loc[df['block_name'] == considered_protocols[0]][channels[channel]], fs, nperseg=1000)
        f, Pxx_den_closed = signal.welch(df.loc[df['block_name'] == considered_protocols[1]][channels[channel]], fs, nperseg=1000)
        plt.xlim([0, 40])
        plt.plot(f, Pxx_den_closed - Pxx_den_open, label=channels[channel])
    plt.title("Protocol {} - Protocol {}".format(considered_protocols[1], considered_protocols[0]))
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
