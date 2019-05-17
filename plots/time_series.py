import matplotlib.pyplot as plt


def plot_time_series(df, window, channels):

    for i in range(df.shape[1]):
        plt.subplot(len(channels)/2, 2, i+1)
        plt.title(i)
        plt.plot(df[window, i])
    plt.show()
