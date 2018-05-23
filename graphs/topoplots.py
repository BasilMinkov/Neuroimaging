import mne
import numpy as np
import matplotlib.pyplot as plt


class PlotSignificanceAndTopography:

    def __init__(self, p_value, vectors_list, values_list, frequencies, channels_in_list, coordinates):

        self.vectors_list = vectors_list
        self.values_list = values_list
        self.coordinates = coordinates
        self.channels_in_list = channels_in_list
        self.frequencies = frequencies
        self.p_value = p_value
        self.colour_map = "PRGn"

        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        a = self.ax[0].imshow(self.p_value, cmap=self.colour_map)
        self.ax[0].set_xlabel("Component")
        self.ax[0].set_ylabel("Frequency Band")
        # self.ax[0].set_xticks([0, 1])
        # self.ax[0].set_xticklabels(["min", "max"])
        self.ax[0].set_yticks(np.arange(0, frequencies.shape[0]))
        self.ax[0].set_yticklabels(frequencies)
        plt.colorbar(a)
        mne.viz.plot_topomap(np.zeros(len(self.channels_in_list)), self.coordinates[:, [1, 0]],
                             names=self.channels_in_list,
                             show_names=True,
                             axes=self.ax[1], cmap=self.colour_map,
                             show=False, contours=0)
        self.ax[1].set_title(("Frequency {} Hz; Component {}".format("?", "?")))
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        print('%s click: x = {}, y = {}'.format(event.xdata, event.ydata))
        # if event.xdata >= 1:
        #     event.xdata = (len(channels_in_list) - 1) + (event.xdata - 1)
        #     print(event.xdata)
        self.ax[1].cla()
        mne.viz.plot_topomap(self.vectors_list[int(event.ydata), :, int(event.xdata)], self.coordinates[:, [1, 0]],
                             names=self.channels_in_list, show_names=True, axes=self.ax[1], cmap=self.colour_map,
                             show=False, contours=0)
        # plt.colorbar(self.ax[1])
        self.ax[1].set_title("Frequency {} Hz; Component {}; Eig {}".format(
            self.frequencies[int(event.ydata)],
            int(event.xdata),
            round(self.values_list[int(event.ydata), int(event.xdata)], 2)))
        self.fig.canvas.draw()


def plot_topomaps(mixing_matrix, channels_in_list, coordinates):

    mm_shape = mixing_matrix.shape[0]
    fig, ax = plt.subplots(nrows=int(mm_shape/5), ncols=5)
    counter = 0
    for i in range(6):
        for j in range(5):
            mne.viz.plot_topomap(data=mixing_matrix[:, counter],
                                 pos=coordinates[:, [1, 0]],
                                 names=channels_in_list,
                                 show_names=False,
                                 axes=ax[i][j],
                                 cmap="PiYG",
                                 contours=0,
                                 show=False)
            ax[i][j].set_title(counter)
            counter += 1
    plt.show()


if __name__ == "__main__":

    import numpy as np

    names = ['p_value',
             'p_value_rightsided',
             'p_value_leftsided',
             'vectors_list',
             'values_list',
             'frequencies',
             "ch2",
             "channels_in_list"]

    vars = []

    for i in range(len(names)):
        vars.append(np.load("/Users/basilminkov/Scripts/python3/Neuroimaging/results/eye_test/{}.npy".format(names[i])))

    p_value = vars[0]
    p_value_rightsided = vars[1]
    p_value_leftsided = vars[2]
    vectors_list = vars[3]
    values_list = vars[4]
    frequencies = vars[5]
    channels_in_list = vars[7]
    ch2 = vars[6]

    print(p_value_rightsided)



    p_value_rightsided[(p_value_rightsided > 0.05) & (p_value_rightsided < 0.95)] = 0.5
    # p_value_leftsided[p_value_leftsided > 0.05] = 1

    plot = PlotSignificanceAndTopography(p_value_rightsided, vectors_list, values_list, frequencies, channels_in_list, ch2)
