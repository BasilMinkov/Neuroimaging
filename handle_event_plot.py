import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat
from data import parse_channels_locations_from_mat, ch_names32


def onclick(event):

    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

    a = np.identity(15) * 1.5 + np.random.random([15, 15])

    ax[0].cla()
    ax[1].cla()
    ax[0].imshow(a, cmap=colour_map)
    mne.viz.plot_topomap(np.random.random(32), ch2[:, [1, 0]], names=ch_names32, show_names=True, axes=ax[1], cmap=colour_map, show=False, contours=False)
    fig.canvas.draw()


colour_map = "viridis"
channels_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/chanlocs_mod.mat"
ch3 = parse_channels_locations_from_mat(channels_path)
ch2 = np.delete(ch3, 2, 1)

a = np.identity(15)*1.5+np.random.random([15, 15])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax[0].imshow(a, cmap=colour_map)

mne.viz.plot_topomap(np.arange(32), ch2[:, [1, 0]], names=ch_names32, show_names=True, axes=ax[1], cmap=colour_map, show=False, contours=False)
ax[1].set_title("My bro!")

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
