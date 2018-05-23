import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from scipy.linalg import solve

from data.nfblab_data import NFBLabData

index = 0


def init_colour_map():
    """Build custom colormap"""

    # Array of positions where each color is defined
    positions = np.arange(0, 1.1, 0.1)
    # Array of RGBA colors. Integer data types are interpreted as 0-255; float data types are interpreted as 0.0-1.0
    color = np.array([[0, 0, 255, 255],
                      [0, 0, 204, 255],
                      [0, 102, 204, 255],
                      [0, 204, 204, 255],
                      [0, 204, 102, 255],
                      [0, 204, 0, 255],
                      [102, 204, 0, 255],
                      [204, 204, 0, 255],
                      [204, 102, 0, 255],
                      [204, 0, 0, 255],
                      [255, 0, 0, 255]], dtype=np.ubyte)
    map = pg.ColorMap(positions, color)
    lut = map.getLookupTable(0.0, 1.0, 256)
    return lut


def sin_generator():
    global index
    index += np.pi / 16
    return np.sin(channels * index)


class Topomap:
    """
    Prepares pixels for plot. Pixel corresponds to amplitude value.
    """
    def __init__(self, pos, res=64):
        """
        :param pos – electrode positions 2D-array:
        :param res – topography resolution in pixels:
        """
        xmin = pos[:, 0].min()
        xmax = pos[:, 0].max()
        ymin = pos[:, 1].min()
        ymax = pos[:, 1].max()
        x = pos[:, 0]
        y = pos[:, 1]
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        xi, yi = np.meshgrid(xi, yi)
        xy = x.ravel() + y.ravel() * -1j
        d = xy[None, :] * np.ones((len(xy), 1))
        d = np.abs(d - d.T)
        n = d.shape[0]  # why not pos.shape[0]?
        d.flat[::n + 1] = 1.  # substitute diagonal 0s by 1s
        g = (d * d) * (np.log(d) - 1.)
        g.flat[::n + 1] = 0.  # substitute diagonal 1s by 0s
        self.g_solver = g
        m, n = xi.shape
        xy = xy.T
        self.g_tensor = np.empty((m, n, xy.shape[0]))
        g = np.empty(xy.shape)
        for i in range(m):
            for j in range(n):
                d = np.abs(xi[i, j] + -1j * yi[i, j] - xy)
                mask = np.where(d == 0)[0]
                if len(mask):
                    d[mask] = 1.
                np.log(d, out=g)
                g -= 1.
                g *= d * d
                if len(mask):
                    g[mask] = 0.
                self.g_tensor[i, j] = g

    def get_topomap(self, v):
        """

        :param v:
        :return:
        """
        weights = solve(self.g_solver, v.ravel())
        return self.g_tensor.dot(weights)


class TopomapWidget(pg.PlotWidget):
    """
    Initialises a PyQt topography widget.
    """

    def __init__(self, pos, res=64, parent=None, autoupdate=True):

        self.lut = init_colour_map()

        # Return a proxy object that delegates method calls to a parent or sibling class of type PlotWidget.
        super(TopomapWidget, self).__init__(parent)

        # Init topomap, that prepares pixels for plot. Pixel corresponds to amplitude value.
        self.topomap = Topomap(pos, res=res)

        # Init image item for the topography
        self.img = pg.ImageItem()
        # Set colour map for topography.
        self.img.setLookupTable(lut=init_colour_map())
        self.img.setLevels([0, 1])

        # Widget settings
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.getPlotItem().getAxis('left').hide()
        self.getPlotItem().getAxis('bottom').hide()
        self.setAspectLocked(True)
        self.addItem(self.img)
        self.setMaximumWidth(res)
        self.setMaximumHeight(res)
        coef = 1
        radius = int(res * coef)
        shift = int(res * (1 - coef) / 2)
        print(shift, radius)
        self.setMask(QtGui.QRegion(QtCore.QRect(shift, shift, radius, radius), QtGui.QRegion.Rectangle))

        # Plot head contour
        x = np.linspace(0, res, res)
        self.plot(x=x, y=np.sqrt((res / 2) ** 2 - (x - res / 2) ** 2) + res / 2, fillLevel=res, brush=(0, 0, 0, 255))
        self.plot(x=x, y=-np.sqrt((res / 2) ** 2 - (x - res / 2) ** 2) + res / 2, fillLevel=0, brush=(0, 0, 0, 255))

        # Plot electrodes
        self.plot(x=pos[:, 0], y=pos[:, 1], pen=None, symbolBrush=(0, 0, 0), symbolPen='w')

        if autoupdate:
            timer = QtCore.QTimer(self)
            timer.timeout.connect(lambda: self.set_topomap(sin_generator()))
            timer.start(100)

    def set_topomap(self, data):
        tmap = self.topomap.get_topomap(data)
        self.img.setImage(tmap.T)


class ColourBarWidget(pg.PlotWidget):
    """
    Initialises a PyQt topography widget.
    """

    def __init__(self, pos, res=64, parent=None):

        # self.lut = init_colour_map()

        # Return a proxy object that delegates method calls to a parent or sibling class of type PlotWidget.
        super(ColourBarWidget, self).__init__(parent)

        self.gl = pg.GradientLegend((10, 200), (-10, 50))
        self.addItem(self.gl)
        self.gl.scale(1, -1)

        # Widget settings
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        # self.getPlotItem().getAxis('left').hide()
        # self.getPlotItem().getAxis('bottom').hide()
        self.setAspectLocked(True)
        self.setMaximumWidth(res)
        self.setMaximumHeight(res)
        # coef = 1
        # radius = int(res * coef)
        # shift = int(res * (1 - coef) / 2)
        # print(shift, radius)
        # self.setMask(QtGui.QRegion(QtCore.QRect(0, 100, 1000, 1000), QtGui.QRegion.Rectangle))


if __name__ == "__main__":
    # Parse chanel info and some sample eeg data
    path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/FB_pilot_feb2_11-01_13-36-17/"
    channels_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/chanlocs_mod.mat"
    e1 = NFBLabData(path, channels_path, parse=False, get_info=False)
    e1.parse_stream_info()
    e1.parse_channels_locations()
    xy = (e1.channels_locations[:, 0:2])

    # Init some random channels
    channels = np.random.normal(size=(len(e1.channels_locations), 1))

    # Run widget
    res = 300
    app = QtGui.QApplication([])
    pos = np.array(((xy - xy.min()) / (xy.max() - xy.min())) * res)
    ww = QtGui.QWidget()
    ll = QtGui.QHBoxLayout(ww)
    tmap_widget = TopomapWidget(pos, res)
    # cb = ColourBarWidget(pos, res)
    ll.addWidget(tmap_widget)
    # ll.addWidget(cb)
    btn = QtGui.QPushButton('next')
    btn.clicked.connect(lambda: tmap_widget.set_topomap(sin_generator()))
    ll.addWidget(btn)
    ww.show()
    app.exec_()
