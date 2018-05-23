# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with
the left/right mouse buttons. Right click on any plot to show a context menu.
"""

# import initExample ## Add path to library (just for examples; you do not need this)

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="Basic plotting examples")
# win.
win.resize(1000, 600)
win.setWindowTitle('pyqtgraph example: Plotting')

ch_names32 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
              'C4', 'T8', 'Tp9',
              'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']

channels = {}

for i in ch_names32[0:16]:

    p = win.addPlot()
    curve = p.plot(pen='y')
    p.setLabel('left', i)
    channels[p] = curve
    win.nextRow()

ptr = 0

def update():

    global curve, data, ptr, p
    curve.setData(np.random.normal(size=100))
    if ptr == 0:
        p.enableAutoRange('xy', False)  # stop auto-scaling after the first data set is plotted
    ptr += 1

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()