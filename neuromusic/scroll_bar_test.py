import sys
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

ch_names32 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
              'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']


class Window(QtWidgets.QScrollArea):

    def __init__(self):
        super(Window, self).__init__()
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setGeometry(300, 300, 250, 150)

        self.setWidget()

        # self.layout = QtWidgets.QVBoxLayout()
        # self.setLayout(self.layout)

        # self.channels = {}
        #
        # for i in ch_names32[0:5]:
        #     p = pg.PlotWidget()
        #     self.layout.addWidget(p)
        #     curve = p.plot(pen='y')
        #     p.setLabel('left', i)
        #     self.channels[p] = curve
        #
        # self.ptr = 0
        #
        # timer = QtCore.QTimer()
        # timer.timeout.connect(self.update)
        # timer.start(50)

    def update(self):

        for p in self.channels:
            self.channels[p].setData(np.random.normal(size=100))
            if self.ptr == 0:
                p.enableAutoRange('xy', False)  # stop auto-scaling after the first data set is plotted
            self.ptr += 1


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()