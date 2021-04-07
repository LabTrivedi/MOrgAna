from PyQt5.QtWidgets import (QApplication, QComboBox, QGridLayout, QGroupBox, QLabel, QPushButton,
        QFileDialog, QMessageBox, QWidget, QSizePolicy, QCheckBox, QLineEdit)
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, time
from skimage.io import imsave
import scipy.ndimage as ndi
from matplotlib.figure import Figure
import matplotlib.cm as cm
from scipy.interpolate import interp1d
import matplotlib as mpl
warnings.filterwarnings("ignore")
from matplotlib import rc
rc('font', size=12)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)

class visualization_1d(QWidget):
    def __init__(self, data, windowTitle, background=None, parent=None):
        super(visualization_1d, self).__init__(parent)

        self.data = data
        self.windowTitle = windowTitle
        if not background:
            self.background = [[0 for gastruloids in group] for group in data]
        else:
            self.background = background

        self.make()

    def make(self):
        self.figure = Figure(figsize=(4, 2.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.figure.clear()

        self.colormap = QComboBox()
        self.colormap.addItem('jet')
        self.colormap.addItem('rainbow')
        self.colormap.addItem('gnuplot')
        self.colormap.addItem('gnuplot2')
        self.colormap.addItem('brg')
        self.colormap.addItem('tab10')
        self.colormap.addItem('spectral')
        self.colormap.addItem('coolwarm')
        self.colormap.addItem('seismic')
        self.colormap.addItem('cool')
        self.colormap.addItem('spring')
        self.colormap.addItem('summer')
        self.colormap.addItem('autumn')
        self.colormap.addItem('winter')

        self.YnormBtn = QComboBox()
        self.YnormBtn.addItem('No normalization')
        self.YnormBtn.addItem('Global')
        self.YnormBtn.addItem('Group')
        self.YnormBtn.addItem('Single gastruloid')
        self.YnormBtn.addItem('Manual')

        self.XnormBtn = QCheckBox('')
        self.XnormBtn.setChecked(False)
        self.XnormBtn.stateChanged.connect(self.onCheckingXnormBtn)

        self.bckgBtn = QComboBox()
        self.bckgBtn.addItem('None')
        self.bckgBtn.addItem('Background')
        self.bckgBtn.addItem('Minimum')

        self.orientationBtn = QComboBox()
        self.orientationBtn.addItem('Signal based')
        self.orientationBtn.addItem('NO')

        self.alignmentBtn = QComboBox()
        self.alignmentBtn.addItem('Left')
        self.alignmentBtn.addItem('Right')
        self.alignmentBtn.addItem('Center')

        self.xlabel = QLineEdit("Time (hr)/Space (mm)")
        self.ylabel = QLineEdit("Fluorescence (a.u.)")

        self.groupSelection = self.makeGroupSelectionBtns()

        self.applyBtn = QPushButton('Apply Settings')
        self.applyBtn.clicked.connect(self.remakePlot)

        lay = QGridLayout(self)
        lay.setSpacing(10)
        lay.addWidget(NavigationToolbar(self.canvas, self),     0,0,1,2)
        lay.addWidget(self.canvas,                              1,0,1,2)
        lay.addWidget(QLabel('Background subtraction type:'),   2,0,1,1)
        lay.addWidget(self.bckgBtn,                             2,1,1,1)
        lay.addWidget(QLabel('Y axis normalization:'),          4,0,1,1)
        lay.addWidget(self.YnormBtn,                            4,1,1,1)
        lay.addWidget(QLabel('X axis normalization:'),          5,0,1,1)
        lay.addWidget(self.XnormBtn,                            5,1,1,1)
        lay.addWidget(QLabel('A-P orientation correction:'),    6,0,1,1)
        lay.addWidget(self.orientationBtn,                      6,1,1,1)
        lay.addWidget(QLabel('Alignment:'),                     7,0,1,1)
        lay.addWidget(self.alignmentBtn,                        7,1,1,1)
        lay.addWidget(QLabel("X label"),                        8,0,1,1)
        lay.addWidget(QLabel("Y label"),                        8,1,1,1)
        lay.addWidget(QLabel('Colormap:'),                      9,0,1,1)
        lay.addWidget(self.colormap,                            9,1,1,1)
        lay.addWidget(self.xlabel,                              10,0,1,1)
        lay.addWidget(self.ylabel,                              10,1,1,1)
        lay.addWidget(self.groupSelection,                      11,0,1,2)
        lay.addWidget(self.applyBtn,                            12,0,1,2)

        self.remakePlot()

        self.setWindowTitle(self.windowTitle)
        QApplication.setStyle('Fusion')

    def onCheckingXnormBtn(self):
        if self.XnormBtn.isChecked():
            self.alignmentBtn.setEnabled(False)
        else:
            self.alignmentBtn.setEnabled(True)

    def makeGroupSelectionBtns(self):
        group = QGroupBox("Groups to plot")
        self.groupPlotBtn = []
        for i in range(len(self.data)):
            self.groupPlotBtn.append(QCheckBox('Group '+str(i)))
            self.groupPlotBtn[-1].setChecked(True)
        
        self.legendBtn = QCheckBox('Legend')
        self.legendBtn.setChecked(False)

        self.rawBtn = QCheckBox('Plot raw data')
        self.rawBtn.setChecked(True)

        lay = QGridLayout()
        for i in range(len(self.data)):
            lay.addWidget(self.groupPlotBtn[i],i,0,1,1)
        lay.addWidget(self.legendBtn,0,1,1,1)
        lay.addWidget(self.rawBtn,1,1,1,1)

        group.setLayout(lay)
        return group

    def remakePlot(self):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(top=0.95,right=0.95,left=0.15,bottom=0.2)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(2,2))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,2))
        ax.set_xlabel(self.xlabel.text())
        ax.set_ylabel(self.ylabel.text())

        n_groups = len(self.data)
        n_gastr = [len(self.data[group_idx]) for group_idx in range(n_groups)]

        # define colors
        cmap = cm.get_cmap(self.colormap.currentText())
        if n_groups == 1:
            colors = [cmap(0)]
        elif self.colormap.currentText()=='tab10':
            colors = [cmap(i) for i in range(n_groups)]
        else:
            colors = [cmap(i/(n_groups-1)) for i in range(n_groups)]

        # rearrange dataset
        data = [[[float(k) for k in list(j)] for j in self.data[i].values] for i in range(n_groups)]

        # subtract background
        for i in range(n_groups):
            for k in range(n_gastr[i]):
                # subtract background or not
                if self.bckgBtn.currentText() == 'Background':
                    # this deals with both the situation when :
                    # 1. val=1D array (always) and bckg = constant (e.g. when looking at AP profile)
                    # 2. val=1D array (always) and bckg = 1D array (e.g. when looking at timelapse of average fluorescence)
                    bckg = np.array(self.background[i][k])
                    val = np.array(data[i][k])
                    data[i][k] = list(val-bckg)
                if self.bckgBtn.currentText() == 'Minimum':
                    _min = np.min(data[i][k])
                    data[i][k] = [ val-_min for val in data[i][k] ]

        # normalize fluorescence intensity accordingly
        if self.YnormBtn.currentText() == 'Global':
            flat = []
            for i in range(n_groups):
                for j in range(n_gastr[i]):
                    for val in data[i][j]:
                        flat.append(val)
            percs = np.percentile(np.array(flat),(.3,99.7))
            for i in range(n_groups):
                for j in range(n_gastr[i]):
                    data[i][j] = np.clip((data[i][j]-percs[0])/(percs[1]-percs[0]),0.,1.)
        elif self.YnormBtn.currentText() == 'Group':
            for i in range(n_groups):
                flat = []
                for j in range(n_gastr[i]):
                    for val in data[i][j]:
                        flat.append(val)
                percs = np.percentile(np.array(flat),(.3,99.7))
                for j in range(n_gastr[i]):
                    data[i][j] = np.clip((data[i][j]-percs[0])/(percs[1]-percs[0]),0.,1.)
        elif self.YnormBtn.currentText() == 'Single gastruloid':
            for i in range(n_groups):
                for j in range(n_gastr[i]):
                    percs = np.percentile(np.array(data[i][j]),(.3,99.7))
                    data[i][j] = np.clip((data[i][j]-percs[0])/(percs[1]-percs[0]),0.,1.)

        # normalize AP axis if necessary
        if self.XnormBtn.isChecked():
            for i in range(n_groups):
                for k in range(n_gastr[i]):
                    profile = data[i][k]
                    x = np.linspace(0,1,len(profile))
                    fun = interp1d(x,profile)
                    new_x = np.linspace(0,1,101)
                    data[i][k] = fun(new_x)

        # compute length of longest trajectory
        max_length = []
        for i in range(n_groups):
            for k in range(n_gastr[i]):
                max_length.append(len(data[i][k]))
        max_length = np.max(max_length)

        # orient plots according to setting
        if self.orientationBtn.currentText() == 'Signal based':
            for i in range(n_groups):
                for k in range(n_gastr[i]):
                    y = np.array(data[i][k])[~np.isnan(data[i][k])]
                    n_p = len(y)
                    if np.sum(y[:int(n_p/2)])>np.sum(y[int(n_p-n_p/2):]):
                        data[i][k] = data[i][k][::-1]

        # pad array to the right or left
        for i in range(n_groups):
            for k in range(n_gastr[i]):
                w = max_length-len(data[i][k])
                if self.alignmentBtn.currentText() == 'Left':
                    pad_width = (0,w)
                if self.alignmentBtn.currentText() == 'Right':
                    pad_width = (w,0)
                elif self.alignmentBtn.currentText() == 'Center':
                    if 2*int(w/2)==w:
                        pad_width = (int(w/2),int(w/2))
                    else:
                        pad_width = (int(w/2)+1,int(w/2))
                data[i][k] = list(np.pad(data[i][k],pad_width,mode='constant',constant_values=(np.nan,)))

        ### make plot
        lines = []
        for i in range(n_groups):
            # plot this group only if the button is checked
            if self.groupPlotBtn[i].isChecked():
                ydata_group = []
                for k in range(n_gastr[i]):
                    ydata_group.append(data[i][k])
                    # plot the raw data if the button is checked
                    if self.rawBtn.isChecked():
                        ax.plot(ydata_group[-1],'-', lw=.5, c=colors[i], alpha = 0.2)
                # compute and plot mean and std
                max_length = np.max([len(d) for d in ydata_group])
                _mean = np.zeros(max_length)
                _std = np.zeros(max_length)
                for j in range(max_length):
                    datapoint = []
                    for d in ydata_group:
                        datapoint.append(d[j])
                    _mean[j] = np.nanmean(datapoint)
                    _std[j] = np.nanstd(datapoint)
                line = ax.plot(_mean,'-',lw=1,c=colors[i],label='Mean')[0]
                ax.fill_between(range(len(_mean)),_mean-_std,_mean+_std,facecolor=colors[i],alpha=.2, linewidth=0.,label='Std')
                lines.append(line)

        # adjust axes lims
        ax.set_ylim(0,None)
        ax.set_xlim(0,None)
        if self.XnormBtn.isChecked():
            ax.set_xlim(0,100)
        if self.YnormBtn.currentText() != 'No normalization':
            ax.set_ylim(0,1)

        # add legend
        if self.legendBtn.isChecked():
            l = ax.legend(lines,['Group '+str(i+1) for i in range(len(self.groupPlotBtn)) if self.groupPlotBtn[i].isChecked()])
            l.get_frame().set_linewidth(0.0)

        self.canvas.draw()
