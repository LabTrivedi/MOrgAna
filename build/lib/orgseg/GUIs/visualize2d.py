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
from scipy.interpolate import interp1d
import matplotlib as mpl
warnings.filterwarnings("ignore")
from matplotlib import rc
rc('font', size=12)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)

class visualization_2d(QWidget):
    def __init__(self, data, windowTitle, background=None, parent=None):
        super(visualization_2d, self).__init__(parent)

        self.data = data
        self.windowTitle = windowTitle
        if not background:
            self.background = [[[0 for row in gastruloid] for gastruloid in group] for group in data]
        else:
            self.background = background

        self.make()

    def make(self):
        self.figure = Figure(figsize=(6, 2.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.figure.clear()

        self.YnormBtn = QComboBox()
        self.YnormBtn.addItem('No normalization')
        self.YnormBtn.addItem('Global')
        self.YnormBtn.addItem('Group')
        self.YnormBtn.addItem('Single gastruloid (2d array)')
        self.YnormBtn.addItem('Single timepoint (row)')
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

        self.xlabel = QLineEdit("Space (mm)")
        self.ylabel = QLineEdit("Time (hr)")

        self.aspectRatioBtn = QCheckBox('')
        self.aspectRatioBtn.setChecked(True)

        self.groupSelection = self.makeGroupSelectionBtns()

        self.applyBtn = QPushButton('Apply Settings')
        self.applyBtn.clicked.connect(self.remakePlot)

        self.saveBtn = QPushButton('Save Tif image')
        self.saveBtn.clicked.connect(self.save_tifs)

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
        lay.addWidget(QLabel('Set axes aspect ratio to equal:'),8,0,1,1)
        lay.addWidget(self.aspectRatioBtn,                      8,1,1,1)
        lay.addWidget(QLabel("X label"),                        9,0,1,1)
        lay.addWidget(QLabel("Y label"),                        9,1,1,1)
        lay.addWidget(self.xlabel,                              10,0,1,1)
        lay.addWidget(self.ylabel,                              10,1,1,1)
        lay.addWidget(self.groupSelection,                      11,0,1,2)
        lay.addWidget(self.applyBtn,                            12,0,1,2)
        lay.addWidget(self.saveBtn,                             13,0,1,2)

        self.remakePlot()

        self.setWindowTitle(self.windowTitle)
        QApplication.setStyle('Fusion')

    def onCheckingXnormBtn(self):
        if self.XnormBtn.isChecked():
            self.alignmentBtn.setEnabled(False)
        else:
            self.alignmentBtn.setEnabled(True)

    def makeGroupSelectionBtns(self):
        group = QGroupBox("Group to visualize")
        self.groupPlotBtn = QComboBox()
        for i in range(len(self.data)):
            self.groupPlotBtn.addItem('Group '+str(i+1))
        
        self.rawBtn = QCheckBox('Plot raw data')
        self.rawBtn.setChecked(True)

        lay = QGridLayout()
        lay.addWidget(self.groupPlotBtn,i,0,1,1)

        group.setLayout(lay)
        return group

    def remakePlot(self):
        # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
        #             'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        self.figure.clear()
        axs = [ self.figure.add_subplot(121), self.figure.add_subplot(122) ]
        self.figure.subplots_adjust(top=0.9,right=0.95,left=0.15,bottom=0.2,hspace=0.1,wspace=0.3)
        for ax in axs:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(2,2))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0,2))
            ax.set_xlabel(self.xlabel.text())
            ax.set_ylabel(self.ylabel.text())

        n_groups = len(self.data)
        n_gastr = [ len(self.data[group_idx]) for group_idx in range(n_groups) ]
        n_rows = [ [len(self.data[group_idx][gastr_idx]) for gastr_idx in range(n_gastr[group_idx]) ] for group_idx in range(n_groups) ]

        # rearrange dataset: [ [ [[]],-> 2d dataset ],-> gastruloid ]-> group
        data = [[[list(k) for k in j ] for j in self.data[i].values] for i in range(n_groups)]

        # subtract background
        for i in range(n_groups):
            for k in range(n_gastr[i]):
                for l in range(n_rows[i][k]):
                    # subtract background or not to every line of the 2D dataset
                    # i.e.: for every timepoint of the kymograph
                    if self.bckgBtn.currentText() == 'Background':
                        # bckg = np.array(self.background[i][k])
                        # val = np.array(data[i][k])
                        # data[i][k] = list(val-bckg)
                        data[i][k][l] = [ val-self.background[i][k][l] for val in data[i][k][l] ]
                    if self.bckgBtn.currentText() == 'Minimum':
                        _min = np.min(data[i][k][l])
                        data[i][k][l] = [ val-_min for val in data[i][k][l] ]

        # normalize fluorescence intensity accordingly
        if self.YnormBtn.currentText() == 'Global':
            flat = []
            for i in range(n_groups):
                for j in range(n_gastr[i]):
                    for k in range(n_rows[i][j]):
                        for val in data[i][j][k]:
                            flat.append(val)
            percs = np.percentile(np.array(flat),(.3,99.7))
            for i in range(n_groups):
                for j in range(n_gastr[i]):
                    for k in range(n_rows[i][j]):
                        data[i][j][k] = np.clip((data[i][j][k]-percs[0])/(percs[1]-percs[0]),0.,1.)
        elif self.YnormBtn.currentText() == 'Group':
            for i in range(n_groups):
                flat = []
                for j in range(n_gastr[i]):
                    for k in range(n_rows[i][j]):
                        for val in data[i][j][k]:
                            flat.append(val)
                percs = np.percentile(np.array(flat),(.3,99.7))
                for j in range(n_gastr[i]):
                    for k in range(n_rows[i][j]):
                        data[i][j][k] = np.clip((data[i][j][k]-percs[0])/(percs[1]-percs[0]),0.,1.)
        elif self.YnormBtn.currentText() == 'Single gastruloid (2d array)':
            for i in range(n_groups):
                for j in range(n_gastr[i]):
                    flat = []
                    for k in range(n_rows[i][j]):
                        for val in data[i][j][k]:
                            flat.append(val)
                    percs = np.percentile(np.array(flat),(.3,99.7))
                    for k in range(n_rows[i][j]):
                        data[i][j][k] = np.clip((data[i][j][k]-percs[0])/(percs[1]-percs[0]),0.,1.)
        elif self.YnormBtn.currentText() == 'Single timepoint (row)':
            for i in range(n_groups):
                for j in range(n_gastr[i]):
                    for k in range(n_rows[i][j]):
                        percs = np.percentile(np.array(data[i][j][k]),(.3,99.7))
                        data[i][j][k] = np.clip((data[i][j][k]-percs[0])/(percs[1]-percs[0]),0.,1.)

        # normalize AP axis if necessary
        if self.XnormBtn.isChecked():
            for i in range(n_groups):
                for j in range(n_gastr[i]):
                    for k in range(n_rows[i][j]):
                        profile = data[i][j][k]
                        x = np.linspace(0,1,len(profile))
                        fun = interp1d(x,profile)
                        new_x = np.linspace(0,1,101)
                        data[i][j][k] = fun(new_x)

        # compute length of longest row
        max_length = []
        for i in range(n_groups):
            for j in range(n_gastr[i]):
                for k in range(n_rows[i][j]):
                    max_length.append(len(data[i][j][k]))
        max_length = np.max(max_length)

        # orient plots according to setting
        if self.orientationBtn.currentText() == 'Signal based':
            for i in range(n_groups):
                for j in range(n_gastr[i]):
                    for k in range(n_rows[i][j]):
                        y = np.array(data[i][j][k])[~np.isnan(data[i][j][k])]
                        n_p = len(y)
                        if np.sum(y[:int(n_p/2)])>np.sum(y[int(n_p-n_p/2):]):
                            data[i][j][k] = data[i][j][k][::-1]

        # pad array to the right or left
        for i in range(n_groups):
            for j in range(n_gastr[i]):
                for k in range(n_rows[i][j]):
                    w = max_length-len(data[i][j][k])
                    if self.alignmentBtn.currentText() == 'Left':
                        pad_width = (0,w)
                    if self.alignmentBtn.currentText() == 'Right':
                        pad_width = (w,0)
                    elif self.alignmentBtn.currentText() == 'Center':
                        if 2*int(w/2)==w:
                            pad_width = (int(w/2),int(w/2))
                        else:
                            pad_width = (int(w/2)+1,int(w/2))
                    data[i][j][k] = np.pad(data[i][j][k],pad_width,mode='constant',constant_values=np.nan)

        # plot the selected group only
        group_idx = self.groupPlotBtn.currentIndex()
        data = data[group_idx]
        n_gastr = n_gastr[group_idx]
        n_rows = n_rows[group_idx]

        # compute and plot mean and std of the selected group
        data = np.array(data)
        data_mean = np.nanmean(data,0)
        data_std = np.nanstd(data,0)
        # print(data_mean,data_std)

        aspect = 'auto'
        if self.aspectRatioBtn.isChecked():
            aspect = 'equal'
            
        im = axs[0].imshow(data_mean, aspect=aspect, vmin = np.nanmin(data_mean), vmax = np.nanmax(data_mean))
        axs[0].set_title('Mean Group '+str(i+1))
        axs[1].imshow(data_std, aspect=aspect, vmin = np.nanmin(data_mean), vmax = np.nanmax(data_mean))
        axs[1].set_title('Std Group '+str(i+1))

        self.figure.colorbar(im, ax=axs[1])        

        if self.XnormBtn.isChecked():
            axs[0].set_xlim(0,100)
            axs[1].set_xlim(0,100)

        self.tifs_data = [data_mean, data_std]

        self.canvas.draw()

    def save_tifs(self):
        titles = ['Save mean data of group1',
                'Save std data of group1']
        for img, title in zip(self.tifs_data, titles):
            name,_ = QFileDialog.getSaveFileName(self, title)
            if name != '':
                ### check file extension: allow to save in other formats, but bias towards tif
                if os.path.splitext(name)[-1]!='.tif':
                    buttonReply = QMessageBox.question(self,'File format warning!','File format not recommended. Do you want to save the image as tif?')
                    if buttonReply == QMessageBox.Yes:
                        name = os.path.splitext(name)[0]+'.tif'
                
                # convert the image into int16 with the right brightness and contrast
                if  self.percs[0]!=None:
                    self.tif_data = (2**16-1)*(self.tif_data-self.percs[0])/(self.percs[1]-self.percs[0])
                imsave(name+'', img.astype(np.uint16))


