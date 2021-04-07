#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:57:50 2019

@author: ngritti
"""
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QDialog,
        QGridLayout, QLabel, QPushButton,
        QWidget, QSizePolicy, QSpinBox, QDoubleSpinBox)
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import sys, warnings, os, time
from skimage.io import imread
import scipy.ndimage as ndi
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches
import utils_postprocessing, utils_image
warnings.filterwarnings("ignore")
#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
# https://stackoverflow.com/questions/52581727/how-to-return-value-from-the-qwidget

class spotWindow(QDialog):
    def __init__(self, val, parent=None):
        super(spotWindow, self).__init__(parent)
        self.val = val
        self.initUI()

    def initUI(self):
        endButton = QPushButton('OK')
        endButton.clicked.connect(self.on_clicked)
        lay = QVBoxLayout(self)
        lay.addWidget(endButton)
        self.setWindowTitle(str(self.val))

    @QtCore.pyqtSlot()
    def on_clicked(self):
        self.val += 1
        self.accept()

app = QApplication(sys.argv)
# in the outside code, use
ex = spotWindow(0)
ex.show()
if ex.exec_() == QtWidgets.QDialog.Accepted:
    print(ex.val)
else:
    print('Bad exit')
'''

class spotWindow(QDialog):
    def __init__(self, input_folder, params, parent=None):
        super(spotWindow, self).__init__(parent)
        self.input_folder = input_folder
        self.params = params
        # load the first image to use for parameter definition and find out the number of channels
        _, cond = os.path.split(input_folder)
        save_folder = os.path.join(input_folder,'result_segmentation')            
        props = utils_postprocessing.load_morpho_params(save_folder, cond)
        props = {key:props[key][0] for key in props}
        mask_file = props['mask_file']
        path_to_mask = os.path.join(input_folder,mask_file)
        self.mask = imread(path_to_mask)[props['slice']].astype(np.float)
        input_file = props['input_file']
        path_to_file = os.path.join(input_folder,input_file)
        self.img = imread(path_to_file).astype(float)
        if len(self.img.shape) == 2:
            self.img = np.expand_dims(self.img,0)
        self.img = np.array([img[props['slice']] for img in self.img])
        self.n_channels = self.img.shape[0]

        # if params are none, set them to default values
        params_default = [0.8,2,0,(2,self.img.shape[1]*self.img.shape[2])]
        for i,p in enumerate(self.params):
            # if there is no channel indexing, create one if length 1
            if p==None:
                self.params[i] = [None for i in self.n_channels]
            # for every element in the channel indexing, if it is None, set it to defualt
            for ch in range(len(p)):
                if (p[ch]==None) or (p[ch]==(None,None)):
                    self.params[i][ch] = params_default[i]

        # create window
        self.initUI()
        self.updateParamsAndFigure()

    def initUI(self):
        self.figure = Figure(figsize=(10, 2.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.figure.clear()
        axs = self.figure.subplots(nrows=1, ncols=4)
        self.figure.subplots_adjust(top=0.95,right=0.95,left=0.2,bottom=0.25)
        for i in [0,1,3]:
            axs[i].axis('off')
        axs[2].set_xlabel('Fluo')
        axs[2].ticklabel_format(axis="x", style="sci", scilimits=(2,2))
        axs[2].set_ylabel('Counts')
        axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0,2))
        self.canvas.draw()

        self.channel = QSpinBox()
        self.channel.setMaximum(self.n_channels-1)
        self.channel.valueChanged.connect(self.updateChannel)
        self.channel.setAlignment(Qt.AlignRight)

        self.enhancement = QDoubleSpinBox()
        self.enhancement.setMinimum(0)
        self.enhancement.setMaximum(1)
        self.enhancement.setSingleStep(0.05)
        self.enhancement.setValue(self.params[0][self.channel.value()])
        self.enhancement.setAlignment(Qt.AlignRight)

        self.nClasses = QSpinBox()
        self.nClasses.setMinimum(2)
        self.nClasses.setValue(self.params[1][self.channel.value()])
        self.nClasses.valueChanged.connect(self.updatenThrChoice)
        self.nClasses.setAlignment(Qt.AlignRight)

        self.nThr = QSpinBox()
        self.nThr.setValue(self.params[2][self.channel.value()])
        self.nThr.setAlignment(Qt.AlignRight)

        self.minSize = QSpinBox()
        self.minSize.setMaximum(self.img.shape[1]*self.img.shape[2])
        self.minSize.setValue(self.params[3][self.channel.value()][0])
        self.minSize.setAlignment(Qt.AlignRight)

        self.maxSize = QSpinBox()
        self.maxSize.setMaximum(self.img.shape[1]*self.img.shape[2])
        self.maxSize.setValue(self.img.shape[1]*self.img.shape[2])
        self.maxSize.setAlignment(Qt.AlignRight)

        applyButton = QPushButton('Apply params')
        applyButton.clicked.connect(self.updateParamsAndFigure)

        endButton = QPushButton('UPDATE AND RETURN PARAMS')
        endButton.clicked.connect(self.on_clicked)

        lay = QGridLayout(self)
        lay.addWidget(NavigationToolbar(self.canvas, self),0,0,1,2)
        lay.addWidget(self.canvas,1,0,1,2)
        lay.addWidget(QLabel('Current channel'),2,0,1,1)
        lay.addWidget(self.channel,2,1,1,1)
        lay.addWidget(QLabel('Enhancement'),3,0,1,1)
        lay.addWidget(self.enhancement,3,1,1,1)
        lay.addWidget(QLabel('Expected classes for thresholding'),4,0,1,1)
        lay.addWidget(self.nClasses,4,1,1,1)
        lay.addWidget(QLabel('Selected threshold'),5,0,1,1)
        lay.addWidget(self.nThr,5,1,1,1)
        lay.addWidget(QLabel('Minimum spot size'),6,0,1,1)
        lay.addWidget(self.minSize,6,1,1,1)
        lay.addWidget(QLabel('Maximum spot size'),7,0,1,1)
        lay.addWidget(self.maxSize,7,1,1,1)
        lay.addWidget(applyButton,8,0,1,2)
        lay.addWidget(endButton,9,0,1,2)

        self.setWindowTitle(self.input_folder)
        QApplication.setStyle('Macintosh')

    def updatenThrChoice(self):
        self.nThr.setMaximum(self.nClasses.value()-2)

    def updateChannel(self):
        ch = self.channel.value()

        self.enhancement.setValue(self.params[0][ch])
        self.nClasses.setValue(self.params[1][ch])
        self.nThr.setValue(self.params[2][ch])
        self.minSize.setValue(self.params[3][ch][0])
        self.maxSize.setValue(self.params[3][ch][1])

        self.updateParamsAndFigure()

    def updateParamsAndFigure(self):
        from matplotlib import rc
        from matplotlib.backends.backend_pdf import PdfPages
        rc('font', size=8)
        rc('font', family='Arial')
        # rc('font', serif='Times')
        rc('pdf', fonttype=42)
        # rc('text', usetex=True)
        self.nThr.setMaximum(self.nClasses.value()-2)

        ch = self.channel.value()
        enhancement = self.enhancement.value()
        nclasses = self.nClasses.value()
        nThr = self.nThr.value()
        sizelims = (self.minSize.value(),self.maxSize.value())
        dict_, enhanced, thrs, objects = utils_image.detect_peaks(self.img[ch], self.mask, 
                            enhancement=enhancement, nclasses=nclasses, nThr=nThr, sizelims=sizelims)

        ### update the values
        self.params[0][ch] = enhancement
        self.params[1][ch] = nclasses
        self.params[2][ch] = nThr
        self.params[3][ch] = sizelims

        ### update the plot
        self.figure.clear()
        axs = self.figure.subplots(nrows=1, ncols=4)
        self.figure.subplots_adjust(top=0.9,right=1.,left=0.,bottom=0.2)#,wspace=0.01)#,hspace=0.01)
        for i in [0,1,3]:
            axs[i].axis('off')
        axs[2].set_xlabel('Fluo')
        axs[2].ticklabel_format(axis="x", style="sci", scilimits=(2,2))
        axs[2].set_ylabel('Counts')
        axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0,2))
        axs[0].set_title('Input image')
        axs[1].set_title('Enhanced image')
        axs[2].set_title('Histogram')
        axs[3].set_title('Segmented spots: %d'%len(dict_['centroid']))
        axs[2].set_yscale('log')

        axs[0].imshow(self.img[ch], cmap='magma',vmin = np.percentile(self.img[ch],0.3), vmax = np.percentile(self.img[ch],99.7))
        axs[1].imshow(enhanced, cmap='magma',vmin = np.percentile(enhanced,0.3), vmax = np.percentile(enhanced,99.7))
        n,_,_ = axs[2].hist(enhanced[self.mask>0],bins=100)
        for thr in thrs:
            axs[2].plot([thr,thr],[0,np.max(n)],'-r')
        axs[2].plot([thrs[nThr]],[np.max(n)],'*r',ms=10)
        axs[3].imshow(objects, cmap='gray')
        for coords, area in zip(dict_['centroid'],dict_['area']):
            # draw circle around segmented coins
            circle = mpatches.Circle((coords[1],coords[0]),radius=np.sqrt(area/np.pi),
                                    fc=(1,0,0,0.5), ec = (1,0,0,1), linewidth=2)
            axs[3].add_patch(circle)
            # axs[3].plot(coords[1],coords[0],'+r',ms=5,alpha=.8)
        self.canvas.draw()

    @QtCore.pyqtSlot()
    def on_clicked(self):
        self.accept()

