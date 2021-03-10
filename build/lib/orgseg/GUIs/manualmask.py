#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:57:50 2019

@author: ngritti
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QDialog, QPushButton)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import sys, warnings, os
from skimage.io import imread, imsave
import matplotlib as mpl
from matplotlib.path import Path as MplPath
warnings.filterwarnings("ignore")

class makeManualMask(QDialog):
    def __init__(self, file_in,subfolder='result_segmentation',fn=None, parent=None, wsize = (1000,1000)):
        super(makeManualMask, self).__init__(parent)
        self.setWindowTitle('Manual mask: '+file_in)
        QApplication.setStyle('Fusion')
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

        self.file_in = file_in
        self.subfolder = subfolder
        self.fn = fn
        img = imread(file_in)
        if len(img.shape)==2:
            img = np.expand_dims(img,0)
        if img.shape[-1] == np.min(img.shape):
            img = np.moveaxis(img, -1, 0)
        self.img = img[0]
        self.x = []
        self.y = []
        
        # a figure instance to plot on
        self.figure = Figure()
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        # self.toolbar = NavigationToolbar(self.canvas, self)

        self.plotImage()

        # Just some button connected to `plot` method
        self.button = QPushButton('Save mask')
        self.button.clicked.connect(self.saveMask)

        # set the layout
        layout = QVBoxLayout()
        # layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.resize(wsize[0],wsize[1])

        self.__cid2 = self.canvas.mpl_connect('button_press_event', self.__button_press_callback)

    def plotImage(self):
        ''' plot some random stuff '''
        # create an axis
        self.ax = self.figure.add_subplot(111)
        # discards the old graph
        self.ax.clear()
        # plot data
        self.ax.imshow(self.img, cmap='gray', vmin=np.percentile(self.img,1.), vmax=np.percentile(self.img,99.))

        self.line = None#ax.plot([],[],'-r')

        # refresh canvas
        self.canvas.draw()
        
    def saveMask(self):
        ny, nx = np.shape(self.img)
        poly_verts = ([(self.x[0], self.y[0])]
                      + list(zip(reversed(self.x), reversed(self.y))))
        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        roi_path = MplPath(poly_verts)
        mask = 1*roi_path.contains_points(points).reshape((ny, nx))

        folder, filename = os.path.split(self.file_in)
        filename, extension = os.path.splitext(filename)
        if self.fn==None:
            self.fn = filename+'_manual'+extension
        imsave(os.path.join(folder,self.subfolder,self.fn), mask.astype(np.uint16))
        
        self.close()
        
    def __button_press_callback(self, event):
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            ax = event.inaxes
            n_p = len(self.x)
            self.ax.clear()
            self.ax.imshow(self.img, cmap='gray', vmin=np.percentile(self.img,1.), vmax=np.percentile(self.img,99.))
            if (event.button == 1) and (event.dblclick is False):

                self.x.append(x)
                self.y.append(y)

                self.line = ax.plot(self.x, self.y,'-or')

            elif (event.button == 3) and (n_p>1):
                self.x = self.x[:-1]
                self.y = self.y[:-1]
                self.line = ax.plot(self.x, self.y,'-or')

            elif (event.button == 3) and (n_p==1):
                self.x = []
                self.y = []
                self.line = None

            elif (((event.button == 1) and (event.dblclick is True)) and (n_p>2)):
                # Close the loop and disconnect
                self.x = self.x[:-1]
                self.y = self.y[:-1]
                self.x.append(x)
                self.x.append(self.x[0])
                self.y.append(y)
                self.y.append(self.y[0])
                self.line = ax.plot(self.x, self.y, '-or')
                # self.canvas.mpl_connect(self.__cid2)
            self.canvas.draw()

