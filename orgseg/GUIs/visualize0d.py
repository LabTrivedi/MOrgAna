from PyQt5.QtWidgets import (QApplication, QComboBox, QGridLayout, QGroupBox, QLabel, QPushButton,
        QFileDialog, QMessageBox, QWidget, QSizePolicy, QCheckBox, QTableWidget, QVBoxLayout,
        QTableWidgetItem, QLineEdit)
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os, time
from skimage.io import imsave
import scipy.ndimage as ndi
from scipy.stats import ttest_ind
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

class visualization_0d(QWidget):
    def __init__(self, data, name, background=None, colormap='gnuplot', parent=None):
        super(visualization_0d, self).__init__(parent)

        self.data = data
        self.dataPlot = data
        self.name = name
        if not background:
            self.background = [[0 for gastruloids in group] for group in data]
        else:
            self.background = background
        self.colormap = colormap

        self.make()

    def make(self):
        self.figure = Figure(figsize=(4, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.plotType = QComboBox()
        self.plotType.addItem('bar')
        self.plotType.addItem('violinplot')
        self.plotType.addItem('boxplot')

        self.colormap = QComboBox()
        self.colormap.addItem('jet')
        self.colormap.addItem('rainbow')
        self.colormap.addItem('gnuplot')
        self.colormap.addItem('gnuplot2')
        self.colormap.addItem('brg')
        self.colormap.addItem('tab10')
        self.colormap.addItem('Spectral')
        self.colormap.addItem('coolwarm')
        self.colormap.addItem('seismic')
        self.colormap.addItem('cool')
        self.colormap.addItem('spring')
        self.colormap.addItem('summer')
        self.colormap.addItem('autumn')
        self.colormap.addItem('winter')

        self.YnormBtn = QComboBox()
        self.YnormBtn.addItem('No normalization')
        self.YnormBtn.addItem('Global percentile')
        self.YnormBtn.addItem('Group percentile')
        self.YnormBtn.addItem('Manual')

        self.bckgBtn = QComboBox()
        self.bckgBtn.addItem('None')
        self.bckgBtn.addItem('Background')

        self.pxlsize = QLineEdit()
        self.pxlsize.setValidator( QDoubleValidator(0, 1000000, 5, notation=QDoubleValidator.StandardNotation) )
        self.pxlsize.setText('1.0')
        self.dimensionality = QLineEdit()
        self.dimensionality.setValidator( QIntValidator() )
        self.dimensionality.setText('1')

        self.groupSelection = self.makeGroupSelectionBtns()

        self.applyBtn = QPushButton('Apply Settings')
        self.applyBtn.clicked.connect(self.remakePlot)

        self.ttestBtn = QPushButton('Compute statistics')
        self.ttestBtn.clicked.connect(self.computeTtest)
        
        self.savexlsxBtn = QPushButton('Save Data as xlsx')
        self.savexlsxBtn.clicked.connect(self.saveData)

        lay = QGridLayout(self)
        lay.setSpacing(10)
        lay.addWidget(NavigationToolbar(self.canvas, self),     0,0,1,2)
        lay.addWidget(self.canvas,                              1,0,1,2)
        lay.addWidget(QLabel('Y axis normalization:'),          2,0,1,1)
        lay.addWidget(self.YnormBtn,                            2,1,1,1)
        lay.addWidget(QLabel('Background subtraction type:'),      3,0,1,1)
        lay.addWidget(self.bckgBtn,                             3,1,1,1)
        lay.addWidget(QLabel('Pixel size/Scaler:'),             4,0,1,1)
        lay.addWidget(self.pxlsize,                             4,1,1,1)
        lay.addWidget(QLabel('Dimensionality:'),                5,0,1,1)
        lay.addWidget(self.dimensionality,                      5,1,1,1)
        lay.addWidget(QLabel('Plot type:'),                     6,0,1,1)
        lay.addWidget(self.plotType,                            6,1,1,1)
        lay.addWidget(QLabel('Colormap:'),                      7,0,1,1)
        lay.addWidget(self.colormap,                            7,1,1,1)
        lay.addWidget(self.groupSelection,                      8,0,1,2)
        lay.addWidget(self.applyBtn,                            9,0,1,2)
        lay.addWidget(self.ttestBtn,                            10,0,1,2)
        lay.addWidget(self.savexlsxBtn,                         11,0,1,2)

        self.remakePlot()

        self.setWindowTitle(self.name)
        QApplication.setStyle('Fusion')

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
        self.figure.subplots_adjust(top=0.95,right=0.95,left=0.15,bottom=0.15)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(2,2))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,2))
        ax.set_ylabel(self.name)

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
        data = [list(self.data[i].values) for i in range(n_groups)]

        # subtract background or not
        if self.bckgBtn.currentText() == 'Background':
            for i in range(n_groups):
                for k in range(n_gastr[i]):
                    data[i][k] -= self.background[i][k]

        # normalize fluorescence intensity accordingly
        if self.YnormBtn.currentText() == 'Global percentile':
            flat = []
            for i in range(n_groups):
                for j in range(n_gastr[i]):
                    flat.append(data[i][j])
            percs = np.percentile(np.array(flat),(.3,99.7))
            for i in range(n_groups):
                data[i] = np.clip((data[i]-percs[0])/(percs[1]-percs[0]),0.,1.)
        elif self.YnormBtn.currentText() == 'Group percentile':
            for i in range(n_groups):
                percs = np.percentile(np.array(data[i]),(.3,99.7))
                data[i] = np.clip((data[i]-percs[0])/(percs[1]-percs[0]),0.,1.)

        # use pixel size value and dimensionality
        for i in range(n_groups):
            for j in range(int(self.dimensionality.text())):
                data[i] = [v*float(self.pxlsize.text()) for v in data[i]]

        # make plot
        lines = []
        for i in range(n_groups):
            if self.groupPlotBtn[i].isChecked():
                # print(data)
                mean = np.mean(data[i])
                std= np.std(data[i])

                if self.plotType.currentText()=='bar':
                    parts = ax.bar(i,mean,yerr=std,color=colors[i])
                elif self.plotType.currentText()=='violinplot':
                    if data[i]!=[]:
                        parts = ax.violinplot(data[i],[i],showmeans=True,showextrema=True)
                        for pc in parts['bodies']:
                            pc.set_color(colors[i])
                            pc.set_alpha(0.5)
                        parts = parts['bodies']
                elif self.plotType.currentText()=='boxplot':
                    if data[i]!=[]:
                        parts = ax.boxplot(data[i], positions=[i],
                                            notch=True,
                                            patch_artist=True)
                        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                            plt.setp(parts[item], color=colors[i])
                        plt.setp(parts["boxes"], facecolor=colors[i])
                        plt.setp(parts["boxes"], alpha=.5)
                        plt.setp(parts["fliers"], markeredgecolor=colors[i])
                        parts = parts['boxes']
                lines.append(parts[0])

                if self.rawBtn.isChecked():
                    x = np.random.normal(i, 0.04, size=len(data[i]))
                    ax.plot(x,data[i],'ok',alpha=.7,ms=2)

        # adjust axes lims
        ax.set_ylim(0,None)
        group_names = []
        for i in range(n_groups):
            if self.groupPlotBtn[i].isChecked():
                group_names.append('Group'+str(i+1))
        # print(groups)
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels(group_names, rotation=15, fontsize=12)

        # add legend
        if self.legendBtn.isChecked():
            l = ax.legend(lines,group_names)
            l.get_frame().set_linewidth(0.0)

        self.dataPlot = data
        self.canvas.draw()

    def saveData(self):
        name,_ = QFileDialog.getSaveFileName(self, 'Save Data as xlsx File')
        fname, ext = os.path.splitext(name)
        if ext == '':
            name = name+'.xlsx'
        elif ext != '.xlsx':
            name = fname+'.xlsx'
        fname, ext = os.path.splitext(name)

        df = pd.DataFrame(self.dataPlot)
        df = df.transpose()
        df.columns =['Group '+str(i) for i in range(len(self.dataPlot))]
        df.to_excel(name)

    def computeTtest(self):
        n_groups = len(self.dataPlot)
        pvals = np.zeros((n_groups, n_groups))
        for i in range(n_groups):
            for j in range(n_groups):
                _, pvals[i,j] = ttest_ind(self.dataPlot[i], self.dataPlot[j])
        self.w = TtestTable(pvals)
        self.w.show()

class TtestTable(QWidget):
    def __init__(self, pvals, parent=None):
        super(TtestTable, self).__init__(parent)
        self.pvals = pvals

        self.setWindowTitle( "Ttest: P values" )
        self.createTable() 

        self.saveBtn = QPushButton('Save pvals')
        self.saveBtn.clicked.connect(self.saveData)

        self.layout = QVBoxLayout() 
        self.layout.addWidget(self.tableWidget) 
        self.layout.addWidget(self.saveBtn)

        self.setLayout(self.layout)

    def createTable(self,):
        self.tableWidget = QTableWidget()
        n_groups = self.pvals.shape[0]

        #Row count 
        self.tableWidget.setRowCount(n_groups)  
        #Column count 
        self.tableWidget.setColumnCount(n_groups)

        for i, row in enumerate(self.pvals):
            for j, val in enumerate(row):
                self.tableWidget.setItem(i,j, QTableWidgetItem('%.5f'%val))
    
    def saveData(self):
        n_groups = self.pvals.shape[0]
        name,_ = QFileDialog.getSaveFileName(self, 'Save Data as xlsx File')
        fname, ext = os.path.splitext(name)
        if ext == '':
            name = name+'.xlsx'
        elif ext != '.xlsx':
            name = fname+'.xlsx'
        fname, ext = os.path.splitext(name)

        df = pd.DataFrame(self.pvals)
        df = df.transpose()
        df.columns =['Group '+str(i) for i in range(n_groups)]
        df.index =['Group '+str(i) for i in range(n_groups)]
        # df = df.rename(index=['Group '+str(i) for i in range(len(self.dataPlot))])

        df.to_excel(name)

