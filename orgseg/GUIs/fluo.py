from PyQt5.QtWidgets import (QApplication, QComboBox, QGridLayout, QGroupBox, QLabel, QPushButton,
        QFileDialog, QMessageBox, QWidget, QSizePolicy, QCheckBox)
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
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


class profileAP_condMode(QWidget):
    def __init__(self, data_all, channel, colors, profileType='APprofile', parent=None, ylabel='Intensity (a.u.)'):
        super(profileAP_condMode, self).__init__(parent)

        self.data_all = data_all
        self.channel = channel
        self.colors = colors
        self.profileType = profileType
        self.ylabel = ylabel

        self.make()

    def make(self):
        self.figure = Figure(figsize=(4, 2.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(top=0.95,right=0.95,left=0.2,bottom=0.25)
        ax.set_xlabel(self.profileType)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(2,2))
        ax.set_ylabel(self.ylabel)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,2))
        # ax.axis('off')
        self.canvas.draw()

        self.YnormBtn = QComboBox()
        self.YnormBtn.addItem('No normalization')
        self.YnormBtn.addItem('Global percentile')
        self.YnormBtn.addItem('Group percentile')
        self.YnormBtn.addItem('Folder percentile')
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

        self.groupSelection = self.makeGroupSelectionBtns()

        self.applyBtn = QPushButton('Apply Settings')
        self.applyBtn.clicked.connect(self.remakePlot)

        lay = QGridLayout(self)
        lay.setSpacing(10)
        lay.addWidget(NavigationToolbar(self.canvas, self),0,0,1,2)
        lay.addWidget(self.canvas,1,0,1,2)
        lay.addWidget(QLabel('Background subtraction type:'),2,0,1,1)
        lay.addWidget(self.bckgBtn,2,1,1,1)
        lay.addWidget(QLabel('Y axis normalization:'),4,0,1,1)
        lay.addWidget(self.YnormBtn,4,1,1,1)
        lay.addWidget(QLabel('X axis normalization:'),5,0,1,1)
        lay.addWidget(self.XnormBtn,5,1,1,1)
        lay.addWidget(QLabel('A-P orientation correction:'),6,0,1,1)
        lay.addWidget(self.orientationBtn,6,1,1,1)
        lay.addWidget(QLabel('Alignment:'),7,0,1,1)
        lay.addWidget(self.alignmentBtn,7,1,1,1)
        lay.addWidget(self.groupSelection,8,0,1,2)
        lay.addWidget(self.applyBtn,9,0,1,2)

        self.remakePlot()

        self.setWindowTitle('Channel')
        QApplication.setStyle('Fusion')

    def onCheckingXnormBtn(self):
        if self.XnormBtn.isChecked():
            self.alignmentBtn.setEnabled(False)
        else:
            self.alignmentBtn.setEnabled(True)

    def makeGroupSelectionBtns(self):
        group = QGroupBox("Groups to plot")
        self.groupPlotBtn = []
        for i in range(len(self.data_all)):
            self.groupPlotBtn.append(QCheckBox('Group '+str(i)))
            self.groupPlotBtn[-1].setChecked(True)
        
        self.legendBtn = QCheckBox('Legend')
        self.legendBtn.setChecked(False)

        self.rawBtn = QCheckBox('Plot raw data')
        self.rawBtn.setChecked(True)

        lay = QGridLayout()
        for i in range(len(self.data_all)):
            lay.addWidget(self.groupPlotBtn[i],i,0,1,1)
        lay.addWidget(self.legendBtn,0,1,1,1)
        lay.addWidget(self.rawBtn,1,1,1,1)

        group.setLayout(lay)
        return group

    def remakePlot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(top=0.95,right=0.95,left=0.2,bottom=0.25)
        ax.set_xlabel(self.profileType)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(2,2))
        ax.set_ylabel(self.ylabel)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,2))
        # ax.axis('off')

        n_groups = len(self.data_all)
        n_folders = [len(self.data_all[group_idx]) for group_idx in range(n_groups)]
        n_gastr = [[len(self.data_all[group_idx][folder_idx]['input_file']) for folder_idx in range(n_folders[group_idx])] for group_idx in range(n_groups)]

        # rearrange dataset
        profiles_all = [[[0 for k in range(n_gastr[i][j])] for j in range(n_folders[i])] for i in range(n_groups)]
        for i in range(n_groups):
            for j in range(n_folders[i]):
                for k in range(n_gastr[i][j]):
                    profiles_all[i][j][k] = np.array(self.data_all[i][j][self.profileType][k][self.channel])
                    # subtract background or not
                    if self.bckgBtn.currentText() == 'Background':
                        profiles_all[i][j][k] -= self.data_all[i][j]['Background'][k][self.channel]
                    if self.bckgBtn.currentText() == 'Minimum':
                        profiles_all[i][j][k] -= np.min(profiles_all[i][j][k])

        # normalize fluorescence intensity accordingly
        if self.YnormBtn.currentText() == 'Global percentile':
            flat = []
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        for l in profiles_all[i][j][k]:
                            flat.append(l)
            percs = np.percentile(np.array(flat),(.3,99.7))
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        profile = np.array(profiles_all[i][j][k])
                        profiles_all[i][j][k] = np.clip((profile-percs[0])/(percs[1]-percs[0]),0,1.)
        elif self.YnormBtn.currentText() == 'Group percentile':
            flat = [[]for i in range(n_groups)]
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        for l in profiles_all[i][j][k]:
                            flat[i].append(l)
            percs = [np.percentile(np.array(f),(.3,99.7)) for f in flat]
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        # print(percs[i])
                        profile = np.array(profiles_all[i][j][k])
                        profiles_all[i][j][k] = np.clip((profile-percs[i][0])/(percs[i][1]-percs[i][0]),0,1.)
        elif self.YnormBtn.currentText() == 'Folder percentile':
            flat = [[[] for j in range(n_folders[i])] for i in range(n_groups)]
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        for l in profiles_all[i][j][k]:
                            flat[i][j].append(l)
            percs = [[np.percentile(np.array(f),(.3,99.7)) for f in ff] for ff in flat]
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        # print(percs[i][j])
                        profile = np.array(profiles_all[i][j][k])
                        profiles_all[i][j][k] = np.clip((profile-percs[i][j][0])/(percs[i][j][1]-percs[i][j][0]),0,1.)
            
        # normalize AP axis if necessary
        if self.XnormBtn.isChecked():
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        profile = profiles_all[i][j][k]
                        x = np.linspace(0,1,len(profile))
                        fun = interp1d(x,profile)
                        new_x = np.linspace(0,1,101)
                        profiles_all[i][j][k] = fun(new_x)

        # compute length of longest gastruloid
        max_length = []
        for i in range(n_groups):
            for j in range(n_folders[i]):
                for k in range(n_gastr[i][j]):
                    max_length.append(len(profiles_all[i][j][k]))
        max_length = np.max(max_length)

        # orient plots according to setting
        if self.orientationBtn.currentText() == 'Signal based':
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        y = np.array(profiles_all[i][j][k])[~np.isnan(profiles_all[i][j][k])]
                        n_p = len(y)
                        if np.sum(y[:int(n_p/2)])>np.sum(y[int(n_p-n_p/2):]):
                            profiles_all[i][j][k] = profiles_all[i][j][k][::-1]

        # pad array to the right or left
        for i in range(n_groups):
            for j in range(n_folders[i]):
                for k in range(n_gastr[i][j]):
                    w = max_length-len(profiles_all[i][j][k])
                    if self.alignmentBtn.currentText() == 'Left':
                        pad_width = (0,w)
                    if self.alignmentBtn.currentText() == 'Right':
                        pad_width = (w,0)
                    elif self.alignmentBtn.currentText() == 'Center':
                        if 2*int(w/2)==w:
                            pad_width = (int(w/2),int(w/2))
                        else:
                            pad_width = (int(w/2)+1,int(w/2))
                    profiles_all[i][j][k] = np.pad(profiles_all[i][j][k],pad_width,mode='constant',constant_values=np.nan)

        ### make plot
        lines = []
        for i in range(n_groups):
            # plot this group only if the button is checked
            if self.groupPlotBtn[i].isChecked():
                ydata_group = []
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        ydata_group.append(profiles_all[i][j][k])
                        # plot the raw data if the button is checked
                        if self.rawBtn.isChecked():
                            ax.plot(ydata_group[-1],'-', lw=.5, c=self.colors[i], alpha = 0.2)
                # compute and plot mean and std
                max_length = np.max([len(d) for d in ydata_group])
                _mean = np.zeros(max_length)
                _std = np.zeros(max_length)
                for j in range(max_length):
                    datapoint = []
                    for data in ydata_group:
                        datapoint.append(data[j])
                    _mean[j] = np.nanmean(datapoint)
                    _std[j] = np.nanstd(datapoint)
                line = ax.plot(_mean,'-',lw=1,c=self.colors[i],label='Mean')[0]
                ax.fill_between(range(len(_mean)),_mean-_std,_mean+_std,facecolor=self.colors[i],alpha=.2, linewidth=0.,label='Std')
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

class profileAP_tlMode(QWidget):
    #############
    # TO BE IMPLEMENTED!!!
    #############
    def __init__(self, data_all, channel, colors, profileType='APprofile', parent=None):
        super(profileAP_tlMode, self).__init__(parent)

        self.data_all = data_all
        self.n_groups = len(data_all)
        self.channel = channel
        self.colors = colors
        self.profileType = profileType

        self.make()

    def make(self):

        self.figure = Figure(figsize=(4, 2.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(top=0.95,right=0.95,left=0.2,bottom=0.25)
        ax.set_xlabel(self.profileType)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(2,2))
        ax.set_ylabel('Time')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,2))
        # ax.axis('off')
        self.canvas.draw()

        ###############################################
        settings_group = QGroupBox('Plot settings')

        self.YnormBtn = QComboBox()
        self.YnormBtn.addItem('No normalization')
        self.YnormBtn.addItem('Global percentile')
        self.YnormBtn.addItem('Group percentile')
        self.YnormBtn.addItem('Folder percentile')
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

        self.aspectRatioBtn = QCheckBox('')
        self.aspectRatioBtn.setChecked(True)

        self.groupPlotBtn = QComboBox()
        for i in range(len(self.data_all)):
            self.groupPlotBtn.addItem('Group '+str(i+1))

        lay = QGridLayout(self)
        lay.addWidget(QLabel('Background subtraction:'),2,0,1,1)
        lay.addWidget(self.bckgBtn,2,1,1,1)
        lay.addWidget(QLabel('Y axis normalization:'),4,0,1,1)
        lay.addWidget(self.YnormBtn,4,1,1,1)
        lay.addWidget(QLabel('X axis normalization:'),5,0,1,1)
        lay.addWidget(self.XnormBtn,5,1,1,1)
        lay.addWidget(QLabel('A-P orientation correction:'),6,0,1,1)
        lay.addWidget(self.orientationBtn,6,1,1,1)
        lay.addWidget(QLabel('Alignment:'),7,0,1,1)
        lay.addWidget(self.alignmentBtn,7,1,1,1)
        lay.addWidget(QLabel('Set axes aspect ratio to equal:'),8,0,1,1)
        lay.addWidget(self.aspectRatioBtn,8,1,1,1)
        lay.addWidget(QLabel('Current group:'),9,0,1,1)
        lay.addWidget(self.groupPlotBtn,9,1,1,2)
        settings_group.setLayout(lay)

        #######################

        self.applyBtn = QPushButton('Apply Settings')
        self.applyBtn.clicked.connect(self.remakePlot)

        self.saveBtn = QPushButton('Save Tif image')
        self.saveBtn.clicked.connect(self.save_tif)

        lay = QGridLayout(self)
        lay.setSpacing(10)
        lay.addWidget(NavigationToolbar(self.canvas, self),0,0,1,2)
        lay.addWidget(self.canvas,1,0,1,2)
        lay.addWidget(settings_group,2,0,1,2)        
        lay.addWidget(self.applyBtn,3,0,1,2)
        lay.addWidget(self.saveBtn,4,0,1,2)

        self.remakePlot()

        self.setWindowTitle('Channel')
        QApplication.setStyle('Macintosh')

    def onCheckingXnormBtn(self):
        if self.XnormBtn.isChecked():
            self.alignmentBtn.setEnabled(False)
        else:
            self.alignmentBtn.setEnabled(True)

    def remakePlot(self):

        n_groups = len(self.data_all)
        n_folders = [len(self.data_all[group_idx]) for group_idx in range(n_groups)]
        n_gastr = [[len(self.data_all[group_idx][folder_idx]['input_file']) for folder_idx in range(n_folders[group_idx])] for group_idx in range(n_groups)]

        # rearrange dataset
        profiles_all = [[[0 for k in range(n_gastr[i][j])] for j in range(n_folders[i])] for i in range(n_groups)]
        for i in range(n_groups):
            for j in range(n_folders[i]):
                for k in range(n_gastr[i][j]):
                    profiles_all[i][j][k] = np.array(self.data_all[i][j][self.profileType][k][self.channel])
                    # subtract background or not
                    if self.bckgBtn.currentText() == 'Background':
                        profiles_all[i][j][k] -= self.data_all[i][j]['Background'][k][self.channel]
                    if self.bckgBtn.currentText() == 'Minimum':
                        profiles_all[i][j][k] -= np.min(profiles_all[i][j][k])

        # normalize fluorescence intensity accordingly
        percs = [None,None]
        if self.YnormBtn.currentText() == 'Global percentile':
            flat = []
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        for l in profiles_all[i][j][k]:
                            flat.append(l)
            percs = np.percentile(np.array(flat),(.3,99.7))
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        profile = np.array(profiles_all[i][j][k])
                        profiles_all[i][j][k] = np.clip((profile-percs[0])/(percs[1]-percs[0]),0,1.)
        elif self.YnormBtn.currentText() == 'Group percentile':
            flat = [[]for i in range(n_groups)]
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        for l in profiles_all[i][j][k]:
                            flat[i].append(l)
            percs = [np.percentile(np.array(f),(.3,99.7)) for f in flat]
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        # print(percs[i])
                        profile = np.array(profiles_all[i][j][k])
                        profiles_all[i][j][k] = np.clip((profile-percs[i][0])/(percs[i][1]-percs[i][0]),0,1.)
        elif self.YnormBtn.currentText() == 'Folder percentile':
            flat = [[[] for j in range(n_folders[i])] for i in range(n_groups)]
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        for l in profiles_all[i][j][k]:
                            flat[i][j].append(l)
            percs = [[np.percentile(np.array(f),(.3,99.7)) for f in ff] for ff in flat]
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        # print(percs[i][j])
                        profile = np.array(profiles_all[i][j][k])
                        profiles_all[i][j][k] = np.clip((profile-percs[i][j][0])/(percs[i][j][1]-percs[i][j][0]),0,1.)
        self.percs = percs
            
        # normalize AP axis if necessary
        if self.XnormBtn.isChecked():
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        profile = profiles_all[i][j][k]
                        x = np.linspace(0,1,len(profile))
                        fun = interp1d(x,profile)
                        new_x = np.linspace(0,1,101)
                        profiles_all[i][j][k] = fun(new_x)

        # compute length of longest gastruloid
        max_length = []
        for i in range(n_groups):
            for j in range(n_folders[i]):
                for k in range(n_gastr[i][j]):
                    max_length.append(len(profiles_all[i][j][k]))
        max_length = np.max(max_length)

        # orient plots according to setting
        if self.orientationBtn.currentText() == 'Signal based':
            for i in range(n_groups):
                for j in range(n_folders[i]):
                    for k in range(n_gastr[i][j]):
                        y = np.array(profiles_all[i][j][k])[~np.isnan(profiles_all[i][j][k])]
                        n_p = len(y)
                        if np.sum(y[:int(n_p/2)])>np.sum(y[int(n_p-n_p/2):]):
                            profiles_all[i][j][k] = profiles_all[i][j][k][::-1]

        # pad array to the right or left
        for i in range(n_groups):
            for j in range(n_folders[i]):
                for k in range(n_gastr[i][j]):
                    w = max_length-len(profiles_all[i][j][k])
                    if self.alignmentBtn.currentText() == 'Left':
                        pad_width = (0,w)
                    if self.alignmentBtn.currentText() == 'Right':
                        pad_width = (w,0)
                    elif self.alignmentBtn.currentText() == 'Center':
                        if 2*int(w/2)==w:
                            pad_width = (int(w/2),int(w/2))
                        else:
                            pad_width = (int(w/2)+1,int(w/2))
                    profiles_all[i][j][k] = np.pad(profiles_all[i][j][k],pad_width,mode='constant',constant_values=np.nan)

        ### make plot
        # lines = []
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(top=0.95,right=0.95,left=0.2,bottom=0.25)
        ax.set_xlabel(self.profileType)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(2,2))
        ax.set_ylabel('Time')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,2))
        # ax.axis('off') 
        
        # plot the selected group only
        i = self.groupPlotBtn.currentIndex()

        # compute and plot mean and std of the selected group
        # prepare blank image
        max_t = np.max([n_gastr[i][j] for j in range(n_folders[i])])
        max_l = np.max([len(profiles_all[i][j][k]) for j in range(n_folders[i]) for k in range(n_gastr[i][j])])

        data_mean = np.zeros((max_t,max_l))
        data_count = np.zeros((max_t,max_l))
        for j in range(n_folders[i]):
            for k in range(n_gastr[i][j]):
                data = np.nan_to_num(profiles_all[i][j][k])
                data_mean[k,:] += data 
                data_count[k,:] += data!=0 
                # plot the raw data if the button is checked
                # if self.rawBtn.isChecked():
                #     ax.plot(data_group[-1],'-', lw=.5, c=self.colors[i], alpha = 0.2)
        data_mean = data_mean.astype(np.float)/data_count.astype(np.float)
        data_mean = np.nan_to_num(data_mean)

        aspect = 'auto'
        if self.aspectRatioBtn.isChecked():
            aspect = 'equal'
            
        ax.imshow(data_mean, aspect=aspect)
        ax.set_title('Group '+str(i+1))
        self.tif_data = data_mean

        self.canvas.draw()
    
    def save_tif(self):
        name,_ = QFileDialog.getSaveFileName(self, 'Save Overview File')
        if name != '':
            ### check file extension: allow to save in other formats, but bias towards tif
            if os.path.splitext(name)[-1]!='.tif':
                buttonReply = QMessageBox.question(self,'File format warning!','File format not recommended. Do you want to save the image as tif?')
                if buttonReply == QMessageBox.Yes:
                    name = os.path.splitext(name)[0]+'.tif'
            
            # convert the image into int16 with the right brightness and contrast
            if  self.percs[0]!=None:
                self.tif_data = (2**16-1)*(self.tif_data-self.percs[0])/(self.percs[1]-self.percs[0])
            imsave(name+'', self.tif_data.astype(np.uint16))


