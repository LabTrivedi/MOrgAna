#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:57:50 2019

@author: ngritti
"""
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import (QApplication, QComboBox, QVBoxLayout, QDialog,
        QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
        QFileDialog, QMessageBox, QTabWidget, QWidget,
        QTableWidget, QTableWidgetItem, QSpinBox, QDoubleSpinBox,QCheckBox,
        QSplitter, QTreeView, QListView, QFileSystemModel, QAbstractItemView)
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import sys, warnings, os, time
from skimage.io import imread, imsave
import scipy.ndimage as ndi
from collections.abc import Iterable

from morgana.GUIs import manualmask
from morgana.GUIs import inspection
from morgana.GUIs import visualize0d
from morgana.GUIs import visualize1d
from morgana.GUIs import visualize2d
from morgana.MLModel import io as ioML
from morgana.MLModel import train
from morgana.MLModel import predict
from morgana.MLModel import overview as overviewML
from morgana.DatasetTools import io as ioDT
from morgana.DatasetTools.morphology import overview as overviewDT
from morgana.DatasetTools import arrangemorphodata
from morgana.DatasetTools import arrangefluodata
from morgana.ImageTools.objectsparsing import objectsparser
warnings.filterwarnings("ignore")


class morganaApp(QWidget):
    def __init__(self, parent=None):
        super(morganaApp, self).__init__(parent)

        self.modelFolder = '-'
        self.imageFolder = '-'
        self.imageImportFolder = '-'
        self.maskFolder = '-'
        self.classifier = None
        self.scaler = None
        self.params = { 'sigmas':       [1,2,5,15],
                        'down_shape':   0.25,
                        'edge_size':    2,
                        'fraction':     0.5,
                        'bias':         0.5,
                        'feature_mode': 'ilastik' }

        tabs = QTabWidget()
        self.maskTab = self.createMaskTab()
        tabs.addTab(self.maskTab,'Generate or Import Masks')

        self.quantificationTab = self.createQuantificationTab()
        tabs.addTab(self.quantificationTab,'Quantification')

        ### defined handler for subwindows
        self.inspector = None
        self.quantifier = []

        ####################################################################################################
        '''
        TESTS WITHOUT CLICKING
        '''

        ####################################################################################################

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(tabs)
        self.setLayout(self.layout)

        self.setWindowTitle('Organoids Segmentation App')
        QApplication.setStyle('Fusion')



    '''
    MASK TAB
    '''
    def createMaskTab(self):    
        mainTab = QWidget()
        self.createModelGroup()
        self.createImportGroup()
        
        self.isMask = QCheckBox("Import external masks")
        self.isMask.toggle()
        self.isMask.stateChanged.connect(self.changeMaskGroup)
        self.isMask.setChecked(False)
        
        mainTabLayout = QVBoxLayout()        
        mainTabLayout.addWidget(self.isMask)
        mainTabLayout.addWidget(self.modelGroup)
        mainTabLayout.addWidget(self.importGroup)
        mainTab.setLayout(mainTabLayout)
        return mainTab
        
    def changeMaskGroup(self):
        if self.isMask.isChecked():
            self.modelGroup.hide()
            self.importGroup.show()
        else:
            self.importGroup.hide()
            self.modelGroup.show()

        
    '''
    Generating model and generation of masks
    '''
    def createModelGroup(self):
        self.modelGroup = QGroupBox("")

        
        ########## create buttons for model definition group ##############
        self.modelDefGroup = QGroupBox("Machine Learning model definition")

        selectModel = QPushButton("Specify model folder")
        selectModel.setFocusPolicy(Qt.NoFocus)
        selectModel.clicked.connect( self.selectModelFolder )
        self.modelFolderSpace = QLineEdit(); self.modelFolderSpace.setText(self.modelFolder)
        self.modelFolderSpace.setReadOnly(True)
        self.modelFolderSpace.setStyleSheet('color:gray;')
        self.deepModel = QCheckBox("Use Multi Layer Perceptrons")
        self.deepModel.setChecked(False)

        self.showMoreButton = QPushButton("Show/Hide params")
        self.showMoreButton.setFocusPolicy(Qt.NoFocus)
        self.showMoreButton.clicked.connect(self.show_hide)

        self.sigmasLabel = QLabel('Sigmas:')
        self.sigmasSpace = QLineEdit(); self.sigmasSpace.setText("-")
        self.sigmasSpace.setEnabled(False)
        self.down_shapeLabel = QLabel('Downscaling:')
        self.down_shapeSpace = QDoubleSpinBox(); self.down_shapeSpace.setSpecialValueText("-")
        self.down_shapeSpace.setMinimum(-1); self.down_shapeSpace.setMaximum(1); self.down_shapeSpace.setSingleStep(.01);
        self.down_shapeSpace.setEnabled(False)
        self.edge_sizeLabel = QLabel('Edge size:')
        self.edge_sizeSpace = QSpinBox(); self.edge_sizeSpace.setSpecialValueText("-")
        self.edge_sizeSpace.setMinimum(0);
        self.edge_sizeSpace.setEnabled(False)
        self.fractionLabel = QLabel('Pixel% extraction:')
        self.fractionSpace = QDoubleSpinBox(); self.fractionSpace.setSpecialValueText("-")
        self.fractionSpace.setMinimum(0); self.fractionSpace.setMaximum(1); self.fractionSpace.setSingleStep(.1);
        self.fractionSpace.setEnabled(False)
        self.biasLabel = QLabel('Extraction bias:')
        self.biasSpace = QDoubleSpinBox(); self.biasSpace.setSpecialValueText("-")
        self.biasSpace.setMinimum(0); self.biasSpace.setMaximum(1); self.biasSpace.setSingleStep(.1);
        self.biasSpace.setEnabled(False)
        self.featuresLabel = QLabel('Features:')
        self.feature_modeSpace = QComboBox();
        self.feature_modeSpace.addItems(['-','daisy','ilastik']);
        self.feature_modeSpace.setCurrentIndex(0)
        self.feature_modeSpace.setEnabled(False)

        self.trainButton = QPushButton("Train model")
        self.trainButton.setEnabled(False)
        self.trainButton.setFocusPolicy(Qt.NoFocus)
        self.trainButton.clicked.connect(self.trainModel)

        ########## create buttons for model application group ##############
        self.predictionGroup = QGroupBox("Machine Learning model application")

        selectFolder = QPushButton("Specify image folder")
        selectFolder.setFocusPolicy(Qt.NoFocus)
        selectFolder.clicked.connect( self.selectImageFolder )
        self.imageFolderSpace = QLineEdit(); self.imageFolderSpace.setText(self.imageFolder)
        self.imageFolderSpace.setReadOnly(True)
        self.imageFolderSpace.setStyleSheet('color:gray;')

        self.predictButton = QPushButton("Generate masks")
        self.predictButton.setFocusPolicy(Qt.NoFocus)
        self.predictButton.clicked.connect(self.predict)
        self.predictButton.setEnabled(False)

        self.recapButton = QPushButton("Save overview image of masks")
        self.recapButton.setFocusPolicy(Qt.NoFocus)
        self.recapButton.clicked.connect(self.makeRecap)
        self.recapButton.setEnabled(False)

        self.inspectButton = QPushButton("Inspect masks")
        self.inspectButton.setFocusPolicy(Qt.NoFocus)
        self.inspectButton.clicked.connect(self.openInspectionWindow)
        self.inspectButton.setEnabled(False)

        ######### create layout for model definition group ########
        layout = QGridLayout()

        # layout.addWidget(self.welcomeText,      0,0,1,2)
        layout.addWidget(selectModel,               1,0,1,2)
        layout.addWidget(QLabel('Model folder:'),   2,0,1,1)
        layout.addWidget(self.modelFolderSpace,     2,1,1,1)
        layout.addWidget(self.deepModel,            3,0,1,1)

        layout.addWidget(self.showMoreButton,       4,0,1,1)
        layout.addWidget(self.trainButton,          4,1,1,1)
        layout.addWidget(self.sigmasLabel,          5,0,1,1)
        layout.addWidget(self.sigmasSpace,          5,1,1,1)
        layout.addWidget(self.down_shapeLabel,      6,0,1,1)
        layout.addWidget(self.down_shapeSpace,      6,1,1,1)
        layout.addWidget(self.edge_sizeLabel,       7,0,1,1)
        layout.addWidget(self.edge_sizeSpace,       7,1,1,1)
        layout.addWidget(self.fractionLabel,        8,0,1,1)
        layout.addWidget(self.fractionSpace,        8,1,1,1)
        layout.addWidget(self.biasLabel,            9,0,1,1)
        layout.addWidget(self.biasSpace,            9,1,1,1)
        layout.addWidget(self.featuresLabel,        10,0,1,1)
        layout.addWidget(self.feature_modeSpace,    10,1,1,1)

        self.modelDefGroup.setLayout(layout)

        ######### create layout for model application group ########
        layout = QGridLayout()

        layout.addWidget(selectFolder,              13,0,1,2)
        layout.addWidget(QLabel('Image folder:'),   14,0,1,1)
        layout.addWidget(self.imageFolderSpace,     14,1,1,1)
        layout.addWidget(self.predictButton,        15,0,1,2)
        layout.addWidget(self.recapButton,          16,0,1,2)
        layout.addWidget(self.inspectButton,        17,0,1,2)

        self.predictionGroup.setLayout(layout)

        ##################################################################
        layout = QVBoxLayout()

        layout.addWidget(self.modelDefGroup)
        layout.addWidget(self.predictionGroup)
        
        self.sigmasLabel.hide()
        self.sigmasSpace.hide()
        self.down_shapeLabel.hide()
        self.down_shapeSpace.hide()
        self.edge_sizeLabel.hide()
        self.edge_sizeSpace.hide()
        self.fractionLabel.hide()
        self.fractionSpace.hide()
        self.biasLabel.hide()
        self.biasSpace.hide()
        self.featuresLabel.hide()
        self.feature_modeSpace.hide()
        self.showMoreModel = False

        self.modelGroup.setLayout(layout)

    def show_hide(self):
        if self.showMoreModel:
            self.sigmasLabel.hide()
            self.sigmasSpace.hide()
            self.down_shapeLabel.hide()
            self.down_shapeSpace.hide()
            self.edge_sizeLabel.hide()
            self.edge_sizeSpace.hide()
            self.fractionLabel.hide()
            self.fractionSpace.hide()
            self.biasLabel.hide()
            self.biasSpace.hide()
            self.featuresLabel.hide()
            self.feature_modeSpace.hide()
            self.showMoreModel = False
        else:
            self.sigmasLabel.show()
            self.sigmasSpace.show()
            self.down_shapeLabel.show()
            self.down_shapeSpace.show()
            self.edge_sizeLabel.show()
            self.edge_sizeSpace.show()
            self.fractionLabel.show()
            self.fractionSpace.show()
            self.biasLabel.show()
            self.biasSpace.show()
            self.featuresLabel.show()
            self.feature_modeSpace.show()
            self.showMoreModel = True

    def selectModelFolder(self):
        self.modelFolder = QFileDialog.getExistingDirectory(self, "Select Input Folder of Model")

        # check if a trainingset is present
        # a trainingset needs to exist for every model, even if the model is already trained.
        trainingset_folder = os.path.join(self.modelFolder,'trainingset')
        if os.path.exists(trainingset_folder):
            flist_in = ioDT.get_image_list(trainingset_folder, string_filter='_GT', mode_filter='exclude')
            flist_in.sort()
            flist_gt = ioDT.get_image_list(trainingset_folder, string_filter='_GT', mode_filter='include')
            flist_gt.sort()

            if len(flist_in) == 0:
                QMessageBox.warning(self,'Warning, no trainingset!','Selected "'+self.modelFolder+'" but no trainingset *data* detected. Transfer some images in the "trainingset" folder.')
                self.modelFolder = '-'
                return
            if len(flist_in)!=len(flist_gt):
                QMessageBox.warning(self,'Warning, trainingset incomplete!','Selected "'+self.modelFolder+'" but not all masks have been created.\nPlease provide manually annotated masks.')
                for f in flist_in:
                    fn,ext = os.path.splitext(f)
                    mask_name = fn+'_GT'+ext
                    if not os.path.exists(mask_name):
                        m = manualmask.makeManualMask(f, subfolder='',fn=fn+'_GT'+ext)
                        # m.setModal(True)
                        m.show()
                        m.exec()
                # self.modelFolder = '-'
                # return
        else:
            QMessageBox.warning(self,'Warning, no trainingset!','Selected "'+self.modelFolder+'" but no "trainingset" folder detected.')
            self.modelFolder = '-'
            return
        # check if the model is already trained.
        # if not, only allow training button
        model_file = os.path.join(self.modelFolder,'scaler.pkl')
        if not os.path.exists(model_file):
            QMessageBox.warning(self,'Warning, train model!','Train the model before loading!\nSetting default parameters...')
        else:
            self.loadModel()
            if self.classifier is None:
                return
            self.predictButton.setEnabled(True)
            self.recapButton.setEnabled(True)
            self.inspectButton.setEnabled(True)
        
        self.modelFolderSpace.setText(self.modelFolder)
        self.set_params()
        self.sigmasSpace.setEnabled(True)
        self.down_shapeSpace.setEnabled(True)
        self.edge_sizeSpace.setEnabled(True)
        self.fractionSpace.setEnabled(True)
        self.biasSpace.setEnabled(True)
        self.feature_modeSpace.setEnabled(True)
        self.trainButton.setEnabled(True)

    def set_params(self):
        self.sigmasSpace.setText(str(self.params['sigmas']))
        self.down_shapeSpace.setValue(self.params['down_shape'])
        self.edge_sizeSpace.setValue(self.params['edge_size'])
        self.fractionSpace.setValue(self.params['fraction'])
        self.biasSpace.setValue(self.params['bias'])
        self.feature_modeSpace.setCurrentIndex(['-','daisy','ilastik'].index(self.params['feature_mode']))
        self.feature_modeSpace.model().item(0).setEnabled(False)

    def read_and_check_params(self):
        s_str = self.sigmasSpace.text().replace(' ','').replace('[','').replace(']','')
        if s_str[-1]==',': s_str = s_str[:-1]
        self.params['sigmas'] = []
        for x in s_str.split(','):
            try:
                self.params['sigmas'].append(float(x))
            except:
                self.params['sigmas'].append(x)
        self.params['down_shape'] = self.down_shapeSpace.value()
        self.params['edge_size'] = self.edge_sizeSpace.value()
        self.params['fraction'] = self.fractionSpace.value()
        self.params['bias'] = self.biasSpace.value()
        self.params['feature_mode'] = self.feature_modeSpace.currentText()
        if not all(isinstance(x, float) for x in self.params['sigmas']):
            QMessageBox.warning(self,'Warning, values of sigmas not valid!','It seems there is at least one sigma that is not a number:\n'+str(self.params['sigmas']))
            
    def trainModel(self, archBox):
        self.read_and_check_params()

        #############################################
        # load images to be used as training set
        #############################################
        training_folder = os.path.join(self.modelFolder,'trainingset')
        flist_in = ioDT.get_image_list(training_folder, string_filter='_GT', mode_filter='exclude')
        img_train = []
        for f in flist_in:
            img = imread(f)
            if len(img.shape) == 2:
                img = np.expand_dims(img,0)
            if img.shape[-1] == np.min(img.shape):
                img = np.moveaxis(img, -1, 0)
            img_train.append( img[0] )
        # img_train = np.array(img_train)

        flist_gt = ioDT.get_image_list(training_folder, string_filter='_GT', mode_filter='include')
        gt_train = [ imread(f) for f in flist_gt ]
        gt_train = [ g.astype(int) for g in gt_train ]

        print('##### Training set:')
        for i,f in enumerate(zip(flist_in,flist_gt)):
            print(i+1,'\t', os.path.split(f[0])[-1],'\t', os.path.split(f[1])[-1])

        #############################################
        # compute features and generate training set and weights
        #############################################

        print('##### Generating training set...')
        X, Y, w, self.scaler = train.generate_training_set( img_train, 
                                                [g.astype(np.uint8) for g in gt_train], 
                                                sigmas=self.params['sigmas'],
                                                down_shape=self.params['down_shape'],
                                                edge_size=self.params['edge_size'],
                                                fraction=self.params['fraction'],
                                                feature_mode=self.params['feature_mode'],
                                                bias=self.params['bias'] )

        #############################################
        # Train the model
        #############################################

        print('##### Training model...')
        start = time.time()
        self.classifier = train.train_classifier( X, Y, w, deep=self.deepModel.isChecked(), hidden=(350, 50) )
        print('Models trained in %.3f seconds.'%(time.time()-start))
        # print('classes_: ', self.classifier.classes_)
        # print('coef_: ', self.classifier.coef_)

        #############################################
        # Save the model
        #############################################

        ioML.save_model( self.modelFolder,
                                self.classifier,
                                self.scaler,
                                sigmas=self.params['sigmas'],
                                down_shape=self.params['down_shape'],
                                edge_size=self.params['edge_size'],
                                fraction=self.params['fraction'],
                                feature_mode=self.params['feature_mode'],
                                bias=self.params['bias'], deep=self.deepModel.isChecked() )
        print('##### Model saved!')
        self.predictButton.setEnabled(True)

    def loadModel(self):
        #############################################
        # load parameters and classifier
        #############################################
        print('##### Loading classifier model and parameters...')
        self.classifier, self.scaler, self.params = ioML.load_model( self.modelFolder, deep=self.deepModel.isChecked() )
        if self.classifier is None:
            QMessageBox.warning(self,'Warning!','Could not find any model')
        else:
            print('Success! Model loaded!')

    def selectImageFolder(self):
        self.imageFolder = QFileDialog.getExistingDirectory(self, "Select Input Folder of Model",
                "C:\\Users\\nicol\\Desktop\\dmso")
        if self.imageFolder == '':
            self.imageFolder = '-'
            return

        self.imageFolderSpace.setText(self.imageFolder)
        self.recapButton.setEnabled(True)
        self.inspectButton.setEnabled(True)
        self.maskFolderSpace.setText(self.imageFolder)
        # self.inspectButtonTL.setEnabled(True)

    def predict(self):
        #############################################
        # apply classifiers and save images
        #############################################

        result_folder = os.path.join(self.imageFolder,'result_segmentation')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        flist_in = ioDT.get_image_list(self.imageFolder)
        flist_in.sort()

        for f_in in flist_in:

            print('#'*20+'\nLoading',f_in,'...')
            img = imread(f_in)
            if len(img.shape) == 2:
                img = np.expand_dims(img,0)
            if img.shape[-1] == np.min(img.shape):
                img = np.moveaxis(img, -1, 0)
            img = img[0]

            print('Predicting image...')
            pred, prob = predict.predict_image( img,
                                self.classifier,
                                self.scaler,
                                sigmas=self.params['sigmas'],
                                new_shape_scale=self.params['down_shape'],
                                feature_mode=self.params['feature_mode'],
                                deep=self.deepModel.isChecked() )

            # remove objects at the border
            negative = ndi.binary_fill_holes(pred==0)
            mask_pred = (pred==1)*negative
            edge_prob = ((2**16-1)*prob[2]).astype(np.uint16)
            mask_pred = mask_pred.astype(np.uint8)

            # save mask
            parent, filename = os.path.split(f_in)
            filename, file_extension = os.path.splitext(filename)
            new_name = os.path.join(parent,'result_segmentation',filename+'_classifier'+file_extension)
            imsave(new_name, pred, check_contrast=False)

            # perform watershed
            mask_final = predict.make_watershed( mask_pred,
                                        edge_prob,
                                        new_shape_scale=self.params['down_shape'] )

            # save final mask
            parent, filename = os.path.split(f_in)
            filename, file_extension = os.path.splitext(filename)
            new_name = os.path.join(parent,'result_segmentation',filename+'_watershed'+file_extension)
            imsave(new_name, mask_final, check_contrast=False)

        print('All images done!')

    def makeRecap(self):
        name,_ = QFileDialog.getSaveFileName(self, 'Save Overview File')
        if name != '':
            overviewML.generate_overview(self.imageFolder, saveFig=True, fileName=name, downshape=5)

    def openInspectionWindow(self):
        self.inspector = inspection.inspectionWindow_20max(self.imageFolder, parent=None, start=0, stop=20)
        self.inspector.show()

    def selectMaskFolder(self):
        self.maskFolder = QFileDialog.getExistingDirectory(self, "Select Input Folder of Masks",
                "C:\\Users\\nicol\\Desktop\\dmso")
        if self.maskFolder == '':
            self.maskFolder = self.imageFolder
            return

        self.maskFolderSpace.setText(self.maskFolder)


    '''
    Import masks if user has already created them
    '''

    def createImportGroup(self):
        self.importGroup = QGroupBox("")
        
        ########## create buttons for import masks and images group ##############
        self.importGroup1 = QGroupBox("If masks are already present, import files.")


        # self.instruct2 = QLabel('If masks are already generated, \nselect image and mask folder here.') 
        
        selectFolder = QPushButton("Specify image folder")
        selectFolder.setFocusPolicy(Qt.NoFocus)
        selectFolder.clicked.connect( self.selectImportImageFolder )
        self.imageImportFolderSpace = QLineEdit()
        self.imageImportFolderSpace.setText(self.imageImportFolder)
        self.imageImportFolderSpace.setReadOnly(True)
        self.imageImportFolderSpace.setStyleSheet('color:gray;')

        selectMaskFolder = QPushButton("Specify mask folder")
        selectMaskFolder.setFocusPolicy(Qt.NoFocus)
        selectMaskFolder.clicked.connect( self.selectMaskFolder )
        self.maskFolderSpace = QLineEdit(); self.maskFolderSpace.setText(self.maskFolder)
        self.maskFolderSpace.setReadOnly(True)
        self.maskFolderSpace.setStyleSheet('color:gray;')
        
        self.maskLabel = QLabel('File identifier of masks:')
        self.maskSpace = QLineEdit(); self.maskSpace.setText("")

        self.isBorder = QCheckBox("Include objects at border of images")
        self.isBorder.setChecked(False)
        

        self.importGroup2 = QGroupBox("")

        self.importButton = QPushButton("Import Masks and Images")
        self.trainButton.setFocusPolicy(Qt.NoFocus)
        self.importButton.clicked.connect(self.importImageMask)

        layout = QGridLayout()
        # layout.addWidget(self.instruct2,            0,0,1,2)
        layout.addWidget(selectFolder,              1,0,1,2)
        layout.addWidget(QLabel('Image folder:'),   2,0,1,1)
        layout.addWidget(self.imageImportFolderSpace,2,1,1,1)

        layout.addWidget(selectMaskFolder,          3,0,1,2)
        layout.addWidget(QLabel('Masks folder:'),   4,0,1,1)
        layout.addWidget(self.maskFolderSpace,      4,1,1,1)
        
        layout.addWidget(self.maskLabel,            5,0,1,1)
        layout.addWidget(self.maskSpace,            5,1,1,1)
        
        layout.addWidget(self.isBorder,             6,0,1,2)
        self.importGroup1.setLayout(layout)

        layout = QGridLayout()
        layout.addWidget(self.importButton,         0,0,1,2)
        self.importGroup2.setLayout(layout)

        layout = QVBoxLayout()
        layout.addWidget(self.importGroup1)
        layout.addWidget(self.importGroup2)
        self.importGroup.setLayout(layout)


    def selectImportImageFolder(self):
        self.imageImportFolder = QFileDialog.getExistingDirectory(self, "Select Input Folder of Model",
                "C:\\Users\\nicol\\Desktop\\dmso")
        if self.imageImportFolder == '':
            self.imageImportFolder = '-'
            return

        self.imageImportFolderSpace.setText(self.imageImportFolder)
        self.maskFolderSpace.setText(self.imageImportFolder)

    def importImageMask(self):
        objectsparser.parsing_images(self.imageImportFolder, \
            self.maskFolder, self.maskSpace.text(), self.isBorder.isChecked())


    '''
    QUANTIFICATION TAB
    '''
    def createQuantificationTab(self):
        self.groups = []

        mainTab = QWidget()
        self.createGroup1()
        self.createGroup2()
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.group1)
        splitter.addWidget(self.group2)
        
        mainTabLayout = QVBoxLayout()        
        mainTabLayout.addWidget(splitter)
        mainTab.setLayout(mainTabLayout)        
        return mainTab

    def group_checked(self, state, group):
        chs = []
        for ch in group.findChildren(QLabel):
            chs.append(ch)
        for ch in group.findChildren(QSpinBox):
            chs.append(ch)
        for ch in group.findChildren(QComboBox):
            chs.append(ch)
        for ch in group.findChildren(QPushButton):
            chs.append(ch)
        for ch in group.findChildren(QCheckBox):
            chs.append(ch)   

        if not state:
            for ch in chs:
                ch.setVisible(False)
        else:
            for ch in chs:
                ch.setVisible(True)

    def createGroup1(self):
        self.group1 = QGroupBox("Groups")
        self.group1.setCheckable(True)
        self.group1.toggled.connect(lambda state, x=self.group1: self.group_checked(state, x))

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)

        self.tabs.tabCloseRequested.connect(self.removeGroup)

        self.AddTabButton = QPushButton("Add New Group")
        self.AddTabButton.clicked.connect(self.addGroup)
        self.addGroup()

        layout = QVBoxLayout()
        layout.addWidget(self.AddTabButton)
        layout.addWidget(self.tabs)
        self.group1.setLayout(layout)

    def addGroup(self):
        class FileDialog(QFileDialog):
            def __init__(self, *args):
                QFileDialog.__init__(self, *args)
                self.setOption(self.DontUseNativeDialog, True)
                self.setFileMode(self.DirectoryOnly)

                for view in self.findChildren((QListView, QTreeView)):
                    if isinstance(view.model(), QFileSystemModel):
                        view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        class MyTable(QTableWidget):
            def keyPressEvent(self, event):
                if event.key() == Qt.Key_Delete:
                    row = self.currentRow()
                    self.removeRow(row)
                else:
                    super().keyPressEvent(event)

        def addDataset():
            dialog = FileDialog()
            if dialog.exec_() == QDialog.Accepted:
                datasets = dialog.selectedFiles()
            else:
                return
                # print(dialog.selectedFiles())

            # dataset = QFileDialog.getExistingDirectory(self, "Select dataset")
            for dataset in datasets:
                if dataset!='':
                    table = self.tabs.widget(self.tabs.currentIndex()).children()[1]
                    rowPosition = table.rowCount()
                    table.insertRow(rowPosition)
                    table.setItem(rowPosition,0,QTableWidgetItem(dataset))

        newTab = QWidget()

        table = MyTable()
        table.insertColumn(0)
        selectFolder = QPushButton("Select new dataset")
        selectFolder.clicked.connect(addDataset)

        tablayout = QGridLayout()
        tablayout.addWidget(table,0,0,1,2)
        tablayout.addWidget(selectFolder,1,0,1,2)
        newTab.setLayout(tablayout)

        # n = self.tabs.tabText(self.tabs.count()-1)
        # 
        self.tabs.addTab(newTab, 'Group '+str(self.tabs.count()+1))

        # print(self.tabs.widget(self.tabs.count()-1).children())

        # return tab
    
    def removeGroup(self,index):
        self.tabs.removeTab(index)

    def selectAllButtonClicked(self):
        if self.selectAllButton.isChecked():
            self.morphoType.setEnabled(False)
            # self.maskType.setEnabled(False)
        else:
            self.morphoType.setEnabled(True)
            # self.maskType.setEnabled(True)

    def createGroup2(self):
        self.group2 = QGroupBox("")

        self.isTimelapse = QCheckBox("Timelapse data")
        self.isTimelapse.setChecked(False)

        def buildGroupVis():
            group = QGroupBox("Visualization functions")
            group.setCheckable(True)
            group.toggled.connect(lambda state, x=group: self.group_checked(state, x))
            group.setChecked(False)

            compositeButton = QPushButton("Create overview composite")
            compositeButton.clicked.connect(self.createCompositeOverviewAll)

            meshgridButton = QPushButton("Create meshgrid overview")
            meshgridButton.clicked.connect(self.createMeshgridOverviewAll)

            layout = QVBoxLayout()
            layout.addWidget(compositeButton)
            layout.addWidget(meshgridButton)
            group.setLayout(layout)
            self.group_checked(False, group)

            return group

        def buildGroupMorpho():
            group = QGroupBox("Morphology quantification")
            group.setCheckable(True)
            group.toggled.connect(lambda state, x=group: self.group_checked(state, x))
            group.setChecked(False)

            self.maskType = QComboBox()
            self.maskType.addItem("Unprocessed")
            self.maskType.addItem("Straightened")

            self.morphoKeys = [
                        'area',
                        'eccentricity',
                        'major_axis_length',
                        'minor_axis_length',
                        'equivalent_diameter',
                        'perimeter',
                        'euler_number',
                        'extent',
                        'form_factor',
                        # 'inertia_tensor',
                        # 'inertia_tensor_eigvals',
                        # 'moments',
                        # 'moments_central',
                        # 'moments_hu',
                        # 'moments_normalized',
                        'orientation',
                        'locoefa_coeff'
                        ]
            self.datamorphotype = [
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                # 0,
                                # 0,
                                # 0,
                                # 0,
                                # 0,
                                # 0,
                                0,
                                1,
                                ]
            self.morphoType = QComboBox()
            for key in self.morphoKeys:
                self.morphoType.addItem(key)

            self.selectAllButton = QCheckBox("Use all parameters")
            self.selectAllButton.clicked.connect(self.selectAllButtonClicked)

            morphologyButton = QPushButton("Visualize Morphological Parameter(s)")
            morphologyButton.clicked.connect(self.createMorphologyPlot)

            layout = QGridLayout()
            layout.addWidget(QLabel("Type of mask:"),           1,0,1,1)
            layout.addWidget(self.maskType,                     1,1,1,1)
            layout.addWidget(QLabel("Morphological parameter"), 2,0,1,1)
            layout.addWidget(self.morphoType,                   2,1,1,1)
            layout.addWidget(self.selectAllButton,              3,0,1,2)
            layout.addWidget(morphologyButton,                  4,0,1,2)
            group.setLayout(layout)
            self.group_checked(False, group)

            return group

        def buildGroupFluo():
            group = QGroupBox("Fluorescence quantification")
            group.setCheckable(True)
            group.toggled.connect(lambda state, x=group: self.group_checked(state, x))
            group.setChecked(False)

            self.fluorescenceChannel = QSpinBox()
            self.fluorescenceChannel.setRange(0,100)
            self.fluorescenceChannel.setAlignment(Qt.AlignRight)

            self.spatialType = QComboBox()
            self.spatialType.addItem('Average')
            self.spatialType.addItem('Antero-Posterior profile')
            self.spatialType.addItem('Left-Right profile')
            self.spatialType.addItem('Radial profile')
            self.spatialType.addItem('Angular profile')

            computeButton = QPushButton("Compute graph")
            computeButton.clicked.connect(self.createFluoGraph)

            layout = QGridLayout()
            layout.addWidget(QLabel("Fluorescence channel:"),   0,0,1,1)
            layout.addWidget(self.fluorescenceChannel,          0,1,1,1)
            layout.addWidget(QLabel("Spatial profile type:"),   2,0,1,1)
            layout.addWidget(self.spatialType,                  2,1,1,1)
            layout.addWidget(computeButton,                     3,0,1,2)
            group.setLayout(layout)
            self.group_checked(False, group)

            return group

        # def buildGroupSpots():
        #     group = QGroupBox("Spots quantification")
        #     group.setCheckable(True)
        #     group.toggled.connect(lambda state, x=group: self.group_checked(state, x))
        #     group.setChecked(False)

        #     self.spotsFluorescenceChannel = QSpinBox()
        #     self.spotsFluorescenceChannel.setRange(0,100)
        #     self.spotsFluorescenceChannel.setAlignment(Qt.AlignRight)

        #     self.spotsSpatialType = QComboBox()
        #     self.spotsSpatialType.addItem('Average')
        #     self.spotsSpatialType.addItem('Antero-Posterior profile')
        #     self.spotsSpatialType.addItem('Left-Right profile')
        #     self.spotsSpatialType.addItem('Radial profile')
        #     self.spotsSpatialType.addItem('Angular profile')

        #     self.spotsCountRadio = QPushButton("Spot count")
        #     self.spotsCountRadio.clicked.connect(self.makeSpotCountPlot)

        #     # # self.spotsPositionRadio = QCheckBox("Position")
        #     # self.spotsAreaRadio = QCheckBox("Area")
        #     # self.spotaPerimeterRadio = QCheckBox("Perimeter")
        #     # self.spotsMajorAxisRadio = QCheckBox('Major axis')
        #     # self.spotsMinorAxisRadio = QCheckBox('Minor Axis')
        #     # self.spotsEccetricityRadio = QCheckBox('Eccentricity')
        #     # self.spotsEftRadio = QCheckBox('Elliptical Fourier Transform')
        #     # self.spotsOrientationRadio = QCheckBox('Orientation')
        #     # self.spotsFluoRadio = QCheckBox('Fluorescence intensity')

        #     # spotsButton = QPushButton("Compute graph")
        #     # spotsButton.clicked.connect(self.createSpotsGraphAll)

        #     layout = QGridLayout()
        #     layout.addWidget(QLabel('Fluorescence channel:'),   0,0,1,1)
        #     layout.addWidget(self.spotsFluorescenceChannel,     0,1,1,1)
        #     layout.addWidget(QLabel('Spatial profile type:'),   1,0,1,1)
        #     layout.addWidget(self.spotsSpatialType,             1,1,1,1)
        #     layout.addWidget(self.spotsCountRadio,              2,0,1,2)
        #     # layout.addWidget(self.spotsAreaRadio,       3,0,1,1)
        #     # layout.addWidget(self.spotaPerimeterRadio,  3,1,1,1)
        #     # layout.addWidget(self.spotsMajorAxisRadio,  4,0,1,1)
        #     # layout.addWidget(self.spotsMinorAxisRadio,  4,1,1,1)
        #     # layout.addWidget(self.spotsEftRadio,        5,0,1,1)
        #     # layout.addWidget(self.spotsOrientationRadio,5,1,1,1)
        #     # layout.addWidget(self.spotsFluoRadio,       6,0,1,1)
        #     # layout.addWidget(spotsButton,               7,0,1,2)
        #     group.setLayout(layout)
        #     self.group_checked(False, group)

        #     return group

        groupVis = buildGroupVis()
        groupMorpho = buildGroupMorpho()
        groupFluo = buildGroupFluo()
        # groupSpots = buildGroupSpots()
        
        layout = QGridLayout()
        layout.addWidget(self.isTimelapse,      0,0,1,1)
        layout.addWidget(groupVis,              2,0,1,2)
        layout.addWidget(groupMorpho,           3,0,1,2)
        layout.addWidget(groupFluo,             4,0,1,2)
        # layout.addWidget(groupSpots,            5,0,1,2)
        self.group2.setLayout(layout)

    def createCompositeOverviewAll(self):

        # for every group
        folders = []
        for i in range(self.tabs.count()):
            # extract table in the group
            children = self.tabs.widget(i).children()
            table = children[1]
            # extract folders (dataset) in the table
            for j in range(table.rowCount()):
                folder = table.item(j,0).text()
                folders.append(folder)
                overviewDT.createCompositeOverview(folder)
            # print(folders)

        file = '_composite_recap.tif/.png'
        text = 'Composite files saved at:'
        for f in folders:
            parent,cond = os.path.split(f)
            text = text + '\n\t'+os.path.join(os.path.split(parent)[-1],'result_segmentation', cond + file)
        QMessageBox.information(self,"Completed successfully",text)

    def createMeshgridOverviewAll(self):

        # for every group
        for i in range(self.tabs.count()):
            # extract table in the group
            children = self.tabs.widget(i).children()
            table = children[1]
            # extract folders (dataset) in the table
            folders = []
            for j in range(table.rowCount()):
                folder = table.item(j,0).text()
                folders.append(folder)
                overviewDT.createMeshgridOverview(folder)
            # print(folders)

        file = '_meshgrid_recap.png'
        text = 'Meshgrid files saved at:'
        for f in folders:
            parent,cond = os.path.split(f)
            text = text + '\n\t'+os.path.join(os.path.split(parent)[-1],'result_segmentation', cond + file)
        QMessageBox.information(self,"Completed successfully",text)

    def createMorphologyPlot(self):

        computeMorpho = [False for key in self.morphoKeys]
        computeMorpho[self.morphoType.currentIndex()] = True
        if self.selectAllButton.isChecked():
            computeMorpho = [True for key in self.morphoKeys]

        # extract all folders to compute
        folders = [[] for i in range(self.tabs.count())]
        for i in range(self.tabs.count()):
            children = self.tabs.widget(i).children()
            table = children[1]
            for j in range(table.rowCount()):
                folders[i].append( table.item(j,0).text() )

        # extract data from all the folders
        data_all, keys = arrangemorphodata.collect_morpho_data( 
                                                                    folders, 
                                                                    self.morphoKeys, 
                                                                    computeMorpho, 
                                                                    self.maskType.currentText(), 
                                                                    self.isTimelapse.isChecked()
                                                                      )

        # for every quantification parameter, make the appropriate plot
        for key in keys:
            data_key = [data[key] for data in data_all]
            # print(data_key)

            # find out number of dimensions of the data_key object by going deeper in the object
            # and checking if the first item of layer n is iterable
            iterable = True
            ndim = 0
            first_object = data_key[0][0]
            while iterable:
                iterable = isinstance(first_object, Iterable)
                if iterable:
                    ndim += 1
                    first_object = first_object[0]
    
            # call the right visualization tool according to the number of dimensions
            ### clean up quantifier handler:
            self.quantifier = [self.quantifier[i] for i in range(len(self.quantifier)) if self.quantifier[i] is not None]

            if ndim == 0:
                self.quantifier.append( visualize0d.visualization_0d( data_key, key ) )
                self.quantifier[-1].show()
            elif ndim == 1:
                self.quantifier.append( visualize1d.visualization_1d( data_key, key ) )
                self.quantifier[-1].show()
            elif ndim == 2:
                self.quantifier.append( visualize2d.visualization_2d( data_key, key ) )
                self.quantifier[-1].show()
            
    def createFluoGraph(self):
        # print('createFluoGraph')
        # return

        # extract all folders to compute
        folders = [[] for i in range(self.tabs.count())]
        for i in range(self.tabs.count()):
            children = self.tabs.widget(i).children()
            table = children[1]
            for j in range(table.rowCount()):
                folders[i].append( table.item(j,0).text() )
        
        channel = self.fluorescenceChannel.value()
        distributionType = ['Average','APprofile','LRprofile','RADprofile','ANGprofile'][self.spatialType.currentIndex()]

        # extract data from all the folders
        data_all = arrangefluodata.collect_fluo_data( 
                                                folders, 
                                                channel, 
                                                distributionType, 
                                                self.isTimelapse.isChecked()
                                                  )

        # if the result is None, something went wrong!
        if not data_all:
            QMessageBox.warning(self,'Warning, invalid channel!','The channel selected doesn\'t appear in the raw data!')
            return

        # print(data_all)
        # make the appropriate plot
        data_key = [data['ch%d_%s'%(channel,distributionType)] for data in data_all]
        data_bckg = [data['ch%d_Background'%(channel)] for data in data_all]

        # find out number of dimensions of the data_key object by going deeper in the object
        # and checking if the first item of layer n is iterable
        iterable = True
        ndim = 0
        first_object = data_key[0][0]
        while iterable:
            iterable = isinstance(first_object, Iterable)
            if iterable:
                ndim += 1
                first_object = first_object[0]

        # call the right visualization tool according to the number of dimensions
        ### clean up quantifier handler:
        self.quantifier = [self.quantifier[i] for i in range(len(self.quantifier)) if self.quantifier[i] is not None]

        if ndim == 0:
            self.quantifier.append( visualize0d.visualization_0d( data_key, distributionType, background=data_bckg ) )
            self.quantifier[-1].show()
        elif ndim == 1:
            self.quantifier.append( visualize1d.visualization_1d( data_key, distributionType, background=data_bckg ) )
            self.quantifier[-1].show()
        elif ndim == 2:
            self.quantifier.append( visualize2d.visualization_2d( data_key, distributionType, background=data_bckg ) )
            self.quantifier[-1].show()

    def makeSpotCountPlot(self):
        # print('createFluoGraph')
        # return

        # extract all folders to compute
        folders = [[] for i in range(self.tabs.count())]
        for i in range(self.tabs.count()):
            children = self.tabs.widget(i).children()
            table = children[1]
            for j in range(table.rowCount()):
                folders[i].append( table.item(j,0).text() )

        # if self.spotsSpatialType.currentText()=='Average':
        #     data_all = utils_quantify.collect_spots_data_from_folders(folders,spatialDistNeeded='count')
        #     if not data_all:
        #         return
        #     utils_quantify.computeAndPlotMorphoAll(data_all,['count'],[True],
        #                         int(self.spotsFluorescenceChannel.value()),
        #                         self.isTimelapse.isChecked(),
        #                         style=self.plotType.currentText())

        # else:
        #     ### plot the AP profile of the fluorescence in the mask
        #     if self.spotsSpatialType.currentText()=='Antero-Posterior profile':
        #         key1, key2 = 'APposition', 'APprofile'
        #     ### plot the LR profile of the fluorescence in the mask
        #     if self.spotsSpatialType.currentText()=='Left-Right profile':
        #         key1, key2 = 'LRposition', 'LRprofile'
        #     ### plot the radial profile of the fluorescence in the mask
        #     if self.spotsSpatialType.currentText()=='Radial profile':
        #         key1, key2 = 'RADposition', 'RADprofile'
        #     ### plot the radial profile of the fluorescence in the mask
        #     if self.spotsSpatialType.currentText()=='Angular profile':
        #         key1, key2 = 'ANGposition', 'ANGprofile'

        #     data_all = utils_quantify.collect_spots_data_from_folders(folders,spatialDistNeeded=key1)
        #     if not data_all:
        #         return
        #     data_all = multi_objects_functions.convert_to_distribution(data_all,'count')
        #     utils_quantify.computeProfileAll( data_all,
        #                             channel = int(self.spotsFluorescenceChannel.value()),
        #                             isTimelapse = self.isTimelapse.isChecked(),
        #                             profileType = key2,
        #                             ylabel='Cell count' )

    def createSpotsGraphAll(self):
        print('createSpotsGraphAll')
        return
    
        # params = ['area','perimeter',
        #           'major_axis_length','minor_axis_length','eccentricity',
        #           'elliptical_fourier_transform','orientation','mean_intensity']

        # toplot = [False for i in params]
        # if self.spotsAreaRadio.isChecked():         toplot[0]=True
        # if self.spotaPerimeterRadio.isChecked():    toplot[1]=True
        # if self.spotsMajorAxisRadio.isChecked():    toplot[2]=True
        # if self.spotsMinorAxisRadio.isChecked():    toplot[3]=True
        # if self.spotsEccetricityRadio.isChecked():  toplot[4]=True
        # if self.spotsEftRadio.isChecked():          toplot[5]=True
        # if self.spotsOrientationRadio.isChecked():  toplot[6]=True
        # if self.spotsFluoRadio.isChecked():         toplot[7]=True

        # # extract all folders to compute
        # folders = [[] for i in range(self.tabs.count())]
        # for i in range(self.tabs.count()):

        #     children = self.tabs.widget(i).children()
        #     table = children[1]
        #     for j in range(table.rowCount()):
        #         folders[i].append( table.item(j,0).text() )

        # if self.spotsSpatialType.currentText()=='Average':
        #     print('To be implemented!')
        #     # data_all = utils_quantify.collect_spots_data_from_folders(folders,spatialDistNeeded='Average')
        #     # success = utils_quantify.computeAndPlotMorphoAll(data_all,params,t,int(self.fluorescenceChannel.value()),self.isTimelapse.isChecked())

        # if self.spotsSpatialType.currentText()=='Antero-Posterior profile':
        #     print("To be implemented!")
        #     # self.createAPprofileAll_spots(folders)
        #     # data_all = utils_quantify.collect_fluo_data_from_folders(folders,spatialDistNeeded='APprofile')
        #     # success = utils_quantify.computeProfileAll(data_all,int(self.fluorescenceChannel.value()),self.isTimelapse.isChecked())

        # if self.spotsSpatialType.currentText()=='Radial profile':
        #     print("To be implemented!")
        #     # data_all = utils_quantify.collect_fluo_data_from_folders(folders,spatialDistNeeded='RadialProfile')
        #     # success = utils_quantify.computeProfileAll(data_all,int(self.fluorescenceChannel.value()),self.isTimelapse.isChecked())

'''
run the main gui from the current file
'''
if __name__ == '__main__':
    def run():
        app = QApplication(sys.argv)
        gallery = morganaApp()
        gallery.show()
        sys.exit(app.exec_())

    run()
