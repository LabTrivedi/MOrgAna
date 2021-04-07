#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:57:50 2019

@author: ngritti
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QVBoxLayout, QDialog,
        QGridLayout, QGroupBox, QLabel, QPushButton,
        QRadioButton, QMessageBox, QWidget,
        QSpinBox, QDoubleSpinBox)
import numpy as np
import warnings, os, tqdm
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

from morgana import GUIs
from morgana import ImageTools
from morgana.DatasetTools import io
from morgana.DatasetTools.segmentation import io as ioSeg
from morgana.ImageTools.segmentation import segment
from morgana.DatasetTools.morphology import overview
from morgana import MLModel

'''
utils_postprocessing
'''
warnings.filterwarnings("ignore")

class inspectionWindow_20max(QDialog):
    def __init__(self, imageFolder, parent=None, start=None, stop=None):
        super(inspectionWindow_20max, self).__init__(parent)
        self.imageFolder = imageFolder
        self.folder, self.cond = os.path.split(self.imageFolder)
        self.flist_in = io.get_image_list(self.imageFolder)
        self.n_imgs = len( self.flist_in )
        self.start = start; self.stop = stop
        self.n_shown_max = self.stop - self.start
        self.make()

    def make(self):
        if self.start==None: self.start=0
        if self.stop==None: len( self.flist_in )
        self.stop = np.clip(self.stop,0,self.n_imgs)
        self.n_shown = self.stop-self.start
        self.showMore = False

        self.overview = MLModel.overview.generate_overview(self.imageFolder, saveFig=False, start = self.start, stop = self.stop, downshape=5)
        self.overview.show()

        if os.path.exists(os.path.join(self.imageFolder,'result_segmentation','segmentation_params.csv')):
            self.flist_in, self.chosen_masks, self.down_shapes, self.thinnings, self.smoothings = ioSeg.load_segmentation_params( os.path.join(self.imageFolder,'result_segmentation') )
            self.flist_in = [os.path.join(self.imageFolder,i) for i in self.flist_in]
        else:
            self.chosen_masks = ['c' for i in range(self.n_imgs)]
            self.down_shapes = [0.50 for i in range(self.n_imgs)]
            self.thinnings = [10 for i in range(self.n_imgs)]
            self.smoothings = [25 for i in range(self.n_imgs)]
        # if os.path.exists(os.path.join(self.imageFolder,'result_segmentation',self.cond+'_morpho_params.pkl')):
        #     utils_postprocessing.generate_final_recap(self.imageFolder, 
        #                                         chosen=[c!='' for c in self.chosen_masks], 
        #                                         saveFig=False)
        
        save_folder = os.path.join(self.imageFolder, 'result_segmentation')
        ioSeg.save_segmentation_params(  save_folder, 
                                                        [os.path.split(fin)[-1] for fin in self.flist_in],
                                                        self.chosen_masks,
                                                        self.down_shapes, 
                                                        self.thinnings, 
                                                        self.smoothings )

        mainTab = QWidget()
        self.createGroup1()
        self.createGroup2()
        mainTabLayout = QVBoxLayout()        
        mainTabLayout.addWidget(self.group2)
        mainTabLayout.addWidget(self.group1)
        mainTab.setLayout(mainTabLayout)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.group2)
        self.layout.addWidget(self.group1)
        self.setLayout(self.layout)

        self.setWindowTitle('Organoids Segmentation App')
        QApplication.setStyle('Fusion')

    def createGroup1(self):
        self.group1 = QGroupBox("")

        self.down_scaleLabel = QLabel('Downsampling')
        self.thinningLabel = QLabel('Thinning param')
        self.smoothingLabel = QLabel('Smoothing param')
        
        self.imageName = []
        self.maskTypeSpaces = []
        self.down_scaleSpaces = []
        self.thinningSpaces = []
        self.smoothingSpaces = []

        for i in range(self.n_shown):
            name = QLabel(os.path.splitext(os.path.split(self.flist_in[self.start+i])[-1])[0])
            self.imageName.append(name)
            
            m = QComboBox()
            m.addItems(['ignore','classifier','watershed','manual'])
            m.setCurrentIndex(['i','c','w','m'].index(self.chosen_masks[self.start+i]))
            self.maskTypeSpaces.append(m)

            m = QDoubleSpinBox()
            m.setMinimum(0)
            m.setMaximum(1000)
            m.setValue(self.down_shapes[self.start+i])
            self.down_scaleSpaces.append(m)

            m = QSpinBox()
            m.setMinimum(0)
            m.setMaximum(100)
            m.setValue(self.thinnings[self.start+i])
            self.thinningSpaces.append(m)

            m = QSpinBox()
            m.setMinimum(0)
            m.setMaximum(100)
            m.setValue(self.smoothings[self.start+i])
            self.smoothingSpaces.append(m)

        self.computeMaskForAllButton = QPushButton("Compute all masks")
        self.computeMaskForAllButton.setFocusPolicy(Qt.NoFocus)
        self.computeMaskForAllButton.clicked.connect(self.computeMaskForAll)

        self.showMoreButton = QPushButton("Show/Hide more parameters")
        self.showMoreButton.setFocusPolicy(Qt.NoFocus)
        self.showMoreButton.clicked.connect(self.show_hide)

        self.moveToNextButton = QPushButton("Next "+str(self.n_shown)+" images")
        self.moveToNextButton.setFocusPolicy(Qt.NoFocus)
        self.moveToNextButton.clicked.connect(self.moveToNext)

        self.moveToPreviousButton = QPushButton("Previous "+str(self.n_shown)+" images")
        self.moveToPreviousButton.setFocusPolicy(Qt.NoFocus)
        self.moveToPreviousButton.clicked.connect(self.moveToPrevious)


        layout = QGridLayout()
        layout.addWidget(self.moveToPreviousButton,0,0,1,1)
        layout.addWidget(self.moveToNextButton,0,1,1,1)
        layout.addWidget(QLabel('Input file'),1,0,1,2)
        layout.addWidget(QLabel('Mask type'),1,2,1,1)
        layout.addWidget(self.down_scaleLabel,1,3,1,1)
        layout.addWidget(self.thinningLabel,1,4,1,1)
        layout.addWidget(self.smoothingLabel,1,5,1,1)
        for i in range(self.n_shown):
            layout.addWidget(self.imageName[i],i+2,0,1,2)
            layout.addWidget(self.maskTypeSpaces[i],i+2,2,1,1)
            layout.addWidget(self.down_scaleSpaces[i],i+2,3,1,1)
            layout.addWidget(self.thinningSpaces[i],i+2,4,1,1)
            layout.addWidget(self.smoothingSpaces[i],i+2,5,1,1)
            self.down_scaleSpaces[i].hide()
            self.thinningSpaces[i].hide()
            self.smoothingSpaces[i].hide()
        self.down_scaleLabel.hide()
        self.thinningLabel.hide()
        self.smoothingLabel.hide()
        layout.addWidget(self.showMoreButton,i+3,0,1,3)

        self.group1.setLayout(layout)

    def createGroup2(self):
        self.group2 = QGroupBox("")

        self.computeMaskForAllButton = QPushButton("Compute all masks")
        self.computeMaskForAllButton.setFocusPolicy(Qt.NoFocus)
        self.computeMaskForAllButton.clicked.connect(self.computeMaskForAll)

        self.compute_meshgrid = QRadioButton("Compute full meshgrid (slow and high disk space usage!)")
        self.compute_meshgrid.setChecked(False)

        self.setParamsForAllButton = QPushButton("Set params for all")
        self.setParamsForAllButton.setFocusPolicy(Qt.NoFocus)
        self.setParamsForAllButton.clicked.connect(self.setForAll)

        self.masksAll = QComboBox()
        self.masksAll.addItems(['ignore','classifier','watershed','manual'])
        self.masksAll.setCurrentIndex(1)

        self.downScaleAll = QDoubleSpinBox()
        self.downScaleAll.setMinimum(0)
        self.downScaleAll.setMaximum(1000)
        self.downScaleAll.setValue(0.5)

        self.thinningAll = QSpinBox()
        self.thinningAll.setMinimum(0)
        self.thinningAll.setMaximum(100)
        self.thinningAll.setValue(10)

        self.smoothingAll = QSpinBox()
        self.smoothingAll.setMinimum(0)
        self.smoothingAll.setMaximum(100)
        self.smoothingAll.setValue(25)

        layout = QGridLayout()
        layout.addWidget(self.computeMaskForAllButton,0,0,1,4)
        layout.addWidget(self.compute_meshgrid,1,0,1,4)
        layout.addWidget(QLabel('Mask type:'),2,0,1,1)
        layout.addWidget(QLabel('Downscale:'),2,1,1,1)
        layout.addWidget(QLabel('Thinning:'),2,2,1,1)
        layout.addWidget(QLabel('Smoothing:'),2,3,1,1)
        layout.addWidget(self.masksAll,3,0,1,1)
        layout.addWidget(self.downScaleAll,3,1,1,1)
        layout.addWidget(self.thinningAll,3,2,1,1)
        layout.addWidget(self.smoothingAll,3,3,1,1)
        
        layout.addWidget(self.setParamsForAllButton,4,0,1,4)

        self.group2.setLayout(layout)
    
    def setForAll(self):
        # print(self.masksAll.currentText())
        # print(self.downScaleAll.value())
        # print(self.thinningAll.value())
        # print(self.smoothingAll.value())

        txt = self.masksAll.currentText()
        idx = ['ignore','classifier','watershed','manual'].index(txt)

        self.chosen_masks = [['i','c','w','m'][idx] for i in range(self.n_imgs)]
        self.down_shapes = [self.downScaleAll.value() for i in range(self.n_imgs)]
        self.thinnings = [self.thinningAll.value() for i in range(self.n_imgs)]
        self.smoothings = [self.smoothingAll.value() for i in range(self.n_imgs)]
        print(self.thinnings)
        # if os.path.exists(os.path.join(self.imageFolder,'result_segmentation',self.cond+'_morpho_params.pkl')):
        #     utils_postprocessing.generate_final_recap(self.imageFolder, 
        #                                         chosen=[c!='' for c in self.chosen_masks], 
        #                                         saveFig=False)
        
        save_folder = os.path.join(self.imageFolder, 'result_segmentation')
        ioSeg.save_segmentation_params(  save_folder, 
                                                        [os.path.split(fin)[-1] for fin in self.flist_in],
                                                        self.chosen_masks,
                                                        self.down_shapes, 
                                                        self.thinnings, 
                                                        self.smoothings )
        self.remake()

    def show_hide(self):
        if self.showMore:
            self.down_scaleLabel.hide()
            self.thinningLabel.hide()
            self.smoothingLabel.hide()
            for i in range(self.n_shown):
                self.down_scaleSpaces[i].hide()
                self.thinningSpaces[i].hide()
                self.smoothingSpaces[i].hide()
            self.showMore = False
        else:
            self.down_scaleLabel.show()
            self.thinningLabel.show()
            self.smoothingLabel.show()
            for i in range(self.n_shown):
                self.down_scaleSpaces[i].show()
                self.thinningSpaces[i].show()
                self.smoothingSpaces[i].show()
            self.showMore = True

    def read_segmentation_params(self):
        for i in range(self.n_shown):
            txt = self.maskTypeSpaces[i].currentText()
            idx = ['ignore','classifier','watershed','manual'].index(txt)
            self.chosen_masks[self.start+i] = ['i','c','w','m'][idx]

            self.down_shapes[self.start+i] = self.down_scaleSpaces[i].value()
            self.thinnings[self.start+i] = self.thinningSpaces[i].value()
            self.smoothings[self.start+i] = self.smoothingSpaces[i].value()

    def computeMaskForAll(self):
        self.read_segmentation_params()
        save_folder = os.path.join(self.imageFolder, 'result_segmentation')
        folder, cond = os.path.split(self.imageFolder)

        #############################################
        # clean masks previously generated
        #############################################

        flist_to_remove = io.get_image_list(save_folder, '_finalMask', 'include')
        for f in flist_to_remove:
            os.remove(f)
        segm_params = os.path.join(save_folder, 'segmentation_params.csv')
        if os.path.exists(segm_params):
            os.remove(segm_params)
        morpho_file = os.path.join(save_folder,cond+'_morpho_params.json')
        if os.path.exists(morpho_file):
            os.remove(morpho_file)

        #############################################
        # save parameters used to make segmentation
        #############################################

        ioSeg.save_segmentation_params(  save_folder, 
                                                        [os.path.split(fin)[-1] for fin in self.flist_in],
                                                        self.chosen_masks,
                                                        self.down_shapes, 
                                                        self.thinnings, 
                                                        self.smoothings )

        #############################################
        # generate final mask
        #############################################

        print('### Generating the smoothened masks.')
        for i in tqdm.tqdm(range(self.n_imgs)):
            folder, filename = os.path.split(self.flist_in[i])
            filename, extension = os.path.splitext(filename)
            # print(i, filename)
            
            if self.chosen_masks[i] == 'w':
                _rawmask = imread( os.path.join(self.imageFolder,'result_segmentation',filename+'_watershed'+extension) )
                mask = segment.smooth_mask( _rawmask, 
                                                    mode='watershed',
                                                    down_shape=self.down_shapes[i], 
                                                    smooth_order=self.smoothings[i] )
                while (np.sum(mask)==0)&(self.smoothings[i]>5):
                    print('Mask failed...')
                    # if mask is zero, try smoothing less
                    self.smoothings[i] -= 2
                    print('Trying with: smoothing', self.smoothings[i])
                    mask = segment.smooth_mask( _rawmask, 
                                                        mode='watershed',
                                                        down_shape=self.down_shapes[i], 
                                                        smooth_order=self.smoothings[i] )
                
            elif self.chosen_masks[i] == 'c':
                _rawmask = imread( os.path.join(self.imageFolder,'result_segmentation',filename+'_classifier'+extension) )
                mask = segment.smooth_mask( _rawmask, 
                                                    mode='classifier',
                                                    down_shape=self.down_shapes[i], 
                                                    smooth_order=self.smoothings[i],
                                                    thin_order=self.thinnings[i] )
                while (np.sum(mask)==0)&(self.smoothings[i]>5)&(self.thinnings[i]>1):
                    print('Mask failed...')
                    # if mask is zero, try smoothing less
                    self.smoothings[i] -= 2
                    self.thinnings[i] -= 1
                    print('Trying with: smoothing', self.smoothings[i],' thinnings', self.thinnings[i])
                    mask = segment.smooth_mask( _rawmask, 
                                                    mode='classifier',
                                                    down_shape=self.down_shapes[i], 
                                                    smooth_order=self.smoothings[i],
                                                    thin_order=self.thinnings[i] )
                
            elif self.chosen_masks[i] == 'm':
                if not os.path.exists(os.path.join(self.imageFolder,'result_segmentation',filename+'_manual'+extension)):
                    self.m = GUIs.manualmask.makeManualMask(self.flist_in[i])
                    self.m.show()
                    self.m.exec()
                else:
                    print('A previously generated manual mask exists!')
                _rawmask = imread( os.path.join(self.imageFolder,'result_segmentation',filename+'_manual'+extension) )
                mask = segment.smooth_mask( _rawmask, 
                                                    mode='manual',
                                                    down_shape=self.down_shapes[i], 
                                                    smooth_order=self.smoothings[i] )
                while (np.sum(mask)==0)&(self.smoothings[i]>5):
                    print('Mask failed...')
                    # if mask is zero, try smoothing less
                    self.smoothings[i] -= 2
                    print('Trying with: smoothing', self.smoothings[i])
                    # if mask is zero, try smoothing less
                    self.smoothings[i] -= 2
                    mask = segment.smooth_mask( _rawmask, 
                                                    mode='manual',
                                                    down_shape=self.down_shapes[i], 
                                                    smooth_order=self.smoothings[i] )
            elif self.chosen_masks[i] == 'i': 
                continue

            if np.sum(mask) == 0:
                QMessageBox.warning(self,'Warning, no trainingset!','The method selected didn\'t generate a valid mask. Please input the mask manually.')

                self.chosen_masks[i] = 'm'
                ioSeg.save_segmentation_params(  save_folder, 
                                                        [os.path.split(fin)[-1] for fin in self.flist_in],
                                                        self.chosen_masks,
                                                        self.down_shapes, 
                                                        self.thinnings, 
                                                        self.smoothings )
                if not os.path.exists(os.path.join(self.imageFolder,'result_segmentation',filename+'_manual'+extension)):
                    self.m = GUIs.manualmask.makeManualMask(self.flist_in[i])
                    self.m.show()
                    self.m.exec()
                else:
                    print('A previously generated manual mask exists!')
                _rawmask = imread( os.path.join(self.imageFolder,'result_segmentation',filename+'_manual'+extension) )
                mask = segment.smooth_mask( _rawmask, 
                                                    mode='manual',
                                                    down_shape=self.down_shapes[i], 
                                                    smooth_order=self.smoothings[i] )

            ioSeg.save_segmentation_params(  save_folder, 
                                                    [os.path.split(fin)[-1] for fin in self.flist_in],
                                                    self.chosen_masks,
                                                    self.down_shapes, 
                                                    self.thinnings, 
                                                    self.smoothings )


            # save final mask
            new_name = os.path.join(folder,'result_segmentation',filename+'_finalMask'+extension)
            imsave(new_name, mask)

        print('### Done computing masks!')

        #############################################
        # compute morphology
        #############################################

        # props = DatasetTools.morphology.computemorphology.compute_morphological_info(self.imageFolder, self.compute_meshgrid.isChecked())
        # DatasetTools.morphology.io.save_morpho_params(save_folder, cond, props)

        #############################################
        # generate recap
        #############################################

        w = overview.generate_overview_finalMask(self.imageFolder, 
                                                chosen=[c!='i' for c in self.chosen_masks], 
                                                saveFig=True, downshape=3)
        w.show()
    
    def moveToNext(self):
        self.read_segmentation_params()
        save_folder = os.path.join(self.imageFolder, 'result_segmentation')
        ioSeg.save_segmentation_params(  save_folder, 
                                                        [os.path.split(fin)[-1] for fin in self.flist_in],
                                                        self.chosen_masks,
                                                        self.down_shapes, 
                                                        self.thinnings, 
                                                        self.smoothings )

        new_stop = np.clip(self.stop+self.n_shown_max,0,self.n_imgs)
        if self.stop == new_stop:
            QApplication.beep()
            print('No more images to display!')
            return
        self.n_shown = new_stop - self.stop
        self.stop = new_stop
        self.start = self.stop-self.n_shown
        plt.close(self.overview)
        self.remake()
#        self.w = inspectionWindow_20max(self.imageFolder, parent=None, start=self.start, stop=self.stop)
#        self.w.show()
#        self.w.exec()
#        self.close()

    def moveToPrevious(self):
        self.read_segmentation_params()
        save_folder = os.path.join(self.imageFolder, 'result_segmentation')
        ioSeg.save_segmentation_params(  save_folder, 
                                                        [os.path.split(fin)[-1] for fin in self.flist_in],
                                                        self.chosen_masks,
                                                        self.down_shapes, 
                                                        self.thinnings, 
                                                        self.smoothings )

        new_start = np.clip(self.start-self.n_shown_max,0,self.n_imgs)
        if self.start == new_start:
            QApplication.beep()
            print('No previous images to display!')
            return
        self.n_shown = self.start - new_start
        self.start = new_start
        self.stop = self.start+self.n_shown# np.clip(self.stop-20,20,self.n_imgs)
        plt.close(self.overview)
        self.remake()
#        self.w = inspectionWindow_20max(self.imageFolder, parent=None, start=self.start, stop=self.stop)
#        self.w.show()
#        self.close()
        
    def remake(self):
        self.n_shown = self.stop-self.start
        self.showMore = False

        self.overview = MLModel.overview.generate_overview(self.imageFolder, saveFig=False, start = self.start, stop = self.stop, downshape=5)
        self.overview.show()

        self.flist_in, self.chosen_masks, self.down_shapes, self.thinnings, self.smoothings = ioSeg.load_segmentation_params( os.path.join(self.imageFolder,'result_segmentation') )
        self.flist_in = [os.path.join(self.imageFolder,i) for i in self.flist_in]

        for i in range(self.n_shown):
            name = os.path.splitext(os.path.split(self.flist_in[self.start+i])[-1])[0]
            self.imageName[i].setText(name)
            
            self.maskTypeSpaces[i].setCurrentIndex(['i','c','w','m'].index(self.chosen_masks[self.start+i]))

            self.down_scaleSpaces[i].setValue(self.down_shapes[self.start+i])

            self.thinningSpaces[i].setValue(self.thinnings[self.start+i])

            self.smoothingSpaces[i].setValue(self.smoothings[self.start+i])
        
