# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:36:41 2021

@author: gritti
"""

import sys, time, tqdm, copy, os, glob
from skimage.io import imread, imsave
import numpy as np
import PyQt5.QtWidgets
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")
from morgana.DatasetTools.morphology import overview as overviewDT
from morgana.DatasetTools import arrangemorphodata
from morgana.DatasetTools import arrangefluodata
from collections.abc import Iterable
from morgana.GUIs import visualize0d
from morgana.GUIs import visualize1d
from morgana.GUIs import visualize2d


#####################################################

# select folder containing all image folders to be analysed
parent_folder = os.path.join('test_data','2020-09-22_conditions')
print('Image subfolders found in: ' + parent_folder)
if os.path.exists(parent_folder):
    print('Path exists! Proceed!')# check if the path exists

# find out all image subfolders in parent_folder
folder_names = next(os.walk(parent_folder))[1] 

model_folders = glob.glob(os.path.join(parent_folder,'model_*'))
model_folders_name = [os.path.split(model_folder)[-1] for model_folder in model_folders]

# exclude folders in exclude_folder
exclude_folder = ['']

image_folders = [g for g in folder_names if not g in model_folders_name + exclude_folder]
image_folders = [os.path.join(parent_folder, i) for i in image_folders]

###########################################################################

### Create Composite & Meshgrid Overviews for each image subfolder
for image_folder in image_folders:
    overviewDT.createCompositeOverview(image_folder)
    overviewDT.createMeshgridOverview(image_folder)
    parent,cond = os.path.split(image_folder)
    text = 'Composite and Meshgrid files saved at:' + '\n\t'+ os.path.join(os.path.split(parent)[-1],'result_segmentation', cond)
    print('Completed successfully. ' + text)
print('All overviews created.')





