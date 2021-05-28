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

### Select morphological parameters to be quantified

maskType = 'Strightened' # Use 'Unprocessed' or 'Straightened' binary mask
Timelapse = False # Do images in the folder belong to a timelapse?

all_morpho_params = False # select true if all parameters are to be used.
# otherwise, select which paramters you would like to compute.
area = False
eccentricity = True
major_axis_length = False
minor_axis_length = False
equivalent_diameter = False
perimeter = False
euler_number = False
extent = False
form_factor = True
orientation = False
locoefa_coeff = True

# Define number of groups/number of conditions and the image subfolders that belong to each group
group1 = [image_folders[0], image_folders[1]]
group2 = [image_folders[2]]
group3 = [image_folders[3]]

groups = [group1, group2, group3]

###########################################################################

### Show morphology plots

morphoKeys = ['area',
              'eccentricity',
              'major_axis_length',
              'minor_axis_length',
              'equivalent_diameter',
              'perimeter',
              'euler_number',
              'extent',
              'form_factor',
              'orientation',
              'locoefa_coeff']

if all_morpho_params:
    computeMorpho = [True for key in morphoKeys]
else:
    computeMorpho = [area, eccentricity, major_axis_length, minor_axis_length, equivalent_diameter, 
                     perimeter, euler_number, extent, form_factor, orientation, locoefa_coeff]
    
# extract data from all the folders
data_all, keys = arrangemorphodata.collect_morpho_data( groups, 
                                                        morphoKeys, 
                                                        computeMorpho, 
                                                        maskType, 
                                                        Timelapse
                                                        )
quantifier = []
app = PyQt5.QtWidgets.QApplication(sys.argv)

# for every quantification parameter, make the appropriate plot
for key in keys:
    data_key = [data[key] for data in data_all]

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
    
    if not PyQt5.QtWidgets.QApplication.instance():
        app = PyQt5.QtWidgets.QApplication(sys.argv)
    else:
        app = PyQt5.QtWidgets.QApplication.instance() 
        
    # call the right visualization tool according to the number of dimensions
    ### clean up quantifier handler:
    quantifier = [quantifier[i] for i in range(len(quantifier)) if quantifier[i] is not None]

    if ndim == 0:
        quantifier.append( visualize0d.visualization_0d( data_key, key ) )
        quantifier[-1].show()
    elif ndim == 1:  
        quantifier.append( visualize1d.visualization_1d( data_key, key ) )
        quantifier[-1].show()
    elif ndim == 2:
        quantifier.append( visualize2d.visualization_2d( data_key, key ) )
        quantifier[-1].show()
    app.exec()
app.quit()

###########################################################################