# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:46:26 2020

@author: gritti
"""

import os, glob, sys
import PyQt5.QtWidgets
from morgana.GUIs import inspection
from morgana.DatasetTools.segmentation import io as ioSeg
from morgana.DatasetTools import io as ioDT

###############################################################################

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

###############################################################################

if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    for image_folder in image_folders:

        ### compute parent folder as absolute path
        image_folder = os.path.abspath(image_folder)
    
        print('\n-------------'+image_folder+'------------\n')

        flist_in = ioDT.get_image_list(image_folder)
        n_imgs = len( flist_in )
        if os.path.exists(os.path.join(image_folder,'result_segmentation','segmentation_params.csv')):
            flist_in, chosen_masks, down_shapes, thinnings, smoothings = ioSeg.load_segmentation_params( os.path.join(image_folder,'result_segmentation') )
            flist_in = [os.path.join(image_folder,i) for i in flist_in]
        else:
            chosen_masks = ['w' for i in range(n_imgs)]
            down_shapes = [0.50 for i in range(n_imgs)]
            thinnings = [10 for i in range(n_imgs)]
            smoothings = [25 for i in range(n_imgs)]

        save_folder = os.path.join(image_folder, 'result_segmentation')
        ioSeg.save_segmentation_params(  save_folder, 
                                                        [os.path.split(fin)[-1] for fin in flist_in],
                                                        chosen_masks,
                                                        down_shapes, 
                                                        thinnings, 
                                                        smoothings )

        
        w = inspection.inspectionWindow_20max(
                image_folder, 
                parent=None, 
                start=0, 
                stop=20
                )
        w.show()
        app.exec()
    app.quit()