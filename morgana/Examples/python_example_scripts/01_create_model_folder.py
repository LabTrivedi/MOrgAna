# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:26:12 2020

@author: gritti
"""

import os, glob, shutil, tqdm
import numpy as np
from numpy.random import default_rng

###############################################################################
# select folder containing all image folders to be analysed
parent_folder = os.path.join('test_data','2020-09-22_conditions')

print('Image subfolders found in: ' + parent_folder)
if os.path.exists(parent_folder):
    print('Path exists! Proceed!')# check if the path exists

# select images for training dataset
start = 0 # increase value to exclude starting images in dataset
dN = 0 # every dNth image will be used for the training dataset; if dN = 0, random images are taken

# True: create one model for all folders; False: create one model for each image subfolder
combine_subfolders = True
   
# add folders that you want to ignore here
exclude_folder = ['model_']
    
###############################################################################

def initialize_model_folder(folder, dN=30, start=0, combine=True):
    
    # create folders
    if combine:
        model_folder = os.path.join(os.path.split(folder)[0],'model_')
    else:
        model_folder = os.path.join(os.path.split(folder)[0], 'model_' + os.path.split(folder)[1])

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
        
    trainingset_folder = os.path.join(model_folder,'trainingset')
    if not os.path.exists(trainingset_folder):
        os.mkdir(trainingset_folder)
    
    # count images and extract trainingset file names
    flist = glob.glob(os.path.join(folder,'*.tif'))
    flist.sort()
    if dN:
        flist = flist[start::dN]
    else: 
        rng = default_rng()
        random_choice = rng.choice(len(flist), size=np.clip(len(flist)//10, 1, None), replace=False)
        flist = [flist[i] for i in random_choice]

    
    # copy images to trainingset folder
    for f in flist:
        fname = os.path.split(folder)[1] + '_' + os.path.split(f)[-1]
        newf = os.path.join(trainingset_folder,fname)
        if not os.path.exists(newf):
            shutil.copy(f,newf)
    
###############################################################################

if __name__ == '__main__':

    # compute parent folder as absolute path
    parent_folder = os.path.abspath(parent_folder)
    
    # find out all image subfolders in parent_folder
    folder_names = next(os.walk(parent_folder))[1] 
    
    # exclude folders in exclude_folder
    folder_names = [g for g in folder_names if not g in exclude_folder ]

    for folder_name in tqdm.tqdm(folder_names):
        if not folder_name in exclude_folder:
            folder_path = os.path.join(parent_folder, folder_name)

            # for the parent_folder/every image subfolder, generate model folder and the trainingset
            initialize_model_folder(folder_path, dN=dN, start=start, combine=combine_subfolders)