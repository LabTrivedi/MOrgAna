# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:26:12 2020

@author: gritti
"""

import tqdm
import os
import glob
import shutil

###############################################################################

parent_folder = os.path.join('test_data','2020-09-22_conditions')

start = 0
dN = 5
combine_subfolders = True
   
### add folders that you want to ignore here
exclude_folder = []
    
###############################################################################

def initialize_model_folder(folder, dN=30, start=0, combine=True):
    
    ### create folders
    if combine:
        model_folder = os.path.join(os.path.split(folder)[0],'model')
    else:
        model_folder = os.path.join(folder,'model')

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
        
    trainingset_folder = os.path.join(model_folder,'trainingset')
    if not os.path.exists(trainingset_folder):
        os.mkdir(trainingset_folder)
    
    ### count images and extract trainingset file names
    flist = glob.glob(os.path.join(folder,'*.tif'))
    flist.sort()
    flist = flist[start::dN]
    
    ### copy images to trainingset folder
    for f in flist:
        fname = os.path.split(f)[-1]
        newf = os.path.join(trainingset_folder,fname)
        if not os.path.exists(newf):
            shutil.copy(f,newf)
    
###############################################################################

if __name__ == '__main__':

    ### compute parent folder as absolute path
    parent_folder = os.path.abspath(parent_folder)
    print(parent_folder)
    
    ### find out all gastruloids in parent_folder
    folder_names = next(os.walk(parent_folder))[1] 
    
    ### exclude gastruloids
    folder_names = [g for g in folder_names if not g in exclude_folder ]

    for folder_name in tqdm.tqdm(folder_names):
        if not folder_name in exclude_folder:
            folder_path = os.path.join(parent_folder, folder_name)

            ### for every gastruloid, generate model folder and the
            ### trainingset (one image every 30)
            initialize_model_folder(folder_path, dN=dN, start=start, combine=combine_subfolders)
