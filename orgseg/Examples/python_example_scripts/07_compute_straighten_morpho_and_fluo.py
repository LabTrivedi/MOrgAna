# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:59:16 2020

@author: gritti
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:24:40 2020

@author: gritti
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:46:26 2020

@author: gritti
"""

import os, glob, tqdm
from skimage.io import imread, imsave
import numpy as np

import sys
sys.path.append(os.path.join('..'))
import DatasetTools.io
import DatasetTools.morphology.overview
import DatasetTools.morphology.computemorphology
import DatasetTools.morphology.io
import DatasetTools.straightmorphology.computestraightmorphology
import DatasetTools.straightmorphology.io
import DatasetTools.fluorescence.computefluorescence
import DatasetTools.fluorescence.io

###############################################################################

image_folders = [
                    os.path.join('test_data','2020-09-22_conditions','init_150cells'),
                    os.path.join('test_data','2020-09-22_conditions','init_300cells'),
                ]

###############################################################################
    
if __name__ == '__main__':

    for image_folder in image_folders:
        
        ### compute parent folder as absolute path
        image_folder = os.path.abspath(image_folder)
    
        print('-------------'+image_folder+'------------')

        result_folder = os.path.join(image_folder, 'result_segmentation')
        folder, cond = os.path.split(image_folder)

        #######################################################################  
        ### compute composite and meshgrid overview
        
        file = '_composite_recap.tif'
        text = 'Composite files saved at:'
        parent,cond = os.path.split(image_folder)
        fname = os.path.join(image_folder,'result_segmentation', cond + file)
        text = text + '\n\t'+fname
        if not os.path.exists(os.path.join(result_folder,image_folder+file)):
            DatasetTools.morphology.overview.createCompositeOverview(image_folder, keep_open=False)        
        print(text)
                
        file = '_meshgrid_recap.png'
        text = 'Meshgrid files saved at:'
        parent,cond = os.path.split(image_folder)
        fname = os.path.join(image_folder,'result_segmentation', cond + file)
        text = text + '\n\t'+fname
        if not os.path.exists(os.path.join(result_folder,image_folder+file)):
            DatasetTools.morphology.overview.createMeshgridOverview(image_folder, keep_open=False)
        print(text)

        #######################################################################
        ### compute morphology if not computed
        compute_morphological_info = DatasetTools.morphology.computemorphology.compute_morphological_info
        save_morphological_info = DatasetTools.morphology.io.save_morpho_params

        file_extension = '_morpho_params.json'
        fname = os.path.join(result_folder,cond+file_extension)

        if not os.path.exists(fname):
            data = compute_morphological_info(image_folder, False)
            save_morphological_info(result_folder, cond, data)

        #######################################################################
        ### compute straight morphology if not computed
        
        compute_morphological_info = DatasetTools.straightmorphology.computestraightmorphology.compute_straight_morphological_info
        save_morphological_info = DatasetTools.straightmorphology.io.save_straight_morpho_params
        
        file_extension = '_morpho_straight_params.json'
        fname = os.path.join(result_folder,cond+file_extension)

        if not os.path.exists(fname):
            data = compute_morphological_info( image_folder )
            save_morphological_info( result_folder, cond, data )
            
        print("Computed all straight morphology information")

        #######################################################################  
        ### compute fluorescence info if not computed
        
        file_extension = '_fluo_intensity.json'
        fname = os.path.join(result_folder,cond+file_extension)

        if not os.path.exists(fname):
            data = DatasetTools.fluorescence.computefluorescence.compute_fluorescence_info( image_folder )
            DatasetTools.fluorescence.io.save_fluo_info( result_folder, cond, data )
        
        print("Computed all fluorescence information")
        
        #######################################################################        
        ### clean up watershed and classifier masks
        
        flist = glob.glob(os.path.join(result_folder,'*_watershed.tif'))            
        print('Cleaning up watershed masks')
        for f in tqdm.tqdm(flist):
            os.remove(f)
        
        flist = glob.glob(os.path.join(result_folder,'*_classifier.tif'))            
        print('Cleaning up classifier masks')
        for f in tqdm.tqdm(flist):
            os.remove(f)

        flist = glob.glob(os.path.join(result_folder,'*_manual.tif'))            
        print('Cleaning up manual masks')
        for f in tqdm.tqdm(flist):
            os.remove(f)

