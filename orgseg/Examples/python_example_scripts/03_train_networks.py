# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:38:10 2020

@author: gritti
"""

import tqdm
import os
from skimage.io import imread
import numpy as np
import time

# import sys
# sys.path.append(os.path.join('..'))
from orgseg.DatasetTools import io as ioDT
from orgseg.MLModel import io as ioML
from orgseg.MLModel import train

###############################################################################

model_folders = [
                    os.path.join('test_data','2020-09-22_conditions','model'),
                ]

### define parameters for network training
sigmas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0, 15.0, 50.0]
downscaling = 0.25
edge_size = 5
pxl_extract_fraction = 0.25
pxl_extract_bias = 0.4
feature_type = 'daisy'

###############################################################################

if __name__ == '__main__':

    ### compute parent folder as absolute path
    model_folders = [os.path.abspath(i) for i in model_folders]
    
    for model_folder in model_folders:
        print('-------------'+model_folder+'------------')

        training_folder = os.path.join(model_folder, 'trainingset')

        ### load images
        flist_in = ioDT.get_image_list(
                                                  training_folder, 
                                                  string_filter='_GT', 
                                                  mode_filter='exclude'
                                                  )
        img_train = []
        for f in flist_in:
            img = imread(f)
            if len(img.shape)==2:
                img = np.expand_dims(img,0)
            if img.shape[-1] == np.min(img.shape):
                img = np.moveaxis(img, -1, 0)
            img_train.append( img[0] )

        ## load ground truth
        flist_gt = ioDT.get_image_list(
                                                training_folder, 
                                                string_filter='_GT', 
                                                mode_filter='include'
                                                )
        gt_train = [ imread(f) for f in flist_gt ]
        gt_train = [ g.astype(int) for g in gt_train ]

        print('##### Training set:')
        for i,f in enumerate(zip(flist_in,flist_gt)):
            print(i+1,'\t', os.path.split(f[0])[-1],'\t', os.path.split(f[1])[-1])

        ###################################################################
        ### compute features and generate training set and weights

        print('##### Generating training set...')
        X, Y, w, scaler = train.generate_training_set( 
                                        img_train, 
                                        [g.astype(np.uint8) for g in gt_train], 
                                        sigmas = sigmas,
                                        down_shape = downscaling,
                                        edge_size = edge_size,
                                        fraction = pxl_extract_fraction,
                                        feature_mode = feature_type,
                                        bias = pxl_extract_bias 
                                        )

        ###################################################################
        ### Train the model

        print('##### Training model...')
        start = time.time()
        classifier = train.train_classifier( X, Y, w )
        print('Models trained in %.3f seconds.'%(time.time()-start))
        print('classes_: ', classifier.classes_)
        print('coef_: ', classifier.coef_)

        ###################################################################
        ### Save the model

        ioML.save_model( 
                        model_folder,
                        classifier,
                        scaler,
                        sigmas = sigmas,
                        down_shape = downscaling,
                        edge_size = edge_size,
                        fraction = pxl_extract_fraction,
                        feature_mode = feature_type,
                        bias = pxl_extract_bias
                        )
        print('##### Model saved!')

