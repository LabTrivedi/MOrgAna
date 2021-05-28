# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:38:10 2020

@author: gritti
"""

import tqdm, os, glob
from skimage.io import imread
import numpy as np
import time
from morgana.DatasetTools import io as ioDT
from morgana.MLModel import io as ioML
from morgana.MLModel import train

###############################################################################
# select folder containing all image folders to be analysed
parent_folder = os.path.join('test_data','2020-09-22_conditions')

print('Image subfolders found in: ' + parent_folder)
if os.path.exists(parent_folder):
    print('Path exists! Proceed!')# check if the path exists

model_folders = glob.glob(os.path.join(parent_folder,'model_*'))
model_folders = [os.path.abspath(i) for i in model_folders]

### define parameters for feature generation for network training
sigmas = [1.0, 5.0, 15.0]
downscaling = 0.25
edge_size = 5
pxl_extract_fraction = 0.25
pxl_extract_bias = 0.4
feature_type = 'daisy' # 'daisy' or 'ilastik'
deep = False # True: deep learning with Multi Layer Perceptrons; False: Logistic regression

###############################################################################

if __name__ == '__main__':
    
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
        classifier = train.train_classifier( X, Y, w, deep = deep )
        print('Models trained in %.3f seconds.'%(time.time()-start))
        # print('classes_: ', classifier.classes_)
        # print('coef_: ', classifier.coef_)

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
                        bias = pxl_extract_bias,
                        deep = deep
                        )
        print('##### Model saved!')

