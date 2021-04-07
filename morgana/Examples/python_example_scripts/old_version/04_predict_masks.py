# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:26:06 2020

@author: gritti
"""

import os, tqdm
from skimage.io import imread, imsave
import numpy as np
import scipy.ndimage as ndi
import multiprocessing
from itertools import repeat

# import sys
# sys.path.append(os.path.join('..'))
from orgseg.DatasetTools import io as ioDT
import orgseg.DatasetTools.multiprocessing.istarmap
from orgseg.MLModel import io as ioML
from orgseg.MLModel import predict

###############################################################################

image_folders = [
                    os.path.join('test_data','2020-09-22_conditions','init_150cells'),
                    os.path.join('test_data','2020-09-22_conditions','init_300cells'),
                ]

model_folder = os.path.join('test_data','2020-09-22_conditions','model')

###############################################################################

def predict_single_image(f_in, classifier, scaler, params):

    parent, filename = os.path.split(f_in)
    filename, file_extension = os.path.splitext(filename)
    new_name_classifier = os.path.join(
                    parent,
                    'result_segmentation',
                    filename+'_classifier'+file_extension
                    )
    new_name_watershed = os.path.join(
                    parent,
                    'result_segmentation',
                    filename+'_watershed'+file_extension
                    )

#    print('#'*20+'\nLoading',f_in,'...')
    img = imread(f_in)
    if len(img.shape)==2:
        img = np.expand_dims(img,0)
    if img.shape[-1] == np.min(img.shape):
        img = np.moveaxis(img, -1, 0)
    img = img[0]

    if not os.path.exists(new_name_classifier):
        # print('Predicting image...')

        pred, prob = predict.predict_image( 
                            img,
                            classifier,
                            scaler,
                            sigmas = params['sigmas'],
                            new_shape_scale = params['down_shape'],
                            feature_mode = params['feature_mode']
                            )
    
        # remove objects at the border
        negative = ndi.binary_fill_holes(pred==0)
        mask_pred = (pred==1)*negative
        edge_prob = ((2**16-1)*prob[2]).astype(np.uint16)
        mask_pred = mask_pred.astype(np.uint8)
    
        # save mask
        imsave(new_name_classifier, pred)

    if not os.path.exists(new_name_watershed):
        # perform watershed
        mask_final = predict.make_watershed(
                            mask_pred,
                            edge_prob,
                            new_shape_scale = params['down_shape'] 
                            )
    
        # save final mask
        imsave(new_name_watershed, mask_final)
    
    return None

###############################################################################
    
if __name__ == '__main__':
    
    for image_folder in image_folders:

        ### compute parent folder as absolute path
        image_folder = os.path.abspath(image_folder)
    
        print('-------------'+image_folder+'------------')
        training_folder = os.path.join(model_folder, 'trainingset')

        print('##### Loading classifier model and parameters...')
        classifier, scaler, params = ioML.load_model( model_folder )
        print('##### Model loaded!')
              
        #######################################################################
        ### apply classifiers and save images

        result_folder = os.path.join(image_folder, 'result_segmentation')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        flist_in = ioDT.get_image_list(image_folder)
        flist_in.sort()        
        N_img = len(flist_in)
        
        # multiprocess
        N_cores = np.clip( int(0.8 * multiprocessing.cpu_count()),1,None )

        # try using multiprocessing
        pool = multiprocessing.Pool(N_cores)
        _ = list(   tqdm.tqdm(
                                pool.istarmap(
                                    predict_single_image, 
                                    zip(    flist_in, 
                                            repeat(classifier),
                                            repeat(scaler),
                                            repeat(params) ) ), 
                                    total = N_img ) )

        print('All images done!')
