import sys, os, time
sys.path.insert(0,os.path.join('..','..','..',))

from skimage.io import imread, imsave
import MLModel.train
import MLModel.io
import MLModel.predict
import DatasetTools.io
import scipy.ndimage as ndi

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

################################################
modelFolder = 'Y:\\Nicola_Gritti\\raw_data\\gastruloids_with_different_cell_number\\2021-01-28_gastr_cell_number_BraGFP2i_vs_BraGFPmKO21i\\deep_test\\model_braGFP_48h_deep'
imageFolder = 'Y:\\Nicola_Gritti\\raw_data\\gastruloids_with_different_cell_number\\2021-01-28_gastr_cell_number_BraGFP2i_vs_BraGFPmKO21i\\deep_test\\data\\init_150cells'

modelFolder = 'C:\\Users\\nicol\\Documents\\Repos\\deep_test\\model_braGFP_48h_deep'
imageFolder = 'C:\\Users\\nicol\\Documents\\Repos\\deep_test\\data\\init_150cells'
#############################################
# load images to be used as training set
#############################################
training_folder = os.path.join(modelFolder,'trainingset')
flist_in = DatasetTools.io.get_image_list(training_folder, string_filter='_GT', mode_filter='exclude')
img_train = []
for f in flist_in:
    img = imread(f)
    if len(img.shape) == 2:
        img = np.expand_dims(img,0)
    if img.shape[-1] == np.min(img.shape):
        img = np.moveaxis(img, -1, 0)
    img_train.append( img[0] )
# img_train = np.array(img_train)

flist_gt = DatasetTools.io.get_image_list(training_folder, string_filter='_GT', mode_filter='include')
gt_train = [ imread(f) for f in flist_gt ]
gt_train = [ g.astype(int) for g in gt_train ]

print('##### Training set:')
for i,f in enumerate(zip(flist_in,flist_gt)):
    print(i+1,'\t', os.path.split(f[0])[-1],'\t', os.path.split(f[1])[-1])

#############################################
# compute features and generate training set and weights
#############################################

params = { 'sigmas':       [.1,.5,1,2.5,5,7.5,10],
                        'down_shape':   0.50,
                        'edge_size':    5,
                        'fraction':     0.1,
                        'bias':         0.5,
                        'feature_mode': 'daisy' }

print('##### Generating training set...')
X, Y, w, scaler = MLModel.train.generate_training_set( img_train, 
                                        [g.astype(np.uint8) for g in gt_train], 
                                        sigmas=params['sigmas'],
                                        down_shape=params['down_shape'],
                                        edge_size=params['edge_size'],
                                        fraction=params['fraction'],
                                        feature_mode=params['feature_mode'],
                                        bias=params['bias'] )

#############################################
# Train the model
#############################################

print(X.shape, Y.shape)

print('##### Training model...')
start = time.time()
classifier = MLModel.train.train_classifier( X, Y, w, deep=True, epochs=10 )
print('Models trained in %.3f seconds.'%(time.time()-start))
# print('classes_: ', classifier.classes_)
# print('coef_: ', classifier.coef_)

#############################################
# Save the model
#############################################

MLModel.io.save_model( modelFolder,
                        classifier,
                        scaler,
                        sigmas=params['sigmas'],
                        down_shape=params['down_shape'],
                        edge_size=params['edge_size'],
                        fraction=params['fraction'],
                        feature_mode=params['feature_mode'],
                        bias=params['bias'],
                        deep=True )
print('##### Model saved!')

#############################################
# Load the model
#############################################

print('##### Loading classifier model and parameters...')
classifier, scaler, params = MLModel.io.load_model( modelFolder, deep=True )
print(params)
print('##### Model loaded!')

result_folder = os.path.join(imageFolder,'result_segmentation')
if not os.path.exists(result_folder):
    os.mkdir(result_folder)

flist_in = DatasetTools.io.get_image_list(imageFolder)
flist_in.sort()

for f_in in flist_in:

    print('#'*20+'\nLoading',f_in,'...')
    img = imread(f_in)
    if len(img.shape) == 2:
        img = np.expand_dims(img,0)
    if img.shape[-1] == np.min(img.shape):
        img = np.moveaxis(img, -1, 0)
    img = img[0]

    print('Predicting image...')
    #######################
    
    start = time.time()
    pred, prob = MLModel.predict.predict_image( img,
                        classifier,
                        scaler,
                        sigmas=params['sigmas'],
                        new_shape_scale=params['down_shape'],
                        feature_mode=params['feature_mode'],
                        deep=True,
                        check_time=True)
    print('Predicted in:',time.time()-start)
    
    ###########################
    
    # remove objects at the border
    start = time.time()
    negative = ndi.binary_fill_holes(pred==0)
    mask_pred = (pred==1)*negative
    edge_prob = ((2**16-1)*prob[2]).astype(np.uint16)
    mask_pred = mask_pred.astype(np.uint8)
    print('Remove edges in:',time.time()-start)

    # save mask
    start = time.time()
    parent, filename = os.path.split(f_in)
    filename, file_extension = os.path.splitext(filename)
    new_name = os.path.join(parent,'result_segmentation',filename+'_classifier'+file_extension)
    imsave(new_name, pred, check_contrast=False)
    print('Saved mask1 in:',time.time()-start)

    # perform watershed
    start = time.time()
    mask_final = MLModel.predict.make_watershed( mask_pred,
                                edge_prob,
                                new_shape_scale=params['down_shape'] )
    print('Watershed in:',time.time()-start)

    # save final mask
    start = time.time()
    parent, filename = os.path.split(f_in)
    filename, file_extension = os.path.splitext(filename)
    new_name = os.path.join(parent,'result_segmentation',filename+'_watershed'+file_extension)
    imsave(new_name, mask_final, check_contrast=False)
    print('Saved mask1 in:',time.time()-start)

print('All images done!')

