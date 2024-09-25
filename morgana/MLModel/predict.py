import os, time
import numpy as np
from skimage import transform, morphology, measure, segmentation
from sklearn.metrics import classification_report
from skimage.io import imread, imsave
import scipy.ndimage as ndi

from morgana.ImageTools import processfeatures

def create_features(_input, scaler,
                    gt=np.array([]),
                    sigmas=[.1,.5,1,2.5,5,7.5,10],
                    new_shape_scale=-1,
                    feature_mode='ilastik',
                    check_time=False):

    ### read in kwargs
    start = time.time()   
    if new_shape_scale != -1:
        shape = (int(_input.shape[0]*new_shape_scale), int(_input.shape[1]*new_shape_scale))
    else:
        shape = _input.shape
    if check_time:
        print(time.time()-start)

    # resize to new shape
    start = time.time()   
    _input = transform.resize(_input.astype(float), shape, preserve_range=True)
    if check_time:
        print(time.time()-start)

    # compute all features and normalize images
    start = time.time()   
    _input = processfeatures.get_features(_input, sigmas, feature_mode=feature_mode)
    _input = np.transpose( np.reshape(_input, (_input.shape[0], np.prod(shape))) ) # flatten the image
    if check_time:
        print(time.time()-start)

    start = time.time()   
    _input = scaler.transform(_input)
    if check_time:
        print(time.time()-start)
        
    return _input, shape

def predict(  _input, classifier,
                    shape=None,
                    check_time=False,
                    gt=np.array([]),
                    deep=False
                    ):
    if shape is None:
        shape = _input.shape

    # use classifier to predict image
    start = time.time()   
    if deep:
        y_prob = classifier.predict(_input) # predict probabilities of every pixel for every class
    else:
        y_prob = classifier.predict_proba(_input)
    y_pred = y_prob.argmax(axis=-1).astype(np.uint8)

    if check_time:
        print(time.time()-start)
        
    if gt:
        gt = transform.resize(gt, shape, order=0, preserve_range=False)
        gt = 1.*(gt>np.min(gt))
        gt = np.reshape(gt, np.prod(shape))
        print(classification_report(gt,y_pred))
        
    return y_pred, y_prob
    
def reshape(y_pred,y_prob,original_shape,shape,n_classes = 3, check_time=False):

    # reshape image back to 2D
    start = time.time()   
    y_pred = np.reshape(y_pred, shape)
    y_prob = np.reshape(np.transpose(y_prob), (n_classes, *shape))
    if check_time:
        print(time.time()-start)

    # resize to new shape
    start = time.time()   
    y_pred = transform.resize(y_pred, original_shape, order=0, preserve_range=True)
    y_prob = transform.resize(y_prob, (n_classes, *original_shape), order=0, preserve_range=True)
    if check_time:
        print(time.time()-start)
        
    return y_pred.astype(np.uint8), y_prob

def predict_image(  _input, classifier, scaler,
                    gt=np.array([]),
                    sigmas=[.1,.5,1,2.5,5,7.5,10],
                    new_shape_scale=-1,
                    feature_mode='ilastik',
                    check_time=False,
                    deep=False):
    original_shape = _input.shape
    n_classes = 3#len(classifier.classes_)

    _input, shape = create_features(_input, scaler,
                    gt=np.array([]),
                    sigmas=sigmas,
                    new_shape_scale=new_shape_scale,
                    feature_mode=feature_mode,
                    check_time=check_time)

    y_pred, y_prob = predict(  _input, classifier,
                                gt=gt,
                                check_time=check_time,
                                shape=shape,
                                deep=deep)

    y_pred, y_prob = reshape(y_pred,y_prob,
                             original_shape,shape,
                             n_classes = n_classes, 
                             check_time=check_time)

    return y_pred.astype(np.uint8), y_prob

def make_watershed( mask, edge,
                    new_shape_scale=-1 ):

    original_shape = mask.shape

    ### read in kwargs
    if new_shape_scale != -1:
        shape = (int(mask.shape[0]*new_shape_scale), int(mask.shape[1]*new_shape_scale))
    else:
        shape = mask.shape

    mask = transform.resize(mask.astype(float), shape, order=0, preserve_range=False)
    edge = transform.resize(edge, shape, order=0, preserve_range=False)
    edge = ((edge-np.min(edge))/(np.max(edge)-np.min(edge)))**2 # make mountains higher

    # label image and compute cm
    labeled_foreground = (mask > np.min(mask)).astype(int)
    properties = measure.regionprops(labeled_foreground, mask)
    if not properties:
        weighted_cm = np.array([shape[0]-1,shape[1]-1])
    else:
        center_of_mass = properties[0].centroid
        weighted_cm = properties[0].weighted_centroid
        weighted_cm = np.array(weighted_cm).astype(np.uint16)

    # move marker to local minimum
    loc_m = morphology.local_minima(np.clip(edge, 0, np.percentile(edge,90)), connectivity=10, indices=True)
    loc_m = np.transpose(np.stack([loc_m[0], loc_m[1]]))
    dist = [np.linalg.norm(weighted_cm-m) for m in loc_m]
    if len(dist)>0:
        weighted_cm = loc_m[dist.index(np.min(dist))]

    # move corner marker to local minimum
    corner = np.array([0,0])
    if edge[-1,-1]<edge[0,0]:
        corner = np.array([edge.shape[0]-1,edge.shape[1]-1])

    # generate seeds
    markers = np.zeros(edge.shape)
    markers[corner[0],corner[1]] = 1
    markers[weighted_cm[0],weighted_cm[1]] = 2

    # perform watershed
    labels = segmentation.watershed(edge, markers.astype(np.uint))
    labels =(labels-np.min(labels))/(np.max(labels)-np.min(labels))
    labels = transform.resize(labels, original_shape, order=0, preserve_range=False).astype(np.uint8)

    return labels

def predict_image_from_file(f_in, classifier, scaler, params, deep=False):

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

        pred, prob = predict_image( 
                            img,
                            classifier,
                            scaler,
                            sigmas = params['sigmas'],
                            new_shape_scale = params['down_shape'],
                            feature_mode = params['feature_mode'],
                            deep=deep
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
        mask_final = make_watershed(
                            mask_pred,
                            edge_prob,
                            new_shape_scale = params['down_shape'] 
                            )
    
        # save final mask
        imsave(new_name_watershed, mask_final)
    
    return None