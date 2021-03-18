import numpy as np
import pandas as pd
import os, tqdm
from scipy.ndimage import map_coordinates
from skimage.io import imread
from scipy.ndimage import label
from skimage import measure, img_as_bool

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join('..','..'))

from morgana.ImageTools.morphology import meshgrid
from morgana.ImageTools.locoefa import computecoeff

def compute_straight_morphological_info( mask, f_in, f_ma, down_shape, prop,
                            parent_folder = '',
                            compute_locoefa = True,
                            keys = [
                                    'centroid',
                                    'slice',
                                    'area',
                                    'eccentricity',
                                    'major_axis_length',
                                    'minor_axis_length',
                                    'equivalent_diameter',
                                    'perimeter',
                                    'euler_number',
                                    'extent',
                                    'inertia_tensor',
                                    'inertia_tensor_eigvals',
                                    'moments',
                                    'moments_central',
                                    'moments_hu',
                                    'moments_normalized',
                                    'orientation',
                                    ] ):
    if mask is None:
        # load mask
        path_to_mask = os.path.join(parent_folder,f_ma)
        mask = img_as_bool( imread(path_to_mask)[prop['slice']].astype(np.float) )
    
    # compute meshgrid if not computed in the previous step already
    tangent = prop['tangent']
    midline = prop['midline']
    width = prop['meshgrid_width']
    mesh = prop['meshgrid']
    if mesh == None:
        mesh = meshgrid.compute_meshgrid(
                                                                    midline,
                                                                    tangent,
                                                                    width
                                                                    )

    # straighten the mask
    ma_straight = np.reshape(map_coordinates(mask,np.reshape(mesh,(mesh.shape[0]*mesh.shape[1],2)).T,order=0,mode='constant',cval=0).T,(mesh.shape[0],mesh.shape[1]))

    # label straighetned mask
    labeled_mask, _ = label(ma_straight)
    
    # keep only larger object
    ma_straight = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))
    labeled_mask, _ = label(ma_straight)

    # compute morphological info
    props = measure.regionprops(labeled_mask)
    dict_ = {}
    for key in keys:
        dict_[key] = props[0][key]

    dict_['form_factor'] = dict_['perimeter']**2/(4*np.pi*dict_['area'])

    # append info that are not computed by default
    dict_['input_file'] = os.path.split(f_in)[1]
    dict_['mask_file'] = os.path.join('result_segmentation', os.path.split(f_ma)[1] )
    if compute_locoefa:
        dict_['locoefa_coeff'] = computecoeff.compute_LOCOEFA_Lcoeff(ma_straight, down_shape).locoefa_coeff.values
    else:
        dict_['locoefa_coeff'] = 0.
        
    return pd.Series(dict_)


if __name__ == '__main__':
    import DatasetTools.segmentation.io
    import DatasetTools.morphology.io

    input_folder = 'C:\\Users\\nicol\\Documents\\Repos\\gastrSegment_testData\\2019-11-30_control_esl2448_esl024_esl_72h\\control'
    input_folder = 'Y:\\Kerim_Anlas\\gastruloid_imaging\\PE_system\\timelapses\\2019-12-15_bragfp_dmso_sb024_xav2448_pd2448_10x_TL_48h__2019-12-15T15_19_49-Measurement 1\\dmso\\A02'

    _, cond = os.path.split(input_folder)
    save_folder = os.path.join(input_folder,'result_segmentation')
    _, _, down_shape, _, _ = DatasetTools.segmentation.io.load_segmentation_params(save_folder)

    morpho_file = os.path.join(save_folder,cond+'_morpho_params.json')
    if os.path.exists(morpho_file):
        props = DatasetTools.morphology.io.load_morpho_params( save_folder, cond )

    i=3
    prop = {key: props[key][i] for key in props.keys()}
    f_ma = prop['mask_file']
    f_in = prop['input_file']

    # load mask
    path_to_mask = os.path.join(input_folder,f_ma)
    mask = img_as_bool( imread(path_to_mask)[prop['slice']].astype(np.float) )

    # compute meshgrid if not computed in the previous step already
    tangent = prop['tangent']
    midline = prop['midline']
    width = prop['meshgrid_width']
    mesh = prop['meshgrid']
    if mesh == None:
        mesh = ImageTools.morphology.meshgrid.compute_meshgrid(
                                                                    midline,
                                                                    tangent,
                                                                    width
                                                                    )

    # straighten the mask
    ma_straight = np.reshape(map_coordinates(mask,np.reshape(mesh,(mesh.shape[0]*mesh.shape[1],2)).T,order=0,mode='constant',cval=0).T,(mesh.shape[0],mesh.shape[1]))

    # label straighetned mask
    labeled_mask, _ = label(ma_straight)

    # compute morphological info
    props = measure.regionprops(labeled_mask)
    dict_ = {}
    # for key in keys:
    #     dict_[key] = props[0][key]

    # append info that are not computed by default
    dict_['input_file'] = os.path.split(f_in)[1]
    dict_['mask_file'] = os.path.join('result_segmentation', os.path.split(f_ma)[1] )
    print(np.max(ma_straight.astype(float)))
    dict_['locoefa_coeff'] = ImageTools.locoefa.computecoeff.compute_LOCOEFA_Lcoeff(ma_straight, down_shape[i]).locoefa_coeff.values

