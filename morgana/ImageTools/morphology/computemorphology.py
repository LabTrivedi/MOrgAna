import numpy as np
import pandas as pd
import os
from skimage.io import imread
from scipy.ndimage import label
from skimage import measure, img_as_bool

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join('..','..'))

from morgana.ImageTools.morphology import anchorpoints, spline, midline, meshgrid
from morgana.ImageTools.locoefa import computecoeff

def compute_morphological_info( mask, f_in, f_ma, down_shape,
                            compute_meshgrid=False,
                            compute_locoefa=True,
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
        mask = img_as_bool( imread(f_ma).astype(float) )

    # print(f_in)
    # label mask
    labeled_mask, _ = label(mask)
    # compute morphological info
    props = measure.regionprops(labeled_mask)
    dict_ = {}
    for key in keys:
        dict_[key] = props[0][key]

    dict_['form_factor'] = dict_['perimeter']**2/(4*np.pi*dict_['area'])

    ## append file info that come as input and are not computed by default
    dict_['input_file'] = os.path.split(f_in)[1]
    dict_['mask_file'] = os.path.join('result_segmentation', os.path.split(f_ma)[1] )

    ### compute the anchor points, spline, midline and meshgrid (optional)
    bf = imread(f_in)
    if len(bf.shape) == 2:
        bf = np.expand_dims(bf,0)
    if bf.shape[-1] == np.min(bf.shape):
        bf = np.moveaxis(bf, -1, 0)
    bf = bf[0][dict_['slice']]
    ma = mask[dict_['slice']]
    anch = anchorpoints.compute_anchor_points(mask,dict_['slice'],down_shape)

    N_points, tck = spline.compute_spline_coeff(ma,bf,anch)

    diagonal = int(np.sqrt(ma.shape[0]**2+ma.shape[1]**2)/2)
    mid, tangent, width = midline.compute_midline_and_tangent(anch,N_points,tck,diagonal)
    mesh = None
    if compute_meshgrid:
        mesh = meshgrid.compute_meshgrid(mid, tangent, width)

    # store all these info in the dicionary
    dict_['anchor_points_midline'] = anch
    dict_['N_points_midline'] = N_points
    dict_['tck'] = tck
    dict_['midline'] = mid
    dict_['tangent'] = tangent
    dict_['meshgrid_width'] = width
    dict_['meshgrid'] = mesh
    if compute_locoefa:
        dict_['locoefa_coeff'] = computecoeff.compute_LOCOEFA_Lcoeff(mask, down_shape).locoefa_coeff.values
    else:
        dict_['locoefa_coeff'] = 0.

    return pd.Series(dict_)


if __name__ == '__main__':
    import DatasetTools.io
    import DatasetTools.segmentation.io
    input_folder = 'C:\\Users\\nicol\\Documents\\Repos\\gastrSegment_testData\\2020-02-20_David_TL\\g03G'
    input_folder = 'Y:\\Kerim_Anlas\\gastruloid_imaging\\PE_system\\timelapses\\2019-12-15_bragfp_dmso_sb024_xav2448_pd2448_10x_TL_48h__2019-12-15T15_19_49-Measurement 1\\dmso\\A02'

    flist_all = DatasetTools.io.get_image_list(input_folder)
    masks_folder = os.path.join(input_folder,'result_segmentation')
    _, chosen_mask, down_shape, _, _ = DatasetTools.segmentation.io.load_segmentation_params(masks_folder)
    flist_in = [flist_all[i] for i in range(len(flist_all)) if chosen_mask[i]!='i']
    flist_ma = DatasetTools.io.get_image_list(masks_folder, string_filter='_finalMask.tif', mode_filter='include')

    # measure region props for every mask
    keys = [
            'input_file',
            'mask_file',
            'centroid',
            'slice',
            'area',
            'eccentricity',
            'major_axis_length',
            'minor_axis_length',
            'equivalent_diameter',
            'perimeter',
            'anchor_points_midline',
            'N_points_midline',
            'tck',
            # 'y_tup',
            'midline',
            'tangent',
            'meshgrid_width',
            'meshgrid',
            'euler_number',
            'extent',
            'inertia_tensor',
            'inertia_tensor_eigvals',
            'moments',
            'moments_central',
            'moments_hu',
            'moments_normalized',
            'orientation',
            'locoefa_coeff'
            ]

    dict_ = {key: [] for key in keys }
    N_img = len(flist_in)

    # compute for the first mask
    i=0
    f_in, f_ma = flist_in[i], flist_ma[i]

    # load mask
    # print(i, os.path.split(f_in)[-1])
    mask = imread(f_ma)

    prop = compute_morphological_info(mask, f_in, f_ma, down_shape[i], False)

    # print(prop)
