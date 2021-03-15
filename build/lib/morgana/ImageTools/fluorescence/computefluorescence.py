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
from morgana.ImageTools.fluorescence import computeprofiles

def compute_fluorescence_info( image, mask, f_in, f_ma, prop, parent_folder = '' ):

    if image is None:
        # load image
        path_to_file = os.path.join(parent_folder,f_in)
        image = imread(path_to_file)
        if image.ndim == 2:
            image = np.expand_dims(image,0)
        if image.shape[-1] == np.min(image.shape):
            image = np.moveaxis(image, -1, 0)
        image = np.stack([ img[prop['slice']].astype(np.float) for img in image ])

    if mask is None:
        # load mask
        path_to_mask = os.path.join(parent_folder,f_ma)
        mask = img_as_bool( imread(path_to_mask)[prop['slice']].astype(np.float) )

    # make sure the input image is a 3D numpy array even if it has only one channel
    N_ch = image.shape[0]

    # setup the dictionary
    dict_ = {}
    dict_['input_file'] = prop['input_file']
    dict_['mask_file'] = prop['mask_file']
    for ch in range(N_ch):
        dict_['ch%d_APprofile'%ch] = []
        dict_['ch%d_LRprofile'%ch] = []
        dict_['ch%d_RADprofile'%ch] = []
        dict_['ch%d_ANGprofile'%ch] = []
        dict_['ch%d_Background'%ch] = []
        dict_['ch%d_Average'%ch] = []

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

    for ch in range(N_ch):
        fl = image[ch]
        # compute average and background in the current channel
        dict_['ch%d_Background'%ch] = np.mean(fl[mask==0])
        dict_['ch%d_Average'%ch] = np.mean(fl[mask])

        # compute meshgrid and straighten
        ap, lr, rad, ang = computeprofiles.compute_profiles_fluo(fl, mask, mesh, visualize=False)
        for j, v in enumerate(ap):
            if not np.isfinite(v):
                ap[j] = 0
        # make the nans equal to the first or last finite element
        left = np.array(ap)[np.array(ap)>0][0]
        right = np.array(ap)[np.array(ap)>0][-1]
        for j, v in enumerate(ap):
            if v==0 and (j<(len(ap)/2)):
                ap[j] = left
            if v==0 and (j>(len(ap)/2)):
                ap[j] = right

        dict_['ch%d_APprofile'%ch] = ap
        dict_['ch%d_LRprofile'%ch] = lr
        dict_['ch%d_RADprofile'%ch] = rad
        dict_['ch%d_ANGprofile'%ch] = ang

        # transform the sub dictionary into a dataframe
        # dict_['ch%d'%ch] = pd.Series(dict_['ch%d'%ch])
        
    return pd.Series( dict_ )

if __name__ == '__main__':
    import DatasetTools.morphology.io

    input_folder = os.path.join('..','..','..','..','..','gastrSegment_testData','2020-02-20_David_TL','g03G')
    save_folder = os.path.join(input_folder, 'result_segmentation')
    cond = 'g03G'
    props = DatasetTools.morphology.io.load_morpho_params( save_folder, cond )

    # compute for the first image
    i = 0
    prop = {key: props[key][i] for key in props.keys()}
    f_ma = prop['mask_file']
    f_in = prop['input_file']

    # load input image and mask
    path_to_mask = os.path.join(input_folder,f_ma)
    path_to_file = os.path.join(input_folder,f_in)
    mask = img_as_bool( imread(path_to_mask)[prop['slice']].astype(np.float) )
    image = imread(path_to_file)
    image = np.stack([ img[prop['slice']].astype(np.float) for img in image ])

    # make sure the input image is a 3D numpy array even if it has only one channel
    if image.ndim == 2:
        image = np.expand_dims(image,0)
    if image.shape[-1] == np.min(image.shape):
        image = np.moveaxis(image, -1, 0)
    N_ch = image.shape[0]

    # compute all fluorescenc eproperties
    prop = compute_fluorescence_info(image, mask, f_in, f_ma, prop)

    print(prop)
