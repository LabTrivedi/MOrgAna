import numpy as np
import pandas as pd
import os, tqdm
from scipy.ndimage import map_coordinates
from itertools import repeat
from skimage.io import imread
from scipy.ndimage import label
from skimage import measure, img_as_bool
import multiprocessing

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join('..','..'))

from morgana.DatasetTools.morphology import io, computemorphology
from morgana.ImageTools.fluorescence import computefluorescence

def compute_fluorescence_info( input_folder ):

    print('### Computing fluorescence info of images in:', input_folder)

    _, cond = os.path.split(input_folder)
    save_folder = os.path.join(input_folder,'result_segmentation')

    # if the morphological info are not computed yet, compute and save for the current folder
    morpho_file = os.path.join(save_folder,cond+'_morpho_params.json')
    if os.path.exists(morpho_file):
        props = io.load_morpho_params( save_folder, cond )
    else:
        props = computemorphology.compute_morphological_info(input_folder)
        io.save_morpho_params( save_folder, cond, props )

    # # if the staight morphological info are not computed yet, compute and save for the current folder
    # straight_morpho_file = os.path.join(save_folder,cond+'_morpho_straight_params.json')
    # if os.path.exists(straight_morpho_file):
    #     props_straight = DatasetTools.straightmorphology.io.load_straight_morpho_params( save_folder, cond )
    # else:
    #     props_straight = DatasetTools.straightmorphology.computestraightmorphology.compute_straight_morphological_info(input_folder)
    #     DatasetTools.straightmorphology.io.save_straight_morpho_params( save_folder, cond, props_straight )

    # measure region props for every mask
    flist_in = [prop['input_file'] for i, prop in props.iterrows()]
    flist_ma = [prop['mask_file'] for i, prop in props.iterrows()]
    N_img = len(flist_in)

    N_cores = np.clip(int(0.8 * multiprocessing.cpu_count()),1,None)

    try:

        df = pd.DataFrame({})
        props = [ {key: props[key][i] for key in props.keys()} for i in range(N_img) ]

        pool = multiprocessing.Pool(N_cores)
        data_list = list(   tqdm.tqdm(
                                pool.istarmap(
                                    computefluorescence.compute_fluorescence_info, 
                                    zip(    repeat( None ), repeat( None ), 
                                            flist_in, flist_ma, props,
                                            repeat( input_folder ) ) ), 
                                    total = N_img ) )
            # print(data_list)
        for row in data_list:
            df = df.append(row, ignore_index=True)

 
    except ValueError:
        print('Failed!!!')
        df = pd.DataFrame({})
        
        # compute all fluorescence profiles
        for i in tqdm.tqdm(range(N_img)):
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
                
            # compute all fluorescenc eproperties
            row = computefluorescence.compute_fluorescence_info(image, mask, f_in, f_ma, prop)
            
            # concatenate  
            df = df.append(row, ignore_index=True)
            
    return df

if __name__ == '__main__':
    import DatasetTools.fluorescence.io
    import time

    input_folder = os.path.join('..','..','..','..','..','gastrSegment_testData','2020-02-20_David_TL','g03G')
    cond = 'g03G'

    _, cond = os.path.split(input_folder)
    save_folder = os.path.join(input_folder,'result_segmentation')
    fname = os.path.join(save_folder,cond+'_fluo_intensity.json')

    ## if morpho_params of straighten image not created yet, compute and save
    ## if not os.path.exists(fname):
    start = time.time()
    data = compute_fluorescence_info(input_folder)
    print(time.time()-start)

    print(data.ch1_APprofile[0])
    DatasetTools.fluorescence.io.save_fluo_info( save_folder, cond, data )

    data = DatasetTools.fluorescence.io.load_fluo_info( save_folder, cond  )
    print(data.ch1_APprofile[0])

