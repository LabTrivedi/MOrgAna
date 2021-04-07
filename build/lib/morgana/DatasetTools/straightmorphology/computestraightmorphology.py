import numpy as np
import pandas as pd
import os, tqdm
from scipy.ndimage import map_coordinates
from skimage.io import imread
from scipy.ndimage import label
from skimage import measure, img_as_bool
import multiprocessing
from itertools import repeat

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join('..','..'))

from morgana.DatasetTools.segmentation import io as ioSeg
from morgana.DatasetTools.morphology import io as ioMorph
from morgana.DatasetTools.morphology import computemorphology
from morgana.ImageTools.straightmorphology import computestraightmorphology

def compute_straight_morphological_info(input_folder, compute_locoefa = True,):

    print('### Computing straightened morphology of images in:', input_folder)

    _, cond = os.path.split(input_folder)
    save_folder = os.path.join(input_folder,'result_segmentation')
    _, _, down_shape, _, _ = ioSeg.load_segmentation_params(save_folder)

    # if the morphological info are not computed yet, compute and save for the current folder
    morpho_file = os.path.join(save_folder,cond+'_morpho_params.json')
    if os.path.exists(morpho_file):
        props = ioMorph.load_morpho_params( save_folder, cond )
    else:
        props = computemorphology.compute_morphological_info(input_folder, compute_meshgrid=False)
        ioMorph.save_morpho_params( save_folder, cond, props )

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
                                    computestraightmorphology.compute_straight_morphological_info, 
                                    zip(    repeat( None ), 
                                            flist_in, flist_ma, down_shape, props,
                                            repeat( input_folder ),
                                            repeat( compute_locoefa ) ) ), 
                                    total = N_img ) )
            # print(data_list)
        for row in data_list:
            df = df.append(row, ignore_index=True)

    except ValueError:
        df = pd.DataFrame({})
        
        # compute eccentricity on the straightened mask
        for i in tqdm.tqdm(range(N_img)):
            prop = {key: props[key][i] for key in props.keys()}
            f_ma = prop['mask_file']
            f_in = prop['input_file']

            # load mask
            path_to_mask = os.path.join(input_folder,f_ma)
            mask = img_as_bool( imread(path_to_mask)[prop['slice']].astype(np.float) )

            row = computestraightmorphology.compute_straight_morphological_info(mask, f_in, f_ma, down_shape[i], prop, compute_locoefa=compute_locoefa)
            
            # concatenate  
            df = df.append(row, ignore_index=True)

    return df
        

if __name__ == '__main__':
    import DatasetTools.straightmorphology.io
    import time

    input_folder = os.path.join('..','..','..','..','..','gastrSegment_testData','2020-02-20_David_TL','g03G')
    cond = 'g03G'

    _, cond = os.path.split(input_folder)
    save_folder = os.path.join(input_folder,'result_segmentation')
    fname = os.path.join(save_folder,cond+'_fluo_intensity.json')

    ## if morpho_params of straighten image not created yet, compute and save
    ## if not os.path.exists(fname):

    start = time.time()
    data = compute_straight_morphological_info(input_folder)
    print(time.time()-start)
    print(data)
    DatasetTools.straightmorphology.io.save_straight_morpho_params( save_folder, cond, data )

    data = DatasetTools.straightmorphology.io.load_straight_morpho_params( save_folder, cond  )
    print(data)


