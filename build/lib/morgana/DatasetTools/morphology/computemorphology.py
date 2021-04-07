import numpy as np
import pandas as pd
import os, tqdm
from skimage.io import imread
from itertools import repeat
import multiprocessing

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join('..','..'))

from morgana.DatasetTools import io
from morgana.ImageTools.morphology import computemorphology
from morgana.DatasetTools.segmentation import io as ioSeg
from morgana.DatasetTools.multiprocessing import istarmap

def compute_morphological_info(input_folder, compute_meshgrid=False, compute_locoefa=True):
    '''
    fdwafwvgrs
    '''

    print('### Computing morphology of images in:', input_folder)

    flist_all = io.get_image_list(input_folder)
    masks_folder = os.path.join(input_folder,'result_segmentation')
    _, chosen_mask, down_shape, _, _ = ioSeg.load_segmentation_params(masks_folder)
    flist_in = [flist_all[i] for i in range(len(flist_all)) if chosen_mask[i]!='i']
    flist_ma = io.get_image_list(masks_folder, string_filter='_finalMask.tif', mode_filter='include')

    # measure region props for every mask
    N_img = len(flist_in)
    
    # multiprocess
    N_cores = np.clip( int(0.8 * multiprocessing.cpu_count()),1,None )

    try:
        # try using multiprocessing
        df = pd.DataFrame({})
        pool = multiprocessing.Pool(N_cores)
        data_list = list(   tqdm.tqdm(
                                pool.istarmap(
                                    computemorphology.compute_morphological_info, 
                                    zip(    repeat( None ), 
                                            flist_in, flist_ma, down_shape, 
                                            repeat( compute_meshgrid ), 
                                            repeat( compute_locoefa ) ) ), 
                                    total = N_img ) )

        # print(data_list)
        for row in data_list:
            df = df.append(row, ignore_index=True)

    except ValueError:
        # if anything goes wrong, fall back to for loop processing
        df = pd.DataFrame({})
        for i in tqdm.tqdm(range(N_img)):
            f_in, f_ma = flist_in[i], flist_ma[i]
            mask = imread(f_ma)

            # compute new row
            row = computemorphology.compute_morphological_info(mask, f_in, f_ma, down_shape[i], compute_meshgrid, compute_locoefa=compute_locoefa)
            
            # concatenate  
            df = df.append(row, ignore_index=True)

    return df

if __name__ == '__main__':
    import DatasetTools.morphology.io
    import time

    input_folder = os.path.join('..','..','..','..','..','gastrSegment_testData','2020-02-20_David_TL','g03G')
    cond = 'g03G'

    _, cond = os.path.split(input_folder)
    save_folder = os.path.join(input_folder,'result_segmentation')
    fname = os.path.join(save_folder,cond+'_fluo_intensity.json')

    ## if morpho_params of straighten image not created yet, compute and save
    ## if not os.path.exists(fname):

    start = time.time()
    data = compute_morphological_info(input_folder)
    print(time.time()-start)
    print(data)
    DatasetTools.morphology.io.save_morpho_params( save_folder, cond, data )

    data = DatasetTools.morphology.io.load_morpho_params( save_folder, cond  )
    print(data)


