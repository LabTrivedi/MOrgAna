import os
import pandas as pd

def save_segmentation_params( save_folder,
                            filename,
                            chosen_mask,
                            down_shape,
                            thinning,
                            smoothing ):

    # Make it work for Python 2+3 and with Unicode
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    params = pd.DataFrame( {  'filename': filename,
                'chosen_mask': chosen_mask,
                'down_shape': down_shape,
                'thinning': thinning,
                'smoothing': smoothing } )
    column_order = ['filename', 'chosen_mask', 'down_shape', 'thinning', 'smoothing']
    
#    print('Saving segmentation parameter file.........')
    params[column_order].to_csv(os.path.join(save_folder,'segmentation_params.csv'))

def load_segmentation_params( save_folder ):

    # with open(os.path.join(save_folder,'segmentation_params.json'), 'r') as f:
    #     params = json.load(f)
    params = pd.read_csv(os.path.join(save_folder,'segmentation_params.csv'))
    filename = params['filename']
    chosen_mask = params['chosen_mask']
    down_shape = params['down_shape']
    for i in range(len(down_shape)):
        if down_shape[i]==500:
            down_shape[i]=500./2160.
    thinning = params['thinning']
    smoothing = params['smoothing']
    return filename, chosen_mask, down_shape, thinning, smoothing
