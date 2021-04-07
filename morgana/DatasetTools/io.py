import os, glob
import numpy as np

def get_image_list(input_folder, string_filter='', mode_filter='include'):

    flist = glob.glob(os.path.join(input_folder,'*.tif'))
    if string_filter:
        if mode_filter=='include':
            flist = [f for f in flist if string_filter in f]
        elif mode_filter=='exclude':
            flist = [f for f in flist if string_filter not in f]
    flist.sort()

    return flist