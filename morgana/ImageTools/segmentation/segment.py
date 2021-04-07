import numpy as np
from scipy.ndimage import binary_fill_holes, label
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage import transform

def smooth_mask( _input,
                down_shape=-1, 
                mode='classifier',
                thin_order=10,
                smooth_order=25 ):
    # print('Smoothing...')

    ### read in kwargs
    original_shape = _input.shape
    if down_shape != -1:
        shape = (int(_input.shape[0]*down_shape), int(_input.shape[1]*down_shape))
    else:
        shape = _input.shape

    _input = transform.resize(_input.astype(float), shape, order=0, preserve_range=True)
    _input = _input/np.max(_input)
    _input = 1.*(_input>np.min(_input))

    if mode == 'classifier':
        # smooth
        _input = binary_fill_holes(_input)

        # remove edge objects
        negative = binary_fill_holes(_input==0)
        _input = _input*negative
        if np.sum(_input) == 0:
            return _input.astype(np.uint8)

        # keep only largest object
        labeled_mask, cc_num = label(_input)
        _input = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))
        _input = binary_fill_holes(_input)

        # thin out
        _input = binary_erosion(_input,disk(thin_order))
        if np.sum(_input) == 0:
            return _input.astype(np.uint8)

        # keep only largest object
        labeled_mask, cc_num = label(_input)
        _input = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))

    # smooth
    _input = binary_erosion(_input,disk(smooth_order))
    _input = binary_dilation(_input,disk(smooth_order))

    _input = transform.resize(_input.astype(float), original_shape, order=0, preserve_range=True)

    return _input.astype(np.uint8)
