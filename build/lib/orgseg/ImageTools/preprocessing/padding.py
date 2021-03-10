import numpy as np
from skimage.io import imread, imsave

def pad_images( img, pad_width_ratio=0.1 ):
    '''
    pad a 2D (+channel) image so that the outpute shape is img.shape*1.1
    the pixel values on the edge follow a normal distribution with :
    - mean = avg-1*std of the outer 0.1 layer of the image
    - std = std of the outer 0.1 layer of the image

    Note:
    - this function is useful when user wants to study objects touching the boundary
    - Images will be saved in the "padded_images" subfolder
    - User should use this subfolder as input for further analysis
    '''
    _type = img.dtype

    if img.ndim==2:
        img = np.expand_dims(img, 0)

    shape = img.shape[1:]
    layer_width = (int(shape[0]*0.1), int(shape[1]*0.1))
    outermask = np.ones(shape)
    outermask[layer_width[0]:-layer_width[0],
            layer_width[1]:-layer_width[1]] = 0
    
    img_padded = []
    for i in img:

        # compute mean and std of outer layer
        img_masked = i * outermask
        img_masked[outermask==0] = np.nan

        mean = np.nanmean(img_masked)
        std = np.nanstd(img_masked)
        mean = mean-std

        # pad image plane with nans
        i_padded = np.pad(i, ((layer_width[0],layer_width[0]),(layer_width[1],layer_width[1])), mode='constant', constant_values=0)

        # padding values
        padding_values = np.clip(mean+std*np.random.randn(*i_padded.shape),0,None).astype(np.uint16)
        padding_values[layer_width[0]:-layer_width[0],
            layer_width[1]:-layer_width[1]] = 0

        # sum the padding values
        i_padded = i_padded+padding_values

        # place image plane in full image
        img_padded.append(i_padded)
    
    img_padded = np.array(img_padded)

    return img_padded.astype(_type)

