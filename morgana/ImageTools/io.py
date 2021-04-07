from skimage.io import imread
import numpy as np

def load_images_ch0(fnames, downshape):
    print('Using multiple cores')
    N_files = len(fnames)
    img = [[] for i in fnames]
    for i in range(N_files):
        fname = fnames[i]
        img1 = imread(fname).astype(float)
        if img1.ndim == 2:
            img1 = np.expand_dims(img1,0)
        if img1.shape[-1] == np.min(img1.shape):
            img1 = np.moveaxis(img1, -1, 0)
        img[i] = img1[0,::downshape,::downshape]

    return img
