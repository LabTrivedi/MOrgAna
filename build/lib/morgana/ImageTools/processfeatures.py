import numpy as np
from scipy.ndimage import gaussian_filter, laplace, gaussian_gradient_magnitude
from skimage.feature import daisy

def get_features(_input, sigmas, feature_mode='ilastik', 
                        radius=10, rings=5, histograms=8, orientations=8):
    # normalize image between 0 and 1
    _input = _input.astype(np.float64) 
    # percs = np.percentile(_input,q=(.03,99.997))
    # _input = np.clip(_input,percs[0],percs[1])
    # _input = (_input-percs[0])/(percs[1]-percs[0])

    out = np.expand_dims(_input,0)

    for sigma in sigmas:
        # gaussian smoothing
        gauss = gaussian_filter(_input,sigma)
        # laplacian of gaussian
        lapl = laplace(gauss)
        # gauss grad mag
        ggm = gaussian_gradient_magnitude(_input,sigma)
        # dog
        dog = gauss - _input

        # append results for the current sigma
        out = np.concatenate((out,np.stack([gauss,lapl,ggm,dog])),axis=0)

    if feature_mode == 'daisy':
        daisy_features = daisy(_input, step=1, 
                                    radius=radius, 
                                    rings=rings, 
                                    histograms=histograms, 
                                    orientations=orientations )
        daisy_features = np.moveaxis(daisy_features, -1, 0)
        daisy_features = np.stack([np.pad(i,(radius, radius),mode='edge') for i in daisy_features])
        out = np.concatenate((out,daisy_features),axis=0)

    return out
