import os
from morgana.DatasetTools.morphology import overview as overviewDT

if __name__=="__main__":

    input_folder = "V:/Lilli_Hahn/data/EH23H_signaling-perturbation/EH24H_DIF01/MOrgAna/images/Control_CHIR/24h"

    """
    overviewDT.createCompositeOverview(input_folder)
    """
    import numpy as np
    import os, tqdm
    from skimage.io import imread, imsave
    from skimage import img_as_bool
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from textwrap import wrap
    from matplotlib import rc
    rc('font', size=17)
    rc('font', family='Arial')
    # rc('font', serif='Times')
    rc('pdf', fonttype=42)
    # rc('text', usetex=True)
    from itertools import repeat
    import multiprocessing

    from morgana.DatasetTools import io
    from morgana.DatasetTools.morphology import computemorphology
    from morgana.DatasetTools.morphology import io as ioMorph
    from morgana.ImageTools import compositeImageJ
    from morgana.ImageTools.morphology import meshgrid


    print('### Generating recap composite movie at',input_folder)
        
    
    _, cond = os.path.split(input_folder)
    segment_folder = os.path.join(input_folder,'result_segmentation')
    
    file_extension = '_morpho_params.json'
    fname = os.path.join(segment_folder,cond+file_extension)

    props = ioMorph.load_morpho_params(segment_folder, cond)

    flist_in = [ os.path.join(input_folder, i) for i in props['input_file'] ] 
    # flist_mask = [ os.path.join(input_folder, i) for i in props['mask_file'] ]
    
    n_imgs = len(flist_in)

    dims = np.zeros((n_imgs,2))
    for i in range(n_imgs):
        _slice = np.array(props['slice'][i])
        dims[i][0] = _slice[0].stop-_slice[0].start
        dims[i][1] = _slice[1].stop-_slice[1].start
        
    max_dim = np.max(dims,0).astype(np.uint16)
    # make sure max_dim is even!
    for i in range(len(max_dim)):
        if np.mod(max_dim[i],2)!=0:
            max_dim[i] += 1
    
    img = imread(flist_in[0])
    if len(img.shape) == 2:
        img = np.expand_dims(img,0)
    if img.shape[-1] == np.min(img.shape):
        img = np.moveaxis(img, -1, 0)
    n_ch = img.shape[0]
    movie = np.zeros((n_imgs,1,n_ch,max_dim[0],max_dim[1]))
    
    for i in tqdm.tqdm(range(n_imgs)):
        imgs = imread(flist_in[i])
        if len(imgs.shape) == 2:
            imgs = np.expand_dims(imgs,0)
        if imgs.shape[-1] == np.min(imgs.shape):
            imgs = np.moveaxis(imgs, -1, 0)
        _slice = props['slice'][i]
        center = [int((_slice[0].stop+_slice[0].start)/2),
                  int((_slice[1].stop+_slice[1].start)/2)]

        # pad the image if the largest mask doesn't fit
        if int((center[0]-max_dim[0]/2))<0:
            w = np.abs(int((center[0]-max_dim[0]/2)))
            imgs = np.stack([np.pad(a,((w,0),(0,0)),mode='constant') for a in imgs])
            center[0] = center[0] + w
        if int((center[1]-max_dim[1]/2))<0:
            w = np.abs(int((center[1]-max_dim[1]/2)))
            imgs = np.stack([np.pad(a,((0,0),(w,0)),mode='constant') for a in imgs])
            center[1] = center[1] + w
        if int((center[0]+max_dim[0]/2))>imgs[0].shape[0]:
            w = np.abs(int((center[0]+max_dim[0]/2-imgs[0].shape[0])))
            imgs = np.stack([np.pad(a,((0,w+10),(0,0)),mode='constant') for a in imgs])
        if int((center[1]+max_dim[1]/2))>imgs[0].shape[1]:
            w = np.abs(int((center[1]+max_dim[1]/2-imgs[0].shape[1])))
            imgs = np.stack([np.pad(a,((0,0),(0,w+10)),mode='constant') for a in imgs])

        _slice_large = [slice(int((center[0]-max_dim[0]/2)),int((center[0]+max_dim[0]/2))),
                        slice(int((center[1]-max_dim[1]/2)),int((center[1]+max_dim[1]/2)))]

        img = np.stack([ a[_slice_large[0], _slice_large[1]] for a in imgs ])
        
        movie[i,0,...] = img

        
    grays = np.tile(np.arange(256, dtype='uint8'), (3, 1))
    green = np.zeros((3, 256), dtype='uint8')
    green[1] = np.arange(256, dtype='uint8')
    # red = np.zeros((3, 256), dtype='uint8')
    # red[0] = np.arange(256, dtype='uint8')
    ijtags = compositeImageJ.imagej_metadata_tags({'LUTs': [grays, green]}, '>')
    imsave(os.path.join(segment_folder,cond+'_composite_recap.tif'),movie.astype(np.uint16), byteorder='>', imagej=True,
                        metadata={'mode': 'composite'}, extratags=ijtags)

