import os, tqdm
import numpy as np
import matplotlib as mpl
from textwrap import wrap
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib import rc
rc('pdf', fonttype=42)
from itertools import repeat
import multiprocessing

from morgana.DatasetTools import io
# from orgseg import ImageTools

def generate_overview(input_folder, saveFig=True, fileName='', start=None, stop=None, downshape=1):
    print('Generating recap image at',input_folder)

    flist_in = io.get_image_list(input_folder)
    segment_folder = os.path.join(input_folder,'result_segmentation')
    flist_ws = io.get_image_list(segment_folder, '_watershed.tif', 'include')
    flist_cl = io.get_image_list(segment_folder, '_classifier.tif', 'include')

    if start==None: start=0
    if stop==None: stop=len(flist_in)
    flist_in = flist_in[start:stop]
    flist_ws = flist_ws[start:stop]
    flist_cl = flist_cl[start:stop]

    n_img = len(flist_in)
    ncols = 5
    nrows = (n_img-1)//5+1

    # ### multiprocess
    # file_names = [[flist_in[i],flist_cl[i],flist_ws[i]] for i in range(n_img)]

    # N_cores = np.clip( int(0.8 * multiprocessing.cpu_count()),1,None )

    # pool = multiprocessing.Pool(N_cores)
    # data_list = list(   tqdm.tqdm(
    #                         pool.istarmap(
    #                             ImageTools.io.load_images_ch0, 
    #                             zip(    file_names, 
    #                                     repeat( downshape ) ) ), 
    #                             total = n_img ) )
    # imgs = [data[0] for data in data_list]
    # classifiers = [data[1] for data in data_list]
    # watersheds = [data[2] for data in data_list]

    ### normal for loop
    imgs = [0. for i in range(n_img)]
    classifiers = [0. for i in range(n_img)]
    watersheds = [0. for i in range(n_img)]
    for i in tqdm.tqdm(range(n_img)):
        img = imread(flist_in[i]).astype(float)
        if img.ndim == 2:
            img = np.expand_dims(img,0)
        if img.shape[-1] == np.min(img.shape):
            img = np.moveaxis(img, -1, 0)
        imgs[i] = img[0,::downshape,::downshape]
        classifiers[i] = imread(flist_cl[i])[::downshape,::downshape].astype(float)
        watersheds[i] = imread(flist_ws[i])[::downshape,::downshape].astype(float)

    ### plotting
    fig,ax = plt.subplots(figsize=(3*ncols,3*nrows), nrows=nrows, ncols=ncols)
    ax = ax.flatten()

    for i in tqdm.tqdm(range(n_img)):
        
        _, filename = os.path.split(flist_in[i])
        filename, _ = os.path.splitext(filename)

        ax[i].imshow(imgs[i], 'gray', interpolation='none', vmin=np.percentile(img,1.), vmax=np.percentile(img,99.))
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['black','red'],256)
        ax[i].imshow(classifiers[i], cmap=cmap, interpolation='none',alpha=.4)
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['black','aqua'],256)
        ax[i].imshow(watersheds[i], cmap=cmap, interpolation='none',alpha=.3)

        ax[i].set_title("\n".join(wrap(filename, 20)),fontsize=8)

    for a in ax:
        a.axis('off')
    for j in range(i+1,len(ax)):
        ax[j].remove()
        
    # plt.show()

    if saveFig:
        print('Saving image...')
        # save figure
        _, cond = os.path.split(input_folder)
        print(fileName)
        if fileName == '':
            fileName = os.path.join(input_folder, 'result_segmentation', cond+'_recap_classifier.png')
        fig.savefig(fileName, dpi=300)
        print('Done saving!')

    return fig
