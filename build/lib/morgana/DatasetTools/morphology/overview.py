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

#########################################################################################################################

def generate_overview_finalMask(input_folder, chosen, saveFig=True, downshape=1, autoclose=False):
    print('### Generating recap image at',input_folder)

    flist_in = io.get_image_list(input_folder)
    flist_in = [flist_in[i] for i in range(len(flist_in)) if chosen[i]]
    segment_folder = os.path.join(input_folder,'result_segmentation')
    flist_ma = io.get_image_list(segment_folder, '_finalMask.tif', 'include')
    
    n_img = len(flist_in)
    ncols = 5
    nrows = (n_img-1)//5+1

    fig,ax = plt.subplots(figsize=(3*ncols,3*nrows), nrows=nrows, ncols=ncols)
    ax = ax.flatten()

    # ### multiprocess
    # file_names = [[flist_in[i],flist_ma[i]] for i in range(n_img)]
    # N_cores = np.clip( int(0.8 * multiprocessing.cpu_count()),1,None )

    # pool = multiprocessing.Pool(N_cores)
    # data_list = list(   tqdm.tqdm(
    #                         pool.istarmap(
    #                             ImageTools.io.load_images_ch0, 
    #                             zip(    flist_in, 
    #                                     flist_ma,
    #                                     repeat( downshape ) ) ), 
    #                             total = n_img ) )
    # imgs = [data[0] for data in data_list]
    # masks = [data[1] for data in data_list]

    ### normal for loop
    imgs = [0. for i in range(n_img)] 
    masks = [0. for i in range(n_img)]
    for i in tqdm.tqdm(range(n_img)):        
        img = imread(flist_in[i]).astype(float)
        if len(img.shape) == 2:
            img = np.expand_dims(img,0)
        if img.shape[-1] == np.min(img.shape):
            img = np.moveaxis(img, -1, 0)
        imgs[i] = img[0,::downshape,::downshape]
        masks[i] = imread(flist_ma[i])[::downshape,::downshape].astype(float)

    ### plotting
    for i in tqdm.tqdm(range(n_img)):
        
        _, filename = os.path.split(flist_in[i])
        filename, _ = os.path.splitext(filename)
        
        ax[i].imshow(imgs[i], 'gray', interpolation='none', vmin=np.percentile(img,1.), vmax=np.percentile(img,99.))
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['black','aqua'],256)
        ax[i].imshow(masks[i], cmap=cmap, interpolation='none',alpha=.3)

        ax[i].set_title(("\n".join(wrap(filename, 20))),fontsize=6)
        
    for a in ax:
        a.axis('off')
    for j in range(i+1,len(ax)):
        ax[j].remove()
        
    # plt.show()
    
    if autoclose:
        plt.pause(10)
        plt.close()
        

    if saveFig:
        print('### Saving image...')
        # save figure
        _, cond = os.path.split(input_folder)
        fig.savefig(os.path.join(input_folder, 'result_segmentation', cond+'_finalMasks.png'), dpi=75)
        print('### Done saving!')

    return fig

#########################################################################################################################

def generate_composite_movie_cropped(input_folder):
    print('### Generating recap composite movie at',input_folder)
    _, cond = os.path.split(input_folder)
    segment_folder = os.path.join(input_folder,'result_segmentation')
    
    file_extension = '_morpho_params.json'
    fname = os.path.join(segment_folder,cond+file_extension)
    if not os.path.exists(fname):
        props = computemorphology.compute_morphological_info(input_folder)
        ioMorph.save_morpho_params( segment_folder, cond, props )
    else:
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

        _slice_large = [slice(int((center[0]-max_dim[0]/2)),int((center[0]+max_dim[0]/2)),None),
                        slice(int((center[1]-max_dim[1]/2)),int((center[1]+max_dim[1]/2)),None)]

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

        _slice_large = [slice(int((center[0]-max_dim[0]/2)),int((center[0]+max_dim[0]/2)),None),
                        slice(int((center[1]-max_dim[1]/2)),int((center[1]+max_dim[1]/2)),None)]


        img = np.stack([ a[_slice_large] for a in imgs ])
        
        movie[i,0,...] = img

        
    grays = np.tile(np.arange(256, dtype='uint8'), (3, 1))
    green = np.zeros((3, 256), dtype='uint8')
    green[1] = np.arange(256, dtype='uint8')
    # red = np.zeros((3, 256), dtype='uint8')
    # red[0] = np.arange(256, dtype='uint8')
    ijtags = compositeImageJ.imagej_metadata_tags({'LUTs': [grays, green]}, '>')
    imsave(os.path.join(segment_folder,cond+'_composite_recap.tif'),movie.astype(np.uint16), byteorder='>', imagej=True,
                        metadata={'mode': 'composite'}, extratags=ijtags)
        
def generate_composite_img_cropped(input_folder, downshape=1, keep_open=True):
    print('### Generating recap composite image at',input_folder)
    _, cond = os.path.split(input_folder)
    segment_folder = os.path.join(input_folder,'result_segmentation')

    file_extension = '_morpho_params.json'
    fname = os.path.join(segment_folder,cond+file_extension)
    if not os.path.exists(fname):
        props = computemorphology.compute_morphological_info(input_folder)
        ioMorph.save_morpho_params( segment_folder, cond, props )
    else:
        props = ioMorph.load_morpho_params(segment_folder, cond)

    flist_in = [ os.path.join(input_folder, i) for i in props['input_file'] ] 
    # flist_mask = [ os.path.join(input_folder, i) for i in props['mask_file'] ]
    
    n_img = len(flist_in)
    img = imread(flist_in[0])
    if len(img.shape) == 2:
        img = np.expand_dims(img,0)
    if img.shape[-1] == np.min(img.shape):
        img = np.moveaxis(img, -1, 0)
    n_col = img.shape[0]+1

    fig,ax = plt.subplots(figsize=(n_col,n_img), nrows=n_img, ncols=n_col)
    plt.subplots_adjust(top=0.99,left=0.01,right=0.99,bottom=0.01)
    # ax = ax.flatten()
    # print(n_img,len(ax))

    for i in tqdm.tqdm(range(n_img)):
        imgs = imread(flist_in[i])
        if len(imgs.shape) == 2:
            imgs = np.expand_dims(imgs,0)
        if imgs.shape[-1] == np.min(imgs.shape):
            imgs = np.moveaxis(imgs, -1, 0)
        _slice = props['slice'][i]
        imgs = np.stack([ a[_slice][::downshape,::downshape] for a in imgs ])

        cmaps = [
                mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['black',i],256) for i in ['white','lime','red','aqua','magenta','green','yellow','blue']
                ]
        alphas = [.5]*10
        alphas[0] = 1.

        for j in range(len(imgs)):
            ax[i,-1].imshow(imgs[j], cmap=cmaps[j], interpolation='none', vmin=np.percentile(imgs[j],1.), vmax=np.percentile(imgs[j],99.),alpha=alphas[j])
            ax[i,j].imshow(imgs[j], cmap=cmaps[j], interpolation='none', vmin=np.percentile(imgs[j],1.), vmax=np.percentile(imgs[j],99.),alpha=1)
            
        name = os.path.split(flist_in[i])[-1]
        ax[i,int(n_col/2)].set_title(("\n".join(wrap(name, 40))),fontsize=6)
        
    for a in ax.flatten():
        a.axis('off')
        
    fig.show()

    print('### Saving image...')
    # save figure
    _, cond = os.path.split(input_folder)
    fig.savefig(os.path.join(segment_folder,cond+'_composite_recap.png'), dpi=75)
    if not keep_open:
        plt.close(fig)
    print('### Done saving!')

def createCompositeOverview(folder, keep_open=True, create_tif=True):
    if create_tif:
        generate_composite_movie_cropped(folder)
    generate_composite_img_cropped(folder, keep_open=keep_open)

##########################################################################################################################

def generate_meshgrid_img_cropped(input_folder, keep_open = True):
    print('### Generating recap meshgrid image at',input_folder)
    _, cond = os.path.split(input_folder)
    segment_folder = os.path.join(input_folder,'result_segmentation')

    file_extension = '_morpho_params.json'
    fname = os.path.join(segment_folder,cond+file_extension)
    if not os.path.exists(fname):
        props = computemorphology.compute_morphological_info(input_folder)
        ioMorph.save_morpho_params( segment_folder, cond, props )
    else:
        props = ioMorph.load_morpho_params(segment_folder, cond)

    flist_in = [ os.path.join(input_folder, i) for i in props['input_file'] ] 
    flist_ma = [ os.path.join(input_folder, i) for i in props['mask_file'] ]
    
    n_img = len(flist_in)
    ncols = 5
    nrows = (n_img-1)//5+1

    fig,ax = plt.subplots(figsize=(3*ncols,3*nrows), nrows=nrows, ncols=ncols)
    plt.subplots_adjust(top=0.95,left=0.05,right=0.95,bottom=0.05,hspace=0.01,wspace=0.01)
    ax = ax.flatten()

    for i in tqdm.tqdm(range(n_img)):
        prop = {key: props[key][i] for key in props}

        tangent = prop['tangent']
        midline = prop['midline']
        width = prop['meshgrid_width']
        mesh = prop['meshgrid']
        if not mesh:
            mesh = meshgrid.compute_meshgrid(midline,tangent,width)
        anch = prop['anchor_points_midline']

        bf = imread(flist_in[i])
        if len(bf.shape) == 2:
            bf = np.expand_dims(bf,0)
        if bf.shape[-1] == np.min(bf.shape):
            bf = np.moveaxis(bf,-1,0)
        bf = bf[0][prop['slice']]
        ma = img_as_bool( imread(flist_ma[i])[prop['slice']].astype(np.float) )

        meshgrid.visualize_meshgrid(midline,tangent,mesh,bf,color='white', ax=ax[i])
        
        ax[i].contour(ma, [0.5], colors='r', alpha=.5)
        ax[i].plot(anch[:,1], anch[:,0], '-or', lw=.5, ms=.5, alpha=.5)

        name = os.path.split(flist_in[i])[-1]
        ax[i].set_title(("\n".join(wrap(name, 20))),fontsize=6)

    for a in ax:
        a.axis('off')
    for j in range(i+1,len(ax)):
        ax[j].remove()

    fig.show()
        
    print('### Saving image...')
    # save figure
    fig.savefig(os.path.join(segment_folder, cond+'_meshgrid_recap.png'), dpi=75)
    if not keep_open:
        plt.close(fig)
    print('### Done saving!')

def createMeshgridOverview(input_folder, keep_open = True):
    generate_meshgrid_img_cropped(input_folder, keep_open=keep_open)

##########################################################################################################################

