import numpy as np

def compute_meshgrid(mid,tan,width):
    width=int(width)
    # print(mid.shape,tan.shape)
    ort = np.stack([tan[:,1],-tan[:,0]]).transpose()
    
    meshgrid = np.zeros((mid.shape[0],2*width+1,2))
    i = 0
    for m,o in zip(mid, ort):
        k = 0
        for w in np.arange(-width,+width+1):
            meshgrid[i,k,:] = m+w*o
            k+=1
        i += 1
    return np.array(meshgrid).astype(int)

def visualize_meshgrid(midline, tangent, meshgrid, img, ma=None, color='lime', N_step=50, ax=None, show_tangent=False):
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    cmap = LinearSegmentedColormap.from_list('mycmap', ['black', color])
    
    if not ax:
        fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(img, cmap=cmap, interpolation='none', vmin=np.percentile(img,1.), vmax=np.percentile(img,99.))
    if ma:
        ax.imshow(ma, 'red', interpolation='none', alpha=.3)
    ax.plot(midline[:,1], midline[:,0], 'w', lw=1.)
    if show_tangent:
        ax.quiver(midline[::N_step,1], midline[::N_step,0], tangent[::N_step,1], tangent[::N_step,0], width=.005, color='r')
        ax.quiver(midline[::N_step,1], midline[::N_step,0], tangent[::N_step,0], -tangent[::N_step,1], width=.005, color='r')
    ax.invert_yaxis()

    for mp in meshgrid[::20]:
        ax.plot(mp[::20,1].flatten(),mp[::20,0].flatten(),'-ow',lw=.1,ms=.2)
    ax.axis('off')
    # ax.set_xlim(0,550)
    # ax.set_ylim(0,550)

