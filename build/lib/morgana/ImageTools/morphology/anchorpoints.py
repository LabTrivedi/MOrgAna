import numpy as np
import os
from scipy import interpolate
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_dilation, disk, medial_axis
from skimage import transform
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=17)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)

def compute_anchor_points(ma,_slice,down_shape,smoothing=1000):
    ### remove edge pixel from mask to take into account objects on the edge
    ma[-1,:] = 0
    ma[0,:] = 0
    ma[:,-1] = 0
    ma[:,0] = 0

    ### resize image to make faster computation
    ma_down = transform.resize(ma.astype(float), (int(ma.shape[0]*down_shape),int(ma.shape[1]*down_shape)), order=0, preserve_range=True)
    ma_down[-1,:] = 0
    ma_down[0,:] = 0
    ma_down[:,-1] = 0
    ma_down[:,0] = 0

    ### find contour
    points = find_contours(ma_down,0.)[0]
    # store x and y of the edge point for the spline computation
    x = points[:,0]
    y = points[:,1]
    # append the starting x,y coordinates for periodic condition
    if (x[-1]!=x[0]) and (y[-1]!=y[0]):
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

    ### find midline with the iterative process until only one branch is found
    # at every step, increase the smoothness of the spline by a factor of 1.2
    _quit = False
    while not _quit:
       # print(smoothing)

        # fit splines to f(t)=(x,y), treating as periodic. also note that s increases every time
        # to force larger smoothing of the spline fit at every iteration.
        tck,_ = interpolate.splprep([x, y], s=smoothing, per=True)
        # evaluate the spline fits for 1000 evenly spaced distance values
        xi, yi = interpolate.splev(np.linspace(0, 1, 1000*len(x)), tck)
        xi = np.clip(xi,2,ma_down.shape[0]-2)
        yi = np.clip(yi,2,ma_down.shape[1]-2)

        # create new mask with spline curve
        n = 1.
        mask = np.zeros((int(n*ma_down.shape[0]),int(n*ma_down.shape[1]))).astype(np.uint16)
        for x1,y1 in zip(xi,yi):
            mask[int(n*x1),int(n*y1)]=1
        mask = binary_fill_holes(mask)

        # find medial axis of new mask
        midlineMask,_ = medial_axis(mask, return_distance=True)

        # find coordinates and check how many end points there are
        midlinePoints = np.where(midlineMask)
        midlinePoints = np.array([midlinePoints[0],midlinePoints[1]])
        midlinePoints = np.transpose(midlinePoints)
        conn = np.zeros((3,3))+1
        key_points = []
        for i, p in enumerate(midlinePoints):
            connectivity = np.sum(midlineMask[p[0]-1:p[0]+2,p[1]-1:p[1]+2]*conn)
            if connectivity==2:
                key_points.append([p,'end',i])
            if connectivity>3:
                key_points.append([p,'branching',i])
        
        # if the condition is met, quit
        # else, increase the smoothing and repeat
        if len(key_points)==2:
            _quit = True
        else:
            smoothing = smoothing * 1.5

        # # plot the result
        # print(_quit,smoothing)
        # fig, ax = plt.subplots(1, 3)
        # # ax.plot(y, x, 'or')
        # ax[0].plot(yi, xi, '-w')
        # ax[0].imshow(ma_down)
        # ax[1].imshow(mask)
        # ax[2].imshow(midlineMask)
        # for p in key_points:
        #     ax[1].plot(p[0][1],p[0][0],'or')
        #     ax[2].plot(p[0][1],p[0][0],'or')
        # plt.show(block=False)
        # plt.pause(5)
        # plt.close(fig)

    ### rescale midline coordinates back to original shape
    midlinePoints = (midlinePoints/down_shape).astype(np.float)
    if midlinePoints.shape[0]==1:
        # midline = np.where(a)
        midlinePoints = np.array([[midlinePoints[0][0]-1,midlinePoints[0][1]+1],[midlinePoints[1][0]-1,midlinePoints[1][1]+1]])
        midlinePoints = np.transpose(midlinePoints)
    key_points = (np.array([k[0] for k in key_points])/down_shape).astype(np.float)

    ### order midline points from one end to another
    points = [np.array(key_points[0])]
    dist = [np.sqrt(np.sum((points[-1]-i)**2)) for i in midlinePoints]
    idx = np.where(dist==np.min(dist))[0]
    remaining = np.delete(midlinePoints,idx,0)
    # print(midline,remaining)
    while remaining.shape[0]>0:
        dist = [np.sqrt(np.sum((points[-1]-i)**2)) for i in remaining]
        idx = np.where(dist==np.min(dist))[0]
        points.append(remaining[idx][0])
        remaining = np.delete(remaining,idx,0)
    anchors = np.array(points).astype(np.float)

    ### find edge point to the left
    tg = np.array([0,0]).astype(np.float)
    n = np.clip(5,0,anchors.shape[0])
    for i in range(1,n):
        tg += (anchors[0]-anchors[i])/np.sqrt(np.sum((anchors[0]-anchors[i])**2))
    tg = tg/5
    tg = tg/np.sqrt(np.sum(tg**2))
    edge_point_L = anchors[0]
    # print(anchors)
    # print(tg)
    while ma[int(edge_point_L[0]),int(edge_point_L[1])]:
        edge_point_L = edge_point_L + tg

    ### find edge point to the right
    tg = np.array([0,0]).astype(np.float)
    for i in range(1,n):
        tg += (anchors[::-1][0]-anchors[::-1][i])/(np.sqrt(np.sum((anchors[::-1][0]-anchors[::-1][i])**2)))
    tg = tg/5
    tg = tg/np.sqrt(np.sum(tg**2))
    edge_point_R = anchors[-1]
    while ma[int(edge_point_R[0]),int(edge_point_R[1])]:
        edge_point_R = edge_point_R + tg

    ### update anchor points with reasonable spacing to avoid overfitting of the spline curve
    edge_dist = np.max([
        np.sqrt(np.sum((edge_point_L-anchors[0])**2)),
        np.sqrt(np.sum((edge_point_R-anchors[-1])**2))
        ])
    # print(edge_point_L)
    # print(edge_point_R)
    # print(edge_dist)
    # print(down_shape)
    # print(int(edge_dist*down_shape/2))
    # print(anchors.shape)
    s = np.max([int(edge_dist*down_shape/2),1])
    anch = np.concatenate((np.array([edge_point_L]),anchors[::s],np.array([edge_point_R])), axis=0).astype(np.float)

    # # plot the result
    # fig, ax = plt.subplots(1, 1)
    # # ax.plot(y, x, 'or')
    # ax.imshow(ma)
    # ax.plot(yi/down_shape, xi/down_shape, '-g')
    # # ax[1].imshow(transform.resize(mask.astype(float), ma.shape, order=0, preserve_range=True))
    # # ax[2].imshow(transform.resize(a.astype(float), ma.shape, order=0, preserve_range=True))
    # for p in anch:
    #     ax.plot(p[1],p[0],'or')
    # plt.pause(5)
    # plt.close(fig)

    ### reset values to cropped image size
    # print(anch.shape)
    anch[:,0] = anch[:,0]-_slice[0].start
    anch[:,1] = anch[:,1]-_slice[1].start
    anch[:,0] = np.clip(anch[:,0],0,_slice[0].stop-_slice[0].start-1)
    anch[:,1] = np.clip(anch[:,1],0,_slice[1].stop-_slice[1].start-1)
    anch = anch.astype(np.uint16)

    ### remove identical points (don't know why they happen)
    _, idx = np.unique(anch,axis=0,return_index=True)
    anch = anch[np.sort(idx)]

    # # plot the result
    # fig, ax = plt.subplots(1, 1)
    # # ax.plot(y, x, 'or')
    # # ax.plot(yi/down_shape, xi/down_shape, '-w')
    # ax.imshow(ma[_slice])
    # # ax[1].imshow(transform.resize(mask.astype(float), ma.shape, order=0, preserve_range=True))
    # # ax[2].imshow(transform.resize(a.astype(float), ma.shape, order=0, preserve_range=True))
    # # for p in anch:
    # ax.plot(anch[:,1],anch[:,0],'-or')
    # # ax.plot(anchors[:,1],anchors[:,0],'-og',lw=.5,alpha=.2)
    # plt.pause(5)
    # plt.close(fig)

    return anch.astype(np.uint16)
