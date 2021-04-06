import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage import label

def compute_profiles_fluo(fl, ma, m, visualize=False):

    fl_straight = np.reshape(map_coordinates(fl,np.reshape(m,(m.shape[0]*m.shape[1],2)).T,order=0,mode='constant',cval=0).T,(m.shape[0],m.shape[1]))
    ma_straight = np.reshape(map_coordinates(ma,np.reshape(m,(m.shape[0]*m.shape[1],2)).T,order=0,mode='constant',cval=0).T,(m.shape[0],m.shape[1]))
    
    # keep only larger object
    labeled_mask, _ = label(ma_straight)
    ma_straight = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))

    # compute AP profile
    apProf = np.sum(fl_straight*ma_straight,1)/np.sum(ma_straight,1)

    # compute LR profile
    col_idx = np.sum(ma_straight,0)>0
    lrProf = np.sum(fl_straight[:,col_idx]*ma_straight[:,col_idx],0)/np.sum(ma_straight[:,col_idx],0)

    # compute radial profile
    fl = fl_straight[:,col_idx]
    ma = ma_straight[:,col_idx].astype(np.uint16)
    radProf = np.array([0. for i in range(int(np.sqrt(fl.shape[0]**2+fl.shape[1]**2)))])
    radCount = np.array([0. for i in range(int(np.sqrt(fl.shape[0]**2+fl.shape[1]**2)))])
    angProf = np.array([0. for i in range(360)])
    angCount = np.array([0. for i in range(360)])
    idxs = np.where(ma>0)
    cx,cy = fl.shape[0]/2,fl.shape[1]/2

    dists = (np.sqrt((idxs[0]-cx)**2+(idxs[1]-cy)**2)).astype(int)
    angs = (np.mod(360+np.arctan2(idxs[1]-cy,idxs[0]-cx)*180/np.pi,360)).astype(int)
    vals = val = fl[idxs[0],idxs[1]]
    for dist, ang, val in zip(dists, angs, vals):
        radProf[dist] += val
        radCount[dist] += 1.
        angProf[ang] += val
        angCount[ang] += 1.
    
    radProf = radProf/radCount
    radProf = radProf[np.logical_not(np.isnan(radProf))]

    angProf = angProf/angCount
    angProf = angProf[np.logical_not(np.isnan(angProf))]

    # plt.figure()
    # plt.plot(radProf)
    # plt.figure()
    # plt.plot(angProf)
    # plt.show()
    # plt.pause(5)

    # radMask = np.zeros(ma.shape).astype(np.uint16)
    # radMask[int(radMask.shape[0]/2),int(radMask.shape[1]/2)] = 1
    # radProf = [np.mean((fl*radMask)[radMask>0])]
    # i=1
    # while np.sum(ma-radMask)>0:
    #     newrad = morphology.binary_dilation(radMask,disk(i))
    #     edge = np.logical_xor(newrad,radMask)
    #     radProf.append(np.mean((fl*edge)[edge>0]))
    #     radMask = newrad
    #     i+=1
    #     # plt.figure()
    #     # plt.imshow(radMask,cmap='gray')
    #     # plt.show()
    #     # plt.pause(10)

    # convert all outputs to list
    apProf = list(apProf) 
    lrProf = list(lrProf)
    radProf = list(radProf)
    angProf = list(angProf)
    return apProf, lrProf, radProf, angProf
