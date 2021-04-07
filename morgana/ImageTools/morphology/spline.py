import numpy as np
from scipy import interpolate
from skimage.morphology import binary_dilation, disk
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=17)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)

def compute_spline_coeff(ma, bf, anch, visualize=False):
    # print('Computing spline')
    import scipy.interpolate as interpolate

    # print('interp')
    t = [0]
    for i in range(anch.shape[0]-1):
        dist = np.sqrt(np.sum((anch[i+1]-anch[i])**2))
        t.append(t[-1]+dist)
    t = np.array(t)*(len(t)-1)/t[-1]

    # t = np.arange(anch.shape[0]) # this is wrong! because the anchor points are not equally spaced!
    k = 3
    if anch.shape[0]<=4:
        k = 2
    tck,_ = interpolate.splprep([anch[:,0],  anch[:,1]], k=k, s=k*500000)

    l = []
    Ns = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,50,75,100]
    # print('eval splines')

    if visualize:
        fig, ax = plt.subplots()
        ax.imshow(bf, cmap='gray')
        ax.contour(binary_dilation(ma, disk(1)),[0.5],colors='w', alpha=.5)
        ax.plot(anch[:,1], anch[:,0], 'oy')
        ax.axis('off')

        from cycler import cycler
        colors = [plt.cm.rainbow(i) for i in np.linspace(0, 0.5, len(Ns))]
        ax.set_prop_cycle(cycler('color', colors))

    for N in Ns:
        new_t = np.linspace(0, 1, N)
        x, y = interpolate.splev(new_t, tck)

        if visualize:        
            ax.plot(y, x)

        l.append(findLength(x,y))

    if visualize:        
        fig, ax = plt.subplots()
        ax.plot(Ns,l)
        ax.set_xlabel('N points')
        ax.set_ylabel('Spline length')

    N_points = int(l[-1])
    return N_points, tck

def findLength(xs,ys):
    xVal = np.array(xs)
    yVal = np.array(ys)
    length = np.sum( np.sqrt(np.diff(xVal)**2 + np.diff(yVal)**2 ) )
    return length
