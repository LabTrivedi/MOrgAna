import numpy as np
from scipy import interpolate

def compute_midline_and_tangent(anch, N_points, tck, width):
    # print('Generating meshgrid')
    t = np.linspace(0, 1, N_points)
    x, y = interpolate.splev(t, tck)
    # y = interpolate.splev(t, y_tup)
    dx, dy = interpolate.splev(t, tck, der=1)
    # dy = interpolate.splev(t, y_tup, der=1)

    lengths = []
    for i in range(len(dx)):
        lengths.append(np.sqrt(dx[i]**2+dy[i]**2)) 
    dx = dx/lengths
    dy = dy/lengths

    midline = np.stack([x,y]).transpose()
    tangent = np.stack([dx,dy]).transpose()

    return midline, tangent, width
