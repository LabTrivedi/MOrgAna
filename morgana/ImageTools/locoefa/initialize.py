import pandas as pd
import numpy as np
from skimage.morphology import binary_dilation, disk, medial_axis
from skimage.measure import find_contours


def read_example_data(fname, N_modes=50):

    # read points
    contour = pd.read_csv(fname)
    if contour.x.values[-1] != contour.x.values[0]:
#        print('Appending first element to make it a close curve')
        newrow = pd.Series({'x': contour.x[0], 'y': contour.y[0]})
        contour.append(newrow)

    initialize_values = [0. for i in range(len(contour.x))]
    variables = ['deltax',  'deltay',       'deltat',       't',
                'xi',       'sumdeltaxj',   'sumdeltayj',   'epsilon']

    for variable in variables:
        contour[variable] = initialize_values

    mode = initialize_mode(N_modes)

    return contour, mode

def initialize_mode(N_modes_original=50):
    # construct the mode dictionary
    N_modes = N_modes_original+2

    variables = ['alpha',           'beta',             'gamma',            'delta',
                'tau',              'alphaprime',       'gammaprime',       'rho',
                'alphastar',        'betastar',         'gammastar',        'deltastar', 
                'r',                'a',                'b',                'c',
                'd',                'aprime',           'bprime',           'cprime',
                'dprime',           'phi',              'theta',            'lambda1',          
                'lambda2',          'lambda21',         'lambda12',         'lambdaplus',       
                'lambdaminus',      'zetaplus',         'zetaminus',        'locooffseta',
                'locooffsetc',      'locolambdaplus',   'locolambdaminus',   'locozetaplus',     
                'locozetaminus',    'locoL',            'locoaplus',        'locobplus', 
                'lococplus',        'locodplus',        'locoaminus',       'locobminus', 
                'lococminus',       'locodminus']
    initialize_values = [0. for i in range(N_modes)]

    mode = pd.DataFrame(dict(zip(variables, [initialize_values]*len(variables))))
    
    return mode

def get_edge_points(mask, N_modes=50):
#    print('Extracting edge points from mask...')

    # make sure there is a edge
    mask[0,:] = mask[-1,:] = mask[:,0] = mask[:,-1] = 0

    # find points on the edge
    points = find_contours(mask,0.)[0]
    contour = pd.DataFrame({
        'x': points[:,0],
        'y': points[:,1]
    })

    initialize_values = [0. for i in range(len(contour.x))]
    variables = ['deltax',  'deltay',       'deltat',       't',
                'xi',       'sumdeltaxj',   'sumdeltayj',   'epsilon']

    for variable in variables:
        contour[variable] = initialize_values

    mode = initialize_mode(N_modes_original=N_modes)

    return contour, mode
