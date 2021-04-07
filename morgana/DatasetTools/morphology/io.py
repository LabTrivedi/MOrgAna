import pandas as pd
import numpy as np
import os

def save_morpho_params( save_folder, cond, props ):

    # Make it work for Python 2+3 and with Unicode
    # print('### Done computing, saving data...')
    # data = pd.DataFrame(props)
    fname = os.path.join(save_folder,cond+'_morpho_params.json')
    props = props.to_json(
                    fname,
                    indent=4,
                    orient='records'
                    )

def load_morpho_params( save_folder, cond ):

    fname = os.path.join(save_folder,cond+'_morpho_params.json')
    data = pd.read_json(fname,orient='records')

    # convert the slice entry into a python slice object
    for i in range(len(data.slice)):
        data.at[i,'slice'] = tuple([slice(j['start'],j['stop']) for j in data.slice[i]])

    # covert other entries into numpy arrays (instead of lists)
    data.meshgrid = data.meshgrid.astype('object')
    for i in range(len(data.meshgrid)):
        if np.isnan(data.meshgrid[i]):
            data.at[i,'meshgrid'] = None

        data.at[i,'centroid'] = np.array(data.centroid[i])
        data.at[i,'anchor_points_midline'] = np.array(data.anchor_points_midline[i])
        data.at[i,'midline'] = np.array(data.midline[i])
        data.at[i,'tangent'] = np.array(data.tangent[i])

    return data
