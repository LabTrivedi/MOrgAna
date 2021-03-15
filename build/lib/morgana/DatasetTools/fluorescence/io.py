import pandas as pd
import numpy as np
import os
import json

def save_fluo_info( save_folder, cond, props ):

    # Make it work for Python 2+3 and with Unicode
    # print('### Done computing, saving data...')
    # data = pd.DataFrame(props)
    fname = os.path.join(save_folder,cond+'_fluo_intensity.json')
    props = props.to_json(fname,
                    indent=4,
                    orient='records'
                    )

def load_fluo_info( save_folder, cond ):
    fname = os.path.join(save_folder,cond+'_fluo_intensity.json')
    data = pd.read_json(fname,orient='records')

    return data
