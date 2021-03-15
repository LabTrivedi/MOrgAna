import pandas as pd
import os

def save_straight_morpho_params( save_folder, cond, props ):

    # Make it work for Python 2+3 and with Unicode
    # print('### Done computing, saving data...')
    data = pd.DataFrame(props)
    fname = os.path.join(save_folder,cond+'_morpho_straight_params.json')
    data = data.to_json(fname,
                    indent=4,
                    orient='records'
                    )

def load_straight_morpho_params( save_folder, cond ):
    fname = os.path.join(save_folder,cond+'_morpho_straight_params.json')
    data = pd.read_json(fname,orient='records')

    return data

