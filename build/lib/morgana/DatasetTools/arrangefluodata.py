import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join('..'))

from morgana.DatasetTools.fluorescence import computefluorescence, io

def collect_fluo_data(groups, channel, distType, isTimelapse=False):

    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    # collect the data for every group
    N_groups = len(groups)
    data_all = [pd.DataFrame({}) for i in range(N_groups)]

    for i in range(N_groups):
        # extract table in the group
        folders = groups[i]
        N_folders = len(folders)
        
        # extract folders (dataset) in the table
        for j in range(N_folders):
            input_folder = folders[j]
            _, cond = os.path.split(input_folder)
            save_folder = os.path.join(input_folder,'result_segmentation')
            fname = os.path.join(save_folder,cond+'_fluo_intensity.json')

            # if morpho_params of straighten image not created yet, compute and save
            if not os.path.exists(fname):
                data = computefluorescence.compute_fluorescence_info(input_folder)
                io.save_fluo_info( save_folder, cond, data )
            else:
                # else, just load it
                data = io.load_fluo_info( save_folder, cond  )
                # if fluo info was created long time ago and doesn't have all parameters, create it again
                if 'ch%d_'%channel+distType not in data.keys():
                    data = computefluorescence.compute_fluorescence_info(input_folder)
                    io.save_fluo_info( save_folder, cond, data )

            # select for the right channel
            if not any( 'ch%d_'%channel in k for k in data.keys() ):
                # if there is no such a channel, quit
                return None

            # compile data in the right format: i.e., always such that the object to return is
            # a a list (groups) of dataframes (one gastruloid per row):
            # [
            #       data
            # g1    val
            # g2    val ;
            #
            #       data
            # g1    val
            # g2    val ;
            # 
            # ...
            # ]
            #
            # note: the ... can be:
            # i. a single number (area in non timelapse mode), 
            # ii. a list (1D object, area in timelapse mode or AP fluorescence profile in non timelapse mode),
            # iii. a list of lists (2D object, AP fluorescence profile in non timelapse mode)
            #

            # print(data.Average,data.Background)
            # select channels needed

            # filter for needed info
            keys = [ 'ch%d_'%channel+distType, 'ch%d_'%channel+'Background' ]
            data = data[keys]
            if not 'ch%d_'%channel+'Background' in data.keys():
                data['ch%d_'%channel+'Background'] = 0.
            # print(data)

            if isTimelapse:
                # if this is a timelapse dataset, all data should be stored in the same object
                rows = pd.Series({ key: list(data[key].values) for key in keys })
            else:
                rows = data

            # concatenate to existing dataframe
            data_all[i] = data_all[i].append(rows, ignore_index=True)

    return data_all

if __name__ == '__main__':
    folders = [['C:\\Users\\nicol\\Documents\\Repos\\gastrSegment_testData\\2020-02-20_David_TL\\g03G']]
    channel = 1
    distributionType = 'Average'
    isTimelapse = False

    data = collect_fluo_data(folders, channel, distributionType, isTimelapse)
    print(data)
    # print(data[0]['ch%d_%s'%(channel,distributionType)].values[0])


'''
[
    {
    'input_file': ''
    'mask_file': ''
    'ch0_AP_profile': [...] 
    },
    {
    'input_file': ''
    'mask_file': ''
    'ch0_AP_profile': [...] 
    },
    {
    'input_file': ''
    'mask_file': ''
    'ch0_AP_profile': [...] 
    },
]

DATAFRAME for not timelapse:
groups, folders
[[f1,f2,f3],[....]]
{
        'ch0_Average'     'ch0_Background'
    'g1'  n                 n1
    'g2'  n                 n2
    'g3'  n                 n3
    'g4'  n                 n4
    'g5'  n                 n5
    'g6'  n                 n6
    'g7'  n                 n7
}

DATAFRAME for not timelapse:
groups, folders
[[f1,f2,f3],[....]]
[
    {
            'ch0_Approfile'     'ch0_Background'
        'g1'  [...]             n1
        'g2'  [...]             n2
        'g3'  [...]             n3
        'g4'  [...]             n4
        'g5'  [...]             n5
        'g6'  [...]             n6
        'g7'  [...]             n7
    },
    {
            'ch0_Approfile'     'ch0_Background'
        'g1'  [...]             n1
        'g2'  [...]             n2
        'g3'  [...]             n3
        'g4'  [...]             n4
        'g5'  [...]             n5
        'g6'  [...]             n6
        'g7'  [...]             n7
    }
]

DATAFRAME with timelapse:
[[f1=g1,f2=g2,f3=g3],[...]]
{
        'ch0_Approfile'                 'ch0_Background'
    'g1'  [[...],[...],...]             [n1,n2,n3,n4,...n1000]
    'g2'  [[...],[...],...]             [n1,n2,n3,n4,...n1000]
    'g3'  [[...],[...],...]             [n1,n2,n3,n4,...n1000]
}


'''
