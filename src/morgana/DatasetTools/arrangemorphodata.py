import os
import pandas as pd

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join('..'))

from morgana.DatasetTools.morphology import computemorphology
from morgana.DatasetTools.morphology import io as ioMorph
from morgana.DatasetTools.straightmorphology import computestraightmorphology
from morgana.DatasetTools.straightmorphology import io as ioStraightMorph

def collect_morpho_data(groups, morpho_params, computeMorpho, maskType, isTimelapse=False):

    if maskType == "Unprocessed":
        compute_morphological_info = computemorphology.compute_morphological_info
        save_morphological_info = ioMorph.save_morpho_params
        load_morphological_info = ioMorph.load_morpho_params
        file_extension = '_morpho_params.json'
    else:
        compute_morphological_info = computestraightmorphology.compute_straight_morphological_info
        save_morphological_info = ioStraightMorph.save_straight_morpho_params
        load_morphological_info = ioStraightMorph.load_straight_morpho_params
        file_extension = '_morpho_straight_params.json'

    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    morpho_params = [ m for m,c in zip(morpho_params,computeMorpho) if c  ]

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
            fname = os.path.join(save_folder,cond+file_extension)

            # if morpho_params of straighten image not created yet, compute and save
            if not os.path.exists(fname):
                data = compute_morphological_info(input_folder)
                save_morphological_info( save_folder, cond, data )
            else:
                # else, just load it
                data = load_morphological_info(  save_folder, cond  )
                # if morpho_params was created long time ago and doesn't have all parameters, create it again
                if not all(mp in data.keys() for mp in morpho_params):
                    data = compute_morphological_info(input_folder)
                    save_morphological_info( save_folder, cond, data )

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

            # filter for the morpho params needed
            data = data[morpho_params]

            if isTimelapse:
                # if this is a timelapse dataset, all data should be stored in the same object
                rows = pd.Series({ key: list(data[key].values) for key in morpho_params })
            else:
                rows = data

            # concatenate to existing dataframe
            data_all[i] = data_all[i].append(rows, ignore_index=True)

    return data_all, morpho_params
    
if __name__ == '__main__':
    folders = [['C:\\Users\\nicol\\Documents\\Repos\\gastrSegment_testData\\2020-02-20_David_TL\\g03G']]
    morpho_params = [
                        'area',
                        'eccentricity',
                        'major_axis_length',
                        'minor_axis_length',
                        'equivalent_diameter',
                        'perimeter',
                        'euler_number',
                        'extent',
                        'orientation',
                        'elliptical_fourier_transform'
                        ]
    computeMorpho = [True for i in morpho_params]
    maskType = ['Unprocessed']
    isTimelapse = False

    data, _ = collect_morpho_data(folders, morpho_params, computeMorpho, maskType, isTimelapse)
    print(data)
    # print(data[0]['ch%d_%s'%(channel,distributionType)].values[0])
