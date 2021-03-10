# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:24:40 2020

@author: gritti
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:46:26 2020

@author: gritti
"""

import os, tqdm
from skimage.io import imread, imsave
import numpy as np

# import sys
# sys.path.append(os.path.join('..'))
from orgseg.DatasetTools import io as ioDT
from orgseg.DatasetTools.segmentation import io as ioSeg
from orgseg.DatasetTools.morphology import io as ioMorph
from orgseg.DatasetTools.morphology import computemorphology, overview
from orgseg.ImageTools.segmentation import segment
from orgseg.GUIs import manualmask

###############################################################################

image_folders = [
                    os.path.join('test_data','2020-09-22_conditions','init_150cells'),
                    os.path.join('test_data','2020-09-22_conditions','init_300cells'),
                ]

###############################################################################

if __name__ == '__main__':

    for image_folder in image_folders:

        ### compute parent folder as absolute path
        image_folder = os.path.abspath(image_folder)
    
        print('-------------'+image_folder+'------------')

        result_folder = os.path.join(image_folder, 'result_segmentation')
        folder, cond = os.path.split(image_folder)

        flist_in, chosen_masks, down_shapes, thinnings, smoothings = ioSeg.load_segmentation_params( result_folder )
        flist_in = [os.path.join(image_folder, f) for f in flist_in]
        n_imgs = len(flist_in)

        #######################################################################
        ### clean masks previously generated
        ### only use this part if you want to rewrite final masks!

#        flist_to_remove = ioDT.get_image_list(result_folder, '_finalMask', 'include')
#        for f in flist_to_remove:
#            os.remove(f)
#        morpho_file = os.path.join(result_folder,cond+'_morpho_params.json')
#        if os.path.exists(morpho_file):
#            os.remove(morpho_file)

        #######################################################################
        ### generate final mask if not there yet
        print('Generating smooth masks:')

        for i in tqdm.tqdm(range(n_imgs)):

            folder, filename = os.path.split(flist_in[i])
            filename, extension = os.path.splitext(filename)
            
            final_mask_name = os.path.join(result_folder,filename+'_finalMask'+extension)
            if not os.path.exists(final_mask_name):
                
                if chosen_masks[i] == 'w':
                    _rawmask = imread( os.path.join(result_folder, filename+'_watershed'+extension) )
                    mask = segment.smooth_mask( 
                                                _rawmask, 
                                                mode='watershed',
                                                down_shape=down_shapes[i], 
                                                smooth_order=smoothings[i] 
                                                )
                    while (np.sum(mask)==0)&(smoothings[i]>5):
                        print('Mask failed...')
                        # if mask is zero, try smoothing less
                        smoothings[i] -= 2
                        print('Trying with: smoothing', smoothings[i])
                        mask = segment.smooth_mask( 
                                                    _rawmask, 
                                                    mode='watershed',
                                                    down_shape=down_shapes[i], 
                                                    smooth_order=smoothings[i] 
                                                    )
                    
                elif chosen_masks[i] == 'c':
                    _rawmask = imread( os.path.join(result_folder, filename+'_classifier'+extension) )
                    mask = segment.smooth_mask( 
                                                _rawmask, 
                                                mode='classifier',
                                                down_shape=down_shapes[i], 
                                                smooth_order=smoothings[i],
                                                thin_order=thinnings[i] 
                                                )
                    while (np.sum(mask)==0)&(smoothings[i]>5)&(thinnings[i]>1):
                        print('Mask failed...')
                        # if mask is zero, try smoothing less
                        smoothings[i] -= 2
                        thinnings[i] -= 1
                        print('Trying with: smoothing', smoothings[i],' thinnings', thinnings[i])
                        mask = segment.smooth_mask( 
                                                    _rawmask, 
                                                    mode='classifier',
                                                    down_shape=down_shapes[i], 
                                                    smooth_order=smoothings[i],
                                                    thin_order=thinnings[i] 
                                                    )
                    
                elif chosen_masks[i] == 'm':
                    if not os.path.exists(os.path.join(result_folder,filename+'_manual'+extension)):
                        m = manualmask.makeManualMask(flist_in[i])
                        m.show()
                        m.exec()
                    else:
                        print('A previously generated manual mask exists!')
                    _rawmask = imread( os.path.join(result_folder,filename+'_manual'+extension) )
                    mask = segment.smooth_mask( 
                                                    _rawmask, 
                                                    mode='manual',
                                                    down_shape=down_shapes[i], 
                                                    smooth_order=smoothings[i] 
                                                    )
                    while (np.sum(mask)==0)&(smoothings[i]>5):
                        print('Mask failed...')
                        # if mask is zero, try smoothing less
                        smoothings[i] -= 2
                        print('Trying with: smoothing', smoothings[i])
                        # if mask is zero, try smoothing less
                        smoothings[i] -= 2
                        mask = segment.smooth_mask( 
                                                        _rawmask, 
                                                        mode='manual',
                                                        down_shape=down_shapes[i], 
                                                        smooth_order=smoothings[i] 
                                                        )
                elif chosen_masks[i] == 'i': 
                    continue
    
                if np.sum(mask) == 0:
                    print('Warning, no trainingset!','The method selected didn\'t generate a valid mask. Please input the mask manually.')
    
                    chosen_masks[i] = 'm'
                    ioSeg.save_segmentation_params(  
                                        result_folder, 
                                        [os.path.split(fin)[-1] for fin in flist_in],
                                        chosen_masks,
                                        down_shapes, 
                                        thinnings, 
                                        smoothings 
                                        )
                    if not os.path.exists(os.path.join(result_folder,filename+'_manual'+extension)):
                        m = manualmask.makeManualMask(flist_in[i])
                        m.show()
                        m.exec()
                    else:
                        print('A previously generated manual mask exists!')
                    _rawmask = imread( os.path.join(result_folder,filename+'_manual'+extension) )
                    mask = segment.smooth_mask( 
                                    _rawmask, 
                                    mode='manual',
                                    down_shape=down_shapes[i], 
                                    smooth_order=smoothings[i] 
                                    )
    
                ### save segmentation parameters
                ioSeg.save_segmentation_params(  
                                result_folder, 
                                [os.path.split(fin)[-1] for fin in flist_in],
                                chosen_masks,
                                down_shapes, 
                                thinnings, 
                                smoothings 
                                )
    
    
                ### save final mask
                imsave(final_mask_name, mask)

        print('### Done computing masks!')

        #######################################################################
        ### compute  morphology (also done in next script)

        morpho_file_name = os.path.join(result_folder,cond+'_morpho_params.json')
        
        if not os.path.exists(morpho_file_name):
            props = computemorphology.compute_morphological_info(image_folder, False)
            ioMorph.save_morpho_params(result_folder, cond, props)

        #######################################################################
        ### generate overview of masks (optional)

#        overview.generate_overview_finalMask(
#                                gastr_folder, 
#                                chosen=[c!='i' for c in chosen_masks], 
#                                saveFig=True, downshape=3
#                                )
