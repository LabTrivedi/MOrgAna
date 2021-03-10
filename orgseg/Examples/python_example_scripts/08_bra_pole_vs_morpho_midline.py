# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:36:41 2020

@author: gritti
"""

import sys, time, tqdm, copy, os
sys.path.append(os.path.join('..'))

from orgseg.ImageTools.morphology import meshgrid
from orgseg.DatasetTools.straightmorphology import io as ioStr
from orgseg.DatasetTools.morphology import io as ioMorph
from orgseg.DatasetTools.fluorescence import io as ioFluo

import pandas as pd
import tqdm
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.io import imread, imsave
from skimage.filters import gaussian
from skimage.measure import regionprops

import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")
from matplotlib import rc
rc('font', size=12)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)

#####################################################

parent_folder = os.path.join('test_data','2020-02-20_conditions')
    
### add gastruloids that you want to ignore here
exclude_gastr = []

###########################################################################

if __name__ == '__main__':
#    print('Waiting 5 hours...')
#    for i in tqdm.tqdm(range(60*5)):
#        time.sleep(60)

    ### compute parent folder as absolute path
    parent_folder = os.path.abspath(parent_folder)
    
    #### find out all gastruloids in parent_folder
    gastr_names = next(os.walk(parent_folder))[1]
    
    ### exclude gastruloids
    gastr_names = [g for g in gastr_names if not g in exclude_gastr ]
    
#    gastr = [gastr_names[0]]
    
    for gastr in gastr_names:
        print('-------------'+gastr+'------------')
        
        save_folder = os.path.join(parent_folder,gastr,'result_segmentation')
        
        bra_fname = os.path.join(save_folder,'Bra_pole_info.json')
        
        if not os.path.exists(bra_fname):
#            print('Waiting 10 minutes...')
#            for i in tqdm.tqdm(range(10)):
#                time.sleep(60)
                
            df_morpho = ioMorph.load_morpho_params( save_folder, gastr )
            df_straight = ioStr.load_straight_morpho_params(
                                                                            save_folder, 
                                                                            gastr
                                                                            )
            df_fluo = ioFluo.load_fluo_info( save_folder, gastr )

            N_img = len(df_morpho.input_file)
            
            x_morphopole = np.array([0. for i in range(N_img)])
            y_morphopole = np.array([0. for i in range(N_img)])
            axratios = np.array([0. for i in range(N_img)])

            x_fluomax = np.array([0. for i in range(N_img)])
            y_fluomax = np.array([0. for i in range(N_img)])
            alphas_fluomax = np.array([0. for i in range(N_img)])
            
            x_fluocm = np.array([0. for i in range(N_img)])
            y_fluocm = np.array([0. for i in range(N_img)])
            alphas_fluocm = np.array([0. for i in range(N_img)])

            times = np.array([0. for i in range(N_img)])

            for i in tqdm.trange(N_img):
#                print(i, df_morpho.input_file[i])
                
                times[i] = i
                
                # load images
                f_in = df_morpho.input_file[i]
                f_ma = df_morpho.mask_file[i]
                _slice = df_morpho.slice[i]
                image = imread(os.path.join(parent_folder,gastr,f_in))
                image = np.stack([ img[_slice].astype(np.float) for img in image ])
                bckg = df_fluo.ch1_Background[i]
                image[1] = image[1].astype(float)-bckg
                image[1] = np.clip(image[1],0,None)
                mask = imread(os.path.join(parent_folder,gastr,f_ma))[_slice]
                
                # compute the meshgrid
                tangent = df_morpho.tangent[i]
                midline = df_morpho.midline[i]
                width = df_morpho.meshgrid_width[i]
                mesh = df_morpho.meshgrid[i]
                if mesh == None:
                    mesh = meshgrid.compute_meshgrid(
                                                                                midline,
                                                                                tangent,
                                                                                width
                                                                                )
            
                # straighten the mask and the image
                mesh_shape = mesh.shape
                coords_flat = np.reshape( mesh, (mesh_shape[0]*mesh_shape[1],2) ).T
            
                ma_straight = map_coordinates(mask,
                                              coords_flat,
                                              order=0,
                                              mode='constant',
                                              cval=0).T
                ma_straight = np.reshape(ma_straight,(mesh_shape[0],mesh_shape[1]))
            
                fl_straight = map_coordinates(image[1],
                              coords_flat,
                              order=0,
                              mode='constant',
                              cval=0).T
                fl_straight = np.reshape(fl_straight,(mesh_shape[0],mesh_shape[1]))
                fl_straight_masked = fl_straight * ma_straight
            
                bf_straight = map_coordinates(image[0],
                              coords_flat,
                              order=0,
                              mode='constant',
                              cval=0).T
                bf_straight = np.reshape(bf_straight,(mesh_shape[0],mesh_shape[1]))
                
                ( length, width ) = ma_straight.shape
            
                # flip images if meshgrid wrong
                APprof = df_fluo.ch1_APprofile[i]
                first_half = APprof[:int(length/2)]
                second_half = APprof[int(length/2):]
                flip = False
                if np.sum(first_half) < np.sum(second_half):
                    ma_straight = ma_straight[::-1]
                    fl_straight_masked = fl_straight_masked[::-1]
                    fl_straight = fl_straight[::-1]
                    bf_straight = bf_straight[::-1]
                    flip = True
                    
                # find coordinate of highest fluorescence value
                fl_gauss = gaussian(fl_straight_masked, sigma=10)
                max_val = np.max(fl_gauss[:int(length/2)])
                max_pos = np.where(fl_gauss[:int(length/2)]==max_val)
                max_pos = np.array([i[0] for i in max_pos])

                # find coordinate of highest fluorescence value in the upward
                prop = regionprops(ma_straight, fl_straight_masked)
                centroid_fluo = prop[0].weighted_centroid
                
                # find coordinate of upward pole
                pole_pos = np.array([0,int(width/2)])
                
                # plot relevant variables
                centroid = np.array(df_straight.centroid[i]).astype(np.uint16)
                if flip:
                    centroid[0] = length - centroid[0]
                
                fig1, ax1 = plt.subplots(1, 2, figsize=(10,5))
                ax1[0].imshow(fl_straight,cmap='gray')
                ax1[0].plot([centroid[1],pole_pos[1]],[centroid[0],pole_pos[0]],'--w')
                ax1[0].plot([centroid[1],max_pos[1]],[centroid[0],max_pos[0]],'--r')
                ax1[0].plot([centroid[1],centroid_fluo[1]],[centroid[0],centroid_fluo[0]],'--g')
                ax1[1].imshow(bf_straight,cmap='gray')
                ax1[1].plot([centroid[1],pole_pos[1]],[centroid[0],pole_pos[0]],'--w')
    #            plt.show(fig1)
#                plt.pause(10)
                fig1.savefig(os.path.join(save_folder,'gastr_tp%05d.pdf'%i))
                plt.close(fig1)
                
                # find angle between
                v_morpho = (pole_pos-centroid)[::-1]
                v_morpho[1] *= -1.
                x_morphopole[i] = v_morpho[0]
                y_morphopole[i] = v_morpho[1]
                v_morpho = v_morpho/np.linalg.norm(v_morpho) # versor in the pole direction
            
                v_fluomax = (max_pos-centroid)[::-1]
                v_fluomax[1] *= -1.
                x_fluomax[i] = v_fluomax[0]
                y_fluomax[i] = v_fluomax[1]
                v_fluomax = v_fluomax/np.linalg.norm(v_fluomax) # versor in the bra max direction
                
                v_fluocm = (centroid_fluo-centroid)[::-1]
                v_fluocm[1] *= -1.
                x_fluocm[i] = v_fluocm[0]
                y_fluocm[i] = v_fluocm[1]
                v_fluocm = v_fluocm/np.linalg.norm(v_fluocm)
            
                dot = np.dot(v_fluomax,v_morpho)
                sign = np.sign((v_fluomax-v_morpho)[0])
                alpha = sign*np.arccos(dot)
                alphas_fluomax[i] = alpha*180/np.pi

                dot = np.dot(v_fluocm,v_morpho)
                sign = np.sign((v_fluocm-v_morpho)[0])
                alpha = sign*np.arccos(dot)
                alphas_fluocm[i] = alpha*180/np.pi
                
                # find axis ratio
                maj_ax = df_straight.major_axis_length[i]
                min_ax = df_straight.minor_axis_length[i]
                axratios[i] = min_ax/maj_ax
                
            data = pd.DataFrame({
                            'x_morphopole':x_morphopole,
                            'y_morphopole':y_morphopole,
                            'x_fluomax':x_fluomax,
                            'y_fluomax':y_fluomax,
                            'alphas_fluomax':alphas_fluomax,
                            'x_fluocm':x_fluocm,
                            'y_fluocm':y_fluocm,
                            'alphas_fluocm':alphas_fluocm,
                            'axratio':axratios,
                            'times':times,
                            })   
        
            data.to_json(bra_fname)
        
