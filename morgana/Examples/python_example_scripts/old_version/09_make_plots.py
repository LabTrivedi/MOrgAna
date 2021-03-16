# # -*- coding: utf-8 -*-
# """
# Created on Thu Apr 23 11:36:41 2020

# @author: gritti
# """

import sys, time, tqdm, copy, os
sys.path.append(os.path.join('..'))

import pandas as pd
import numpy as np

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

# #####################################################

parent_folder = os.path.join('test_data','2020-02-20_conditions')
    
### add gastruloids that you want to ignore here
exclude_gastr = []

#####################################################

if __name__ == '__main__':
#    print('Waiting 5 hours...')
#    for i in tqdm.tqdm(range(60*5)):
#        time.sleep(60)

#    exp_folder = 'Y:\\Germaine_Aalderink\\raw_data\\2020-03-03_HCR_tbxta_pescoids'

    # find out aall gastruloids in parent_folder
    gastr_names = next(os.walk(parent_folder))[1]
    gastr_names = [g for g in gastr_names if not g in exclude_gastr ]
    
#     #####################################################
#     gastr = gastr_names[0]
    
# #    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    for gastr in tqdm.trange(gastr_names):
        
        save_folder = os.path.join(parent_folder,gastr,'result_segmentation')
        print(save_folder)
        
        bra_fname = os.path.join(save_folder,'Bra_pole_info.json')
        
#         data = pd.read_json(bra_fname)
#     #    data = data.sort_values(by='axratio')
#         fig2, ax2 = plt.subplots(1, 1, figsize=(15,10))
#         ax2.plot(data.times,data.x_morphopole,'-b',alpha=0.2)
#         ax2.plot(data.times,data.x_fluocm,'-b')
#         ax2.plot(data.times,data.y_fluocm,'-r')
#         ax2.plot(data.times,data.y_morphopole,'-r',alpha=0.2)
#         ax2.plot(data.times,data.axratio,'-k')
#         ax2.set_title('X-Y pole and fluo cm pos')
#         ax2.set_title('Ax ratio')

#         plt.show(fig2)
#         # fig2.savefig(os.path.join(save_folder,'gastr_brapole_vs_morphomid.pdf'))
#         # plt.close(fig2)
        
# #        ax[0].plot(data.times,data.alphas,'-k',alpha=0.4)
# #        ax[1].plot(data.times,data.axratio,'-k',alpha=0.4)

# #    ax[0].plot([0,np.max(data.times)+1],[0,0],'--k')
# #    plt.show()
        
#     #plt.figure()
#     #plt.plot(data.axratio,data.distances,'-ob')
#     #plt.show()
        
