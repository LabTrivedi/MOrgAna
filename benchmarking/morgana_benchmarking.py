# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:28:47 2021

@author: nicol
"""

from skimage.io import imread
from sklearn.metrics import jaccard_score, recall_score, precision_score, accuracy_score
import os, glob, tqdm
import numpy as np
from skimage import color
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from matplotlib import rc
rc('font', size=18)
rc('font', family='Arial')
rc('pdf', fonttype=42)
plt.rcParams.update({'font.size': 15})

path_gt = os.path.join('ground_truth')
list_gt = glob.glob(os.path.join(path_gt,'*_GT.tif'))
list_gt.sort()

path_morgana = os.path.join('images','result_segmentation')
list_morgana = glob.glob(os.path.join(path_morgana,'*_finalMask.tif'))
list_morgana.sort()

path_cp = os.path.join('cellprofiler')
list_cp = glob.glob(os.path.join(path_cp,'*_finalMask.tiff'))
list_cp.sort()

path_organoseg = os.path.join('organoseg')
list_orgseg = glob.glob(os.path.join(path_organoseg,'*_colored.tif'))
list_orgseg.sort()

times = [16*60+54, 37*60+45, 18*60+15]

n_img = len(list_gt)
n_max = -1

if not os.path.exists('scores_benchmarking.npz'):
    print('Computing')

    score_morgana = []
    score_cp = []
    score_orgseg = []
    for f_gt, f_morgana, f_cp, f_orgseg in tqdm.tqdm(zip(list_gt[:n_max], list_morgana[:n_max], list_cp[:n_max], list_orgseg[:n_max])):
        gt = (imread(f_gt)>0).flatten()
        morgana = (imread(f_morgana)>0).flatten()
        cp = (imread(f_cp)>0).flatten()
        orgseg = color.rgb2gray(imread(f_orgseg))
        orgseg = (orgseg*(2**16-1)).astype(np.uint16)
        l = label(orgseg)
        rp = regionprops(l)
        size = max([i.area for i in rp])
        orgseg = remove_small_objects(orgseg, min_size=size-1)
        orgseg = (orgseg>0).flatten()
        
        score_morgana.append([
                        jaccard_score(gt, morgana), 
                        precision_score(gt, morgana), 
                        recall_score(gt, morgana), 
                        accuracy_score(gt, morgana)
                        ])
        
        score_cp.append([
                        jaccard_score(gt, cp), 
                        precision_score(gt, cp), 
                        recall_score(gt, cp), 
                        accuracy_score(gt, cp)
                        ])
    
        score_orgseg.append([
                        jaccard_score(gt, orgseg), 
                        precision_score(gt, orgseg), 
                        recall_score(gt, orgseg), 
                        accuracy_score(gt, orgseg)
                        ])
        
    score_morgana = np.array(score_morgana)    
    score_cp = np.array(score_cp)
    score_orgseg = np.array(score_orgseg)
    
    print(np.mean(score_morgana,0),np.std(score_morgana,0))
    print(np.mean(score_cp,0),np.std(score_cp,0))
    print(np.mean(score_orgseg,0),np.std(score_orgseg,0))

    np.savez('scores_benchmarking.npz', morgana=score_morgana, cellprofiler=score_cp, organoseg=score_orgseg)

else:
    print('Loading')

    score_npz = np.load('scores_benchmarking.npz')
    
    score_morgana = score_npz['morgana']
    score_cp = score_npz['cellprofiler']
    score_orgseg = score_npz['organoseg']

print(score_morgana)

df_morgana = pd.DataFrame({'software':'MOrgAna',
                           'jaccard':score_morgana[:,0],
                           'precision':score_morgana[:,1],
                           'recall':score_morgana[:,2],
                           'accuracy':score_morgana[:,3],
                           'times':times[0]/60})

df_cp = pd.DataFrame({'software':'CellProfiler',
                           'jaccard':score_cp[:,0],
                           'precision':score_cp[:,1],
                           'recall':score_cp[:,2],
                           'accuracy':score_cp[:,3],
                           'times':times[1]/60})

df_orgseg = pd.DataFrame({'software':'OrganoSeg',
                           'jaccard':score_orgseg[:,0],
                           'precision':score_orgseg[:,1],
                           'recall':score_orgseg[:,2],
                           'accuracy':score_orgseg[:,3],
                           'times':times[2]/60})

df = pd.concat([df_morgana, df_cp, df_orgseg])

fig,ax = plt.subplots(nrows=1,ncols=5, figsize=(10,5))
fig.subplots_adjust(top=0.97, bottom=0.2, left=0.07, right=0.97, wspace=0.55)


sns.barplot(data=df, x='software', y='times', ax=ax[0])
sns.boxplot(data=df, x='software', y='jaccard', ax=ax[1], showfliers=False)
sns.boxplot(data=df, x='software', y='precision', ax=ax[2], showfliers=False)
sns.boxplot(data=df, x='software', y='recall', ax=ax[3], showfliers=False)
sns.boxplot(data=df, x='software', y='accuracy', ax=ax[4], showfliers=False)

ax[1].set_yticks([0.4,0.6,0.8,1.0])
ax[2].set_yticks([0.4,0.6,0.8,1.0])
ax[3].set_yticks([0.7,0.8,0.9,1.0])
ax[4].set_yticks([0.7,0.8,0.9,1.0])


for a in ax:
    l = a.get_xticklabels()
    a.set_xticklabels(l, rotation=45, ha='right')
    a.set_xlabel('')

ax[0].set_ylabel('Time')
ax[1].set_ylabel('Jaccard')
ax[2].set_ylabel('Precision')
ax[3].set_ylabel('Recall')
ax[4].set_ylabel('Accuracy')
    
    
ax[1].set_ylim(0.4,1)
ax[2].set_ylim(0.4,1)
ax[3].set_ylim(0.7,1)
ax[4].set_ylim(0.7,1)

fig.savefig('Benchmarking.pdf')

