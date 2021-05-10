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

path_organoseg = os.path.join('organoseg')
list_orgseg = glob.glob(os.path.join(path_organoseg,'*_colored.tif'))
list_orgseg.sort()

n_img = len(list_orgseg)

for f_orgseg in tqdm.tqdm(list_orgseg):
    orgseg = color.rgb2gray(imread(f_orgseg))
    orgseg = (orgseg*(2**16-1)).astype(np.uint16)
    l = label(orgseg)
    rp = regionprops(l)
    size = max([i.area for i in rp])
    orgseg = remove_small_objects(orgseg, min_size=size-1)

    imsave('mask%02d.tif', orgseg)
    
    