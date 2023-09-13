# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:42:13 2023

@author: joche
"""

import os
from funcs_photos import *
import pandas as pd

rootdir = 'Camera analysis\\exp_time'
i=0
pixel_array = []
var_array = []
var_path_len = 6
file_type_len = 4
var_path_start = var_path_len+file_type_len
float_factor = 100
saturation_value = 65520

for subdir, dirs, files in os.walk(rootdir):
    i += 1
    print(i)
    for file in files:
        path = os.path.join(subdir,file)
        # print(path)
        im = Image.open(path)
        pixel_array += [np.array(im).T]
        pixel_im = plt.imshow(np.array(im),interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=saturation_value)
        plt.colorbar(mappable=pixel_im,label='Normalised intensity')
        plt.show()
        var_array += [float(path[-var_path_start:-file_type_len])/float_factor]

var_array = np.array(var_array)
pixel_array = np.array(pixel_array)

valid_indices = np.logical_and(pixel_array < saturation_value, pixel_array > 0)
rel_avg = []
for i in range(len(pixel_array)-1):
    comb_valid_indices = np.logical_and.reduce(valid_indices[i:i+2],axis=0)
    rel_pixels = pixel_array[i+1][comb_valid_indices]/pixel_array[i][comb_valid_indices]
    rel_avg += [np.mean(rel_pixels)]

    
rel_avg = np.array(rel_avg)
print(rel_avg)
rel_var = var_array[1:]/var_array[:-1]
print(rel_var)
print(var_array)

fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
plt.scatter(rel_var,rel_avg,marker='x')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
max_val = max(xlim[-1],ylim[-1])
plt.xlim(xlim)
plt.ylim(ylim)
plt.plot([0,max_val],[0,max_val],linestyle='--',zorder=0)
plt.show()

fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
plt.scatter(var_array[1:],rel_avg/rel_var,marker='x')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
plt.xlim(xlim)
plt.ylim(ylim)
plt.plot([xlim[0],xlim[-1]],[1,1],linestyle='--',zorder=0)
plt.show()

        