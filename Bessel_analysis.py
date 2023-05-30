# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 23:36:09 2023

@author: joche
"""
import os
from funcs_photos import *
import pandas as pd

rootdir = 'Feb\\Bessel\\3rd'
pixel_array = []
var_array = []
file_type_len = 4
float_factor = 1000
saturation_value = 65520
pixel_size = np.average([6.14/1280,4.9/1024]) #In mm


for subdir, dirs, files in os.walk(rootdir):
    for file in files:

        path = os.path.join(subdir,file)
        # print(path)
        im = Image.open(path)
        pixel_array += [np.array(im)]
        var_array += [float(path[path.rfind("_")+1:-file_type_len])/float_factor]

var_array = np.array(var_array)
var_indices = np.argsort(var_array)
var_array = var_array[var_indices]
pixel_array = np.array(pixel_array)[var_indices]
var_array = 25.-var_array

xticks = np.array([0,1.2,2.4,3.6,4.8,6])/pixel_size
yticks = np.array([0,0.9,1.8,2.7,3.6,4.5])/pixel_size

i=0
j=0
j_jump = 0.15
size = 0.45
fig = plt.figure()
for k in range(len(var_array)):
    i+=1
    if i == 1:
        ax = fig.add_axes((-size*i,size*j,size,size))
        pixel_im = plt.imshow(pixel_array[k],interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=saturation_value)
        # plt.colorbar(mappable=pixel_im,label='Normalised intensity')
    else:
        ax = fig.add_axes((-size*i,size*j,size,size))
        pixel_im = plt.imshow(pixel_array[k],interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=saturation_value)

    if j == 0:
        plt.xticks(xticks[:-1],labels=xticks[:-1]*pixel_size)
        plt.xlabel("Horizontal image axis (mm)")
    else:
        plt.xticks([])

    
    if i == 2:
        plt.ylabel("Vertical image axis (mm)")
        plt.yticks(yticks,labels=yticks*pixel_size)
        if j == 0:

            plt.xticks(xticks[:-1],labels=xticks[:-1]*pixel_size)
            plt.xlabel("Horizontal image axis (mm)")
        else:
            plt.xticks([])
        i = 0
        j += 1 + j_jump
    else:
        plt.yticks([])
    plt.title("Distance {} mm".format(var_array[k]))

path = 'Feb\\Gaussian_Beam_2\\Gaussian_beam_01037us.pgm'
file_type_len = 4
float_factor = 1000
# saturation_value /= 2**8

ax = fig.add_axes((0.15,(4*(1+j_jump)/2-1)*size,1.5*size,1.5*size))
im = Image.open(path)
pixels = np.array(im)/saturation_value
pixel_im = plt.imshow(pixels,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
plt.colorbar(mappable=pixel_im,label='Normalised intensity')
plt.yticks(yticks,labels=yticks*pixel_size)
plt.xticks(xticks,labels=xticks*pixel_size)
plt.xlabel("Horizontal image axis (mm)")
plt.ylabel("Vertical image axis (mm)")
plt.title("Gaussian beam")
plt.show()    