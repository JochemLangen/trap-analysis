# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:06:17 2023

@author: joche
"""

import os
from funcs_photos import *
import pandas as pd

rootdir = 'Camera analysis\\paper_test'
#8 bit images
exp_array = []
sum_array = []
sat_pix = []
zero_pix = []
pixel_array = []
zero_indices = []
sat_indices = []
zero_ind_array = []
sat_ind_array = []
float_factor = 100
saturation_value = 2**8-1
exp_time_error = 0.005 #us
plot_im = False

for subdir, dirs, files in os.walk(rootdir):
    directory_name = (subdir[:-1])[subdir[:-1].rfind("\\")+1:]
    for file in files:
        path = os.path.join(subdir,file)
        if path[-3:] == 'bmp':
            im = Image.open(path)
            pixels = np.array(im)
            
            non_sat_indices = pixels < saturation_value
            no_sat_pix = np.size(non_sat_indices)-np.count_nonzero(non_sat_indices)
            no_sat_pix_perc = no_sat_pix/np.size(non_sat_indices)
            print("There are {} saturated pixels ({}%).".format(no_sat_pix,np.round(no_sat_pix_perc*100,3)))
            sat_pix += [no_sat_pix_perc]
            
            zero_indices = pixels > 0
            no_zero_pix = np.size(zero_indices)-np.count_nonzero(zero_indices)
            no_zero_pix_perc = no_zero_pix/np.size(zero_indices)
            print("There are {} zero pixels ({}%).".format(no_zero_pix,np.round(no_zero_pix_perc*100,3)))
            zero_pix += [no_zero_pix_perc]
            
            # pixel_im = plt.imshow(pixels,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=saturation_value)
            # plt.colorbar(mappable=pixel_im,label='Normalised intensity')
            # plt.title(file)
            # plt.show()
            sum_array += [np.sum(pixels)]
            exp_scnd_index = path.rfind("us")
            reduced_path = path[:exp_scnd_index]
            exp_array += [float(reduced_path[reduced_path.rfind("_")+1:])/float_factor]
            
            pixel_array += [pixels]
            sat_ind_array += [np.logical_not(non_sat_indices)]
            zero_ind_array += [np.logical_not(zero_indices)]
            
exp_array = np.asarray(exp_array)
sum_array = np.asarray(sum_array)
sat_pix = np.asarray(sat_pix)
zero_pix = np.asarray(zero_pix)
pixel_array = np.asarray(pixel_array)
sat_ind_array = np.asarray(sat_ind_array)
zero_ind_array = np.asarray(zero_ind_array)


sorted_indices = np.argsort(exp_array)
exp_array = exp_array[sorted_indices]
sum_array = sum_array[sorted_indices]
sat_pix = sat_pix[sorted_indices]
zero_pix = zero_pix[sorted_indices]
pixel_array = pixel_array[sorted_indices]
sat_ind_array = sat_ind_array[sorted_indices]
zero_ind_array = zero_ind_array[sorted_indices]

avg_exp_array = interval_avg(exp_array, 2)
avg_sum_array = interval_avg(sum_array, 2)
avg_sat_pix = interval_avg(sat_pix,2)
avg_zero_pix = interval_avg(zero_pix,2)

dimension = int(len(sum_array) / 2)
reshaped_sum = np.reshape(sum_array, (dimension,2))
sum_SE = np.std(reshaped_sum,axis=1)

# print(sum_array)
# print(avg_exp_array, avg_sum_array,avg_sat_pix,avg_zero_pix)


#Remove any of the images with zero or saturated pixels
remaining_indices = np.logical_and(avg_sat_pix == 0, avg_zero_pix == 0)
avg_exp_array = avg_exp_array[remaining_indices]
avg_exp_err = exp_time_error/np.sqrt(2)
avg_sum_array = avg_sum_array[remaining_indices]
sum_SE = sum_SE[remaining_indices]
print(avg_exp_array)
print(avg_sum_array)



#Generate normalised results
sum_SE = avg_sum_array/avg_sum_array[0]*np.sqrt((sum_SE/avg_sum_array)**2+(sum_SE[0]/avg_sum_array[0])**2)
avg_sum_array /= avg_sum_array[0]

avg_exp_err = avg_exp_array/avg_exp_array[0]*np.sqrt((avg_exp_err/avg_exp_array)**2+(avg_exp_err/avg_exp_array[0])**2)
avg_exp_array /= avg_exp_array[0]

def linear_no_cnst(x,m):
    return m*x

# #Perform fitting
popt, pcov = curve_fit(linear,avg_exp_array,avg_sum_array,sigma=sum_SE,absolute_sigma=True,p0=[1,0],bounds=([0,-np.inf],np.inf))
m, a = popt
m_err, a_err = np.sqrt(np.diagonal(pcov))
print("The linearity coefficient is: {} +/- {}".format(m,m_err))
print("The constant is: {} +/- {}".format(a,a_err))

#Perform fitting
# popt, pcov = curve_fit(linear_no_cnst,avg_exp_array,avg_sum_array,sigma=sum_SE,absolute_sigma=True,p0=[1,1],bounds=([0,-np.inf],np.inf))
# m = popt
# m_err = np.sqrt(np.diagonal(pcov))
# print("The linearity coefficient is: {} +/- {}".format(m,m_err))


exp_values = np.linspace(avg_exp_array[0],avg_exp_array[-1],2)
sum_array = linear(exp_values,m,a)

#Generate residuals and their error
sum_SE_err = abs(sum_SE/np.sqrt(2)) #sqrt(2*N-2)
norm_residuals = (avg_sum_array - linear(avg_exp_array,m,a))/sum_SE
resid_err = np.sqrt(sum_SE**2 + ((linear(avg_exp_array,m+m_err,a+a_err)-linear(avg_exp_array,m-m_err,a-a_err))/2)**2)
norm_resid_err = abs(norm_residuals)*np.sqrt((sum_SE_err/sum_SE)**2 + (resid_err/(norm_residuals*sum_SE))**2)

#Generate plot
fontsize = 18
D = 1
resid_ylim = 3

fig = plt.figure()
fig.add_axes((0,0,1,1))
plt.errorbar(avg_exp_array,avg_sum_array,yerr=sum_SE,xerr=avg_exp_err,marker='x',linestyle='',capsize=2,color='black')
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
exp_values = np.linspace(xlim[0],xlim[-1],2)
sum_array = linear(exp_values,m,a)
plt.plot(exp_values,sum_array,zorder=0)

plt.ylabel("Normalised total pixel intensity",fontsize=fontsize)
plt.xlim(xlim)
plt.ylim(ylim)
plt.xticks([])
plt.yticks(fontsize=fontsize)

plot = fig.add_axes((0,-0.3,1,0.3))
plot_resid(norm_residuals, norm_resid_err, avg_exp_array, plot, D, xlim[0], xlim[-1], fontsize, resid_ylim)
plt.xlabel("Normalised exposure time",fontsize=fontsize)
rootdir = "D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col"
plt.savefig(rootdir+"\\lin_lin.svg",dpi=300,bbox_inches='tight')
plt.show()

        