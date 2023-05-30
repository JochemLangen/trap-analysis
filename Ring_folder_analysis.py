# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:40:05 2023

@author: joche
"""

import os
from funcs_photos import *
import pandas as pd
pi = np.pi

# rootdir = 'New_col\\Flipped_ring_axicon\\Dist_range'
rootdir = 'New_col\\Flipped_ring_axicon\\Img_dist_range'
i=0
pixel_array = []
dist_array = []
dist_float_factor = 1000
exp_array = []
m_array = []
m_err_array = []
R_array = []
R_err_array = []
exp_float_factor = 100
# var_path_len = 6
# file_type_len = 4
# var_path_start = var_path_len+file_type_len
single_dir = True

saturation_value = 2**16-2**4

dir_index = 0
for subdir, dirs, files in os.walk(rootdir):
    if single_dir == True:
        if dir_index > 0:
            break
    dir_index += 1
    directory_name = (subdir[:-1])[subdir[:-1].rfind("\\")+1:]
    for file in files:
        path = os.path.join(subdir,file)
        if path[-3:] == 'csv':
            csv_file = pd.read_csv(path[:-4]+'.csv')
            R, R_err, x_offset, x_offset_err, y_offset, y_offset_err, avg_m, avg_m_err, m_std, m_std_err, avg_b, avg_b_err, b_std, b_std_err, avg_I_peak, avg_I_peak_err, avg_std, avg_std_err, avg_R, avg_R_err, avg_grad_peak, avg_grad_peak_err = csv_file.values[0,1:]
            m_array += [avg_m]
            m_err_array += [avg_m_err]
            R_array += [R]
            R_err_array += [R_err]
            
            dist_first_index = path.rfind("_")+1
            dist_array += [float(path[dist_first_index:path.rfind("mm")])/dist_float_factor]
            reduced_path = path[:dist_first_index-1]
            exp_array += [float(reduced_path[reduced_path.rfind("_")+1:reduced_path.rfind("us")])/exp_float_factor]

dist_array = np.asarray(dist_array)
exp_array = np.asarray(exp_array)
m_array = np.asarray(m_array)
m_err_array = np.asarray(m_err_array)
R_array = np.asarray(R_array)
R_err_array = np.asarray(R_err_array)

sorted_indices = np.argsort(dist_array)
dist_array = dist_array[sorted_indices]
exp_array = exp_array[sorted_indices]
m_array = m_array[sorted_indices]
m_err_array = m_err_array[sorted_indices]
R_array = R_array[sorted_indices]
R_err_array = R_err_array[sorted_indices]
print(m_array)

coeff_guess = 0.01
const_guess = 0
popt, pcov = curve_fit(linear,R_array,m_array,sigma=m_err_array,p0=[coeff_guess,const_guess],bounds=([0,-np.inf],np.inf))
coeff, const = popt
coeff_err, const_err = np.sqrt(np.diagonal(pcov))
print(coeff, coeff_err)
print(const,const_err)


fontsize=13
plt.figure()
plt.errorbar(R_array,m_array,xerr=R_err_array,yerr=m_err_array,marker='x',linestyle='',color=(0,0,0.4),zorder=5,capsize=2)
xmin, xmax, ymin, ymax = plt.axis()
x_values = np.linspace(xmin,xmax,2)
plt.plot(x_values,linear(x_values,coeff,const))

rootdir = 'New_col\\Flipped_ring_axicon\\Dist_range'
i=0
pixel_array = []
dist_array = []
dist_float_factor = 1000
exp_array = []
m_array = []
m_err_array = []
R_array = []
R_err_array = []
exp_float_factor = 100
# var_path_len = 6
# file_type_len = 4
# var_path_start = var_path_len+file_type_len
single_dir = True

saturation_value = 2**16-2**4

dir_index = 0
for subdir, dirs, files in os.walk(rootdir):
    if single_dir == True:
        if dir_index > 0:
            break
    dir_index += 1
    directory_name = (subdir[:-1])[subdir[:-1].rfind("\\")+1:]
    for file in files:
        path = os.path.join(subdir,file)
        if path[-3:] == 'csv':
            csv_file = pd.read_csv(path[:-4]+'.csv')
            R, R_err, x_offset, x_offset_err, y_offset, y_offset_err, avg_m, avg_m_err, m_std, m_std_err, avg_b, avg_b_err, b_std, b_std_err, avg_I_peak, avg_I_peak_err, avg_std, avg_std_err, avg_R, avg_R_err, avg_grad_peak, avg_grad_peak_err = csv_file.values[0,1:]
            m_array += [avg_m]
            m_err_array += [avg_m_err]
            R_array += [R]
            R_err_array += [R_err]
            
            dist_first_index = path.rfind("_")+1
            dist_array += [float(path[dist_first_index:path.rfind("mm")])/dist_float_factor]
            reduced_path = path[:dist_first_index-1]
            exp_array += [float(reduced_path[reduced_path.rfind("_")+1:reduced_path.rfind("us")])/exp_float_factor]

dist_array = np.asarray(dist_array)
exp_array = np.asarray(exp_array)
m_array = np.asarray(m_array)
m_err_array = np.asarray(m_err_array)
R_array = np.asarray(R_array)
R_err_array = np.asarray(R_err_array)

sorted_indices = np.argsort(dist_array)
dist_array = dist_array[sorted_indices]
exp_array = exp_array[sorted_indices]
m_array = m_array[sorted_indices]
m_err_array = m_err_array[sorted_indices]
R_array = R_array[sorted_indices]
R_err_array = R_err_array[sorted_indices]
print(m_array)

coeff_guess = 0.01
const_guess = 0
popt, pcov = curve_fit(linear,R_array,m_array,sigma=m_err_array,p0=[coeff_guess,const_guess],bounds=([0,-np.inf],np.inf))
coeff, const = popt
coeff_err, const_err = np.sqrt(np.diagonal(pcov))
print(coeff, coeff_err)
print(const,const_err)

plt.errorbar(R_array,m_array,xerr=R_err_array,yerr=m_err_array,marker='x',linestyle='',color=(0.5,0,0),zorder=5,capsize=2)
x_values = np.linspace(xmin,xmax,2)
plt.plot(x_values,linear(x_values,coeff,const),color=(1,0,0))

plt.ylabel("Average power law exponent",fontsize=fontsize)
plt.xlabel("Ring radius",fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim([xmin,xmax])
# plt.ylim([ymin,ymax])
plt.show()

