# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:32:10 2023

@author: joche
"""

import os
from funcs_photos import *
import pandas as pd
pi = np.pi

fontsize = 15
dist_err = 0.0005
last_index = 0 #In case any values at the end of the array should not be used, the last index should be set to what will be used or 0 if everything should be.
save_results = False
distances =  True #Set to True if the file contains data at more than 1 distance.

#Folder:
# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\New_set\\Dist_range'
# top_index = 3
# last_index = -1

# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\Dist_range'
# top_index = 12
# last_index = -1

# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\125_mm_lens'
# top_index = 15

rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\200_mm_lens'
top_index = 11

#THis one does not have the std std yet:
# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\200_mm_lens\\Ax_dist_range'
# top_index = 7

# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_axicon_lens\\New_set'
# top_index = 2


# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flipped_ring_axicon\\Dist_range_2'
# top_index=7
# last_index = -1

# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_lens_axicon\\Ax_dist_range'
# dist_err = 0.1
# top_index = 1

# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\New_set\\Iris_range'
# dist_err = 0.05
# top_index = 4

#No distances:
# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_lens_axicon'
# top_index = 0
# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_lens_axicon\\Obs_obj'
# top_index = 0
# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_axicon_lens\\Obs_obj'
# top_index = 0
# rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_axicon_lens\\Obs_obj\\closer_obj'
# top_index = 0

rootdir_1 = rootdir_unpr_1 + '\\Processed'
#File name with the results from the analysis
file_name_1 = rootdir_1+'\\'+rootdir_unpr_1[rootdir_unpr_1.rfind("\\")+1:]+'_analysed.csv'
csv_file_1 = pd.read_csv(file_name_1)

#Folder:
rootdir_unpr_2 = rootdir_unpr_1 + '\\Adj_cam'
rootdir_2 = rootdir_unpr_2 + '\\Processed'
#File name with the results from the analysis
file_name_2 = rootdir_2+'\\'+rootdir_unpr_2[rootdir_unpr_2.rfind("\\")+1:]+'_analysed.csv'
try:
    csv_file_2 = pd.read_csv(file_name_2)
    
    if last_index == 0:
        #Extract data:
        dist_array_1, exp_array_1, R_array_1, R_err_array_1, R_std_array_1, R_std_err_array_1, x_offset_array_1, x_offset_err_array_1, y_offset_array_1, y_offset_err_array_1, avg_m_array_1, avg_m_err_array_1, m_std_array_1, m_std_err_array_1, avg_b_array_1, avg_b_err_array_1, b_std_array_1, b_std_err_array_1, avg_I_peak_array_1, avg_I_peak_err_array_1, I_std_array_1, I_std_err_array_1, rel_darkness_array_1, rel_darkness_err_array_1, avg_std_array_1, avg_std_err_array_1, std_std_array_1 , std_std_err_array_1, total_std_array_1, total_std_err_array_1 = csv_file_1.values[:,1:].T
    
        #Extract data:
        dist_array_2, exp_array_2, R_array_2, R_err_array_2, R_std_array_2, R_std_err_array_2, x_offset_array_2, x_offset_err_array_2, y_offset_array_2, y_offset_err_array_2, avg_m_array_2, avg_m_err_array_2, m_std_array_2, m_std_err_array_2, avg_b_array_2, avg_b_err_array_2, b_std_array_2, b_std_err_array_2, avg_I_peak_array_2, avg_I_peak_err_array_2, I_std_array_2, I_std_err_array_2, rel_darkness_array_2, rel_darkness_err_array_2, avg_std_array_2, avg_std_err_array_2, std_std_array_2 , std_std_err_array_2, total_std_array_2, total_std_err_array_2 = csv_file_2.values[:,1:].T
    else:
        #Extract data:
        dist_array_1, exp_array_1, R_array_1, R_err_array_1, R_std_array_1, R_std_err_array_1, x_offset_array_1, x_offset_err_array_1, y_offset_array_1, y_offset_err_array_1, avg_m_array_1, avg_m_err_array_1, m_std_array_1, m_std_err_array_1, avg_b_array_1, avg_b_err_array_1, b_std_array_1, b_std_err_array_1, avg_I_peak_array_1, avg_I_peak_err_array_1, I_std_array_1, I_std_err_array_1, rel_darkness_array_1, rel_darkness_err_array_1, avg_std_array_1, avg_std_err_array_1, std_std_array_1 , std_std_err_array_1, total_std_array_1, total_std_err_array_1 = csv_file_1.values[:last_index,1:].T
    
        #Extract data:
        dist_array_2, exp_array_2, R_array_2, R_err_array_2, R_std_array_2, R_std_err_array_2, x_offset_array_2, x_offset_err_array_2, y_offset_array_2, y_offset_err_array_2, avg_m_array_2, avg_m_err_array_2, m_std_array_2, m_std_err_array_2, avg_b_array_2, avg_b_err_array_2, b_std_array_2, b_std_err_array_2, avg_I_peak_array_2, avg_I_peak_err_array_2, I_std_array_2, I_std_err_array_2, rel_darkness_array_2, rel_darkness_err_array_2, avg_std_array_2, avg_std_err_array_2, std_std_array_2 , std_std_err_array_2, total_std_array_2, total_std_err_array_2 = csv_file_2.values[:last_index,1:].T
    print(dist_array_1)
    # fig = plt.figure()
    # ax = fig.add_axes((0,-0.3,1,0.3))
    # plt.errorbar(dist_array_1,avg_b_array_1,yerr=avg_b_err_array_1, xerr=dist_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.ylabel("Fitted constant",fontsize=fontsize,labelpad=2)
    # plt.xlabel("Distance (mm)")
    # plt.show()
    # fig = plt.figure()
    # ax = fig.add_axes((0,-0.3,1,0.3))
    # plt.errorbar(dist_array_2,avg_b_array_2,yerr=avg_b_err_array_2, xerr=dist_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.ylabel("Fitted constant",fontsize=fontsize,labelpad=2)
    # plt.xlabel("Distance (mm)")
    # plt.show()
    
    #Combine data:
    R_array, R_err_array = weighted_avg_2D_0(np.asarray([R_array_1,R_array_2]),np.asarray([R_err_array_1,R_err_array_2]))
    R_std_array, R_std_err_array = weighted_avg_2D_0(np.asarray([R_std_array_1,R_std_array_2]),np.asarray([R_std_err_array_1,R_std_err_array_2]))
    x_offset_array, x_offset_err_array = weighted_avg_2D_0(np.asarray([x_offset_array_1,x_offset_array_2]),np.asarray([x_offset_err_array_1,x_offset_err_array_2]))
    y_offset_array, y_offset_err_array = weighted_avg_2D_0(np.asarray([y_offset_array_1,y_offset_array_2]),np.asarray([y_offset_err_array_1,y_offset_err_array_2]))
    avg_m_array, avg_m_err_array = weighted_avg_2D_0(np.asarray([avg_m_array_1,avg_m_array_2]),np.asarray([avg_m_err_array_1,avg_m_err_array_2]))
    m_std_array, m_std_err_array = weighted_avg_2D_0(np.asarray([m_std_array_1,m_std_array_2]),np.asarray([m_std_err_array_1,m_std_err_array_2]))
    avg_b_array, avg_b_err_array = weighted_avg_2D_0(np.asarray([avg_b_array_1,avg_b_array_2]),np.asarray([avg_b_err_array_1,avg_b_err_array_2]))
    b_std_array, b_std_err_array = weighted_avg_2D_0(np.asarray([b_std_array_1,b_std_array_2]),np.asarray([b_std_err_array_1,b_std_err_array_2]))
    avg_I_peak_array, avg_I_peak_err_array = weighted_avg_2D_0(np.asarray([avg_I_peak_array_1,avg_I_peak_array_2]),np.asarray([avg_I_peak_err_array_1,avg_I_peak_err_array_2]))
    I_std_array, I_std_err_array = weighted_avg_2D_0(np.asarray([I_std_array_1,I_std_array_2]),np.asarray([I_std_err_array_1,I_std_err_array_2]))
    rel_darkness_array, rel_darkness_err_array = weighted_avg_2D_0(np.asarray([rel_darkness_array_1,rel_darkness_array_2]),np.asarray([rel_darkness_err_array_1,rel_darkness_err_array_2]))
    avg_std_array, avg_std_err_array = weighted_avg_2D_0(np.asarray([avg_std_array_1,avg_std_array_2]),np.asarray([avg_std_err_array_1,avg_std_err_array_2]))
    std_std_array, std_std_err_array = weighted_avg_2D_0(np.asarray([std_std_array_1,std_std_array_2]),np.asarray([std_std_err_array_1,std_std_err_array_2]))
    total_std_array, total_std_err_array = weighted_avg_2D_0(np.asarray([total_std_array_1,total_std_array_2]),np.asarray([total_std_err_array_1,total_std_err_array_2]))
    dist_array = dist_array_1
except:
    print("No adj_cam file found")
    if last_index == 0:
        #Extract data:
        print(csv_file_1.values[:,:].T)
        dist_array, exp_array, R_array, R_err_array, R_std_array, R_std_err_array, x_offset_array, x_offset_err_array, y_offset_array, y_offset_err_array, avg_m_array, avg_m_err_array, m_std_array, m_std_err_array, avg_b_array, avg_b_err_array, b_std_array, b_std_err_array, avg_I_peak_array, avg_I_peak_err_array, I_std_array, I_std_err_array, rel_darkness_array, rel_darkness_err_array, avg_std_array, avg_std_err_array, std_std_array, std_std_err_array, total_std_array, total_std_err_array = csv_file_1.values[:,1:].T

    else:
        #Extract data:
        dist_array, exp_array, R_array, R_err_array, R_std_array, R_std_err_array, x_offset_array, x_offset_err_array, y_offset_array, y_offset_err_array, avg_m_array, avg_m_err_array, m_std_array, m_std_err_array, avg_b_array, avg_b_err_array, b_std_array, b_std_err_array, avg_I_peak_array, avg_I_peak_err_array, I_std_array, I_std_err_array, rel_darkness_array, rel_darkness_err_array, avg_std_array, avg_std_err_array, std_std_array, std_std_err_array, total_std_array, total_std_err_array = csv_file_1.values[:last_index,1:].T

    
# scl_R_std_array = R_std_array/R_array
# scl_R_std_err_array = np.sqrt(R_std_err_array**2 + (scl_R_std_array*R_err_array)**2)/R_array
#R is already scaled as the std was determined in the coordinates normalised by the radius

scl_m_std_array = m_std_array/avg_m_array
scl_m_std_err_array = np.sqrt(m_std_err_array**2 + (scl_m_std_array*avg_m_err_array)**2)/avg_m_array
scl_b_std_array = b_std_array/avg_b_array
scl_b_std_err_array = np.sqrt(b_std_err_array**2 + (scl_b_std_array*avg_b_err_array)**2)/avg_b_array


scl_I_std_array = I_std_array/avg_I_peak_array
scl_I_std_err_array = np.sqrt(I_std_err_array**2 + (scl_I_std_array*avg_I_peak_err_array)**2)/avg_I_peak_array

scl_std_std_array = std_std_array/avg_std_array
scl_std_std_err_array = np.sqrt(std_std_err_array**2 + (scl_std_std_array*avg_std_err_array)**2)/avg_std_array


pixel_size = np.average([6.14/1280,4.9/1024,4.8e-3],weights=[0.005/1280,0.05/1024,0.05e-3]) #In mm
pixel_size_err = 1/np.sqrt(1/(0.005/1280)**2 + 1/(0.05/1024)**2 + 1/(0.05e-3)**2)
R_array *= pixel_size
R_err_array = np.sqrt((R_err_array*pixel_size_err)**2 + (pixel_size_err*R_array/pixel_size)**2)
# R_std_array *= pixel_size 
# R_std_err_array = np.sqrt((R_std_err_array*pixel_size_err)**2 + (pixel_size_err*R_std_array/pixel_size)**2)

#Generate relative avg std values
rel_avg_std_array = avg_std_array/avg_I_peak_array
rel_avg_std_err_array = np.sqrt(avg_std_err_array**2+(avg_I_peak_err_array*rel_avg_std_array)**2)/avg_I_peak_array

#Obtain fractional darkness
rel_darkness_array /= 100
rel_darkness_err_array /= 100

def chi_sqrd_red(y_fit,y,y_err,v):
    return np.sum(((y_fit - y)/y_err)**2)/v

def plot_and_fit_var(x_values,x_err,y_values,y_errs,top_index,xlim,coeff_guess=10):
    # L_x_values = x_values[:top_index+1]
    # R_x_values = x_values[top_index:]
    # # L_x_values = x_values[:top_index+1]-x_values[top_index]
    # # R_x_values = x_values[top_index:]-x_values[top_index]
    # L_y_values = y_values[:top_index+1]
    # R_y_values = y_values[top_index:]
    # L_y_errs = y_errs[:top_index+1]
    # R_y_errs = y_errs[top_index:]

    top_y = y_values[top_index]
    top_y_err = y_errs[top_index]
    top_x = x_values[top_index]
    
    x_values -= top_x
    
    def lin_simp(x,coeff,const):
        return coeff*x + const#+ y_values[top_index]
    
    def lin_comb(x,coeff_L,coeff_R,const):
        L_x = x[:top_index]
        R_x = x[top_index:]
        L_y = coeff_L*L_x + const
        R_y = coeff_R*R_x + const
        return np.append(L_y,R_y)
    
    def lin_comb_m(x,coeff_L,coeff_R):
        L_x = x[:top_index]
        R_x = x[top_index:]
        L_y = coeff_L*L_x + top_y
        R_y = coeff_R*R_x + top_y
        return np.append(L_y,R_y)
    
    def lin_comb_h(x,coeff_L,coeff_R):
        L_x = x[:top_index]
        R_x = x[top_index:]
        L_y = coeff_L*L_x + top_y+top_y_err
        R_y = coeff_R*R_x + top_y+top_y_err
        return np.append(L_y,R_y)
        
    popt_m, pcov_m = curve_fit(lin_comb_m,x_values,y_values,sigma=y_errs,absolute_sigma=True,p0=[coeff_guess,coeff_guess],bounds=([-np.inf,-np.inf],[np.inf,np.inf]))
    L_coeff, R_coeff = popt_m
    L_coeff_err, R_coeff_err = np.sqrt(np.diagonal(pcov_m))
    popt_h, pcov_h = curve_fit(lin_comb_h,x_values,y_values,sigma=y_errs,absolute_sigma=True,p0=[coeff_guess,coeff_guess],bounds=([-np.inf,-np.inf],[np.inf,np.inf]))
    L_coeff_h, R_coeff_h = popt_h
    L_coeff_err = np.sqrt(L_coeff_err**2+(L_coeff-L_coeff_h)**2)
    R_coeff_err = np.sqrt(R_coeff_err**2+(R_coeff-R_coeff_h)**2)
    # print(abs(L_coeff-L_coeff_h),abs(R_coeff-R_coeff_h))
    # print(L_coeff_err, R_coeff_err)
    
    # L_coeff, R_coeff, const = popt
    # L_coeff_err, R_coeff_err, const_err = np.sqrt(np.diagonal(pcov))
    # v = len(y_values) - 2
    # chi_red = chi_sqrd_red(lin_comb(x_values,L_coeff,R_coeff,const),y_values,y_errs,v)
    
    # popt, pcov = curve_fit(lin_comb,x_values,y_values,sigma=y_errs,absolute_sigma=True,p0=[coeff_guess,coeff_guess,top_y],bounds=([-np.inf,-np.inf,top_y-top_y_err],[np.inf,np.inf,top_y+top_y_err]))
    # L_coeff, R_coeff, const = popt
    # L_coeff_err, R_coeff_err, const_err = np.sqrt(np.diagonal(pcov))
    # v = len(y_values) - 2
    # chi_red = chi_sqrd_red(lin_comb(x_values,L_coeff,R_coeff,const),y_values,y_errs,v)
    # print(L_coeff,L_coeff_err,R_coeff,R_coeff_err)
    # print(top_y,top_y_err)
    # print(chi_red)
    # #Left side fitting:
    # popt, pcov = curve_fit(lin_simp,L_x_values,L_y_values,sigma=L_y_errs,absolute_sigma=True,p0=[coeff_guess,top_y],bounds=([-np.inf,top_y-top_y_err],[np.inf,top_y+top_y_err]))
    # #Absolute sigma has been set to false as this provided a larger and thus more realistic error
    # L_coeff,L_const = popt
    # L_coeff_err = np.sqrt(np.diagonal(pcov))[0]
    # v = len(L_y_values) - 2
    # L_chi_red = chi_sqrd_red(lin_simp(L_x_values,L_coeff,L_const),L_y_values,L_y_errs,v)
    # print(L_coeff,L_coeff_err,L_const,L_chi_red)
    # print(chi)
    
    # #Right side fitting:
    # popt, pcov = curve_fit(lin_simp,R_x_values,R_y_values,sigma=R_y_errs,absolute_sigma=True,p0=[coeff_guess],bounds=(-np.inf,np.inf))
    # #Absolute sigma has been set to false as this provided a larger and thus more realistic error
    # R_coeff = popt[0]
    # R_coeff_err = np.sqrt(np.diagonal(pcov))[0]
    # v = len(R_y_values) - 1
    # R_chi_red = chi_sqrd_red(lin_simp(R_x_values,R_coeff),R_y_values,R_y_errs,v)
    # print(R_coeff,R_coeff_err,R_chi_red)


    plt.errorbar(x_values,y_values,yerr=y_errs,xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    plt.scatter(top_x,top_y,color='red',marker='o',zorder=6)
    # L_x_plot_val = np.asarray([xlim[0],top_x])
    # R_x_plot_val = np.asarray([top_x,xlim[1]])
    plt.plot(x_values,lin_comb_m(x_values,L_coeff,R_coeff),color='blue',zorder=0)
    # plt.plot(x_values,lin_comb(x_values,L_coeff,R_coeff,const),color='blue',zorder=0)
    # plt.scatter(top_x,const,color='red',marker='x')
    # plt.plot(L_x_plot_val,lin_simp(L_x_plot_val,L_coeff,L_const),color='blue',zorder=0)
    # plt.plot(R_x_plot_val,lin_simp(R_x_plot_val,R_coeff),color='blue',zorder=0)
    plt.xlim(xlim)


    return L_coeff, L_coeff_err, R_coeff, R_coeff_err#, const, const_err

# x_values = R_array
# x_err = R_err_array
# x_label = "R (mm)"
# xlim = [R_array[0]-0.02,R_array[-1]+0.02]
if distances == True:
    x_values = dist_array-dist_array[top_index]
    x_err = np.sqrt(2)*dist_err
    xlim = [x_values[0]-1,x_values[-1]+1]
    x_label = "d (mm)"
    # x_values = avg_I_peak_array
    # x_err = avg_I_peak_err_array
    # x_label = "I"
    # xlim = [avg_I_peak_array[0]-1,avg_I_peak_array[-1]+1]
    
    #Big combined plot:
    print(dist_array[top_index])
    
        
    fig = plt.figure()
    
    ax = fig.add_axes((0,0,1,0.3))
    L_m_coeff, L_m_coeff_err, R_m_coeff, R_m_coeff_err = plot_and_fit_var(x_values,x_err,avg_m_array,avg_m_err_array,top_index,xlim,coeff_guess=10)
    # plt.errorbar(x_values,avg_m_array,yerr=avg_m_err_array,xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.scatter(x_values[top_index],avg_m_array[top_index],color='red',marker='o',zorder=6)
    # plt.xlim(xlim)
    plt.ylabel(r"$\overline{m}$",fontsize=fontsize)
    # plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
    #             labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
    ax.xaxis.tick_top()
    yticks = ax.get_yticks()
    plt.xticks(fontsize=fontsize)
    plt.yticks(yticks[1:],fontsize=fontsize)
    
    # ax = fig.add_axes((0,-0.3,1,0.3))
    # plt.errorbar(x_values,avg_b_array,yerr=avg_b_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.ylabel("Constant",fontsize=fontsize,labelpad=4)
    # plt.xticks([])
    # plt.xlim(xlim)
    
    ax = fig.add_axes((0,-0.32,1,0.3))
    L_dark_coeff, L_dark_coeff_err, R_dark_coeff, R_dark_coeff_err = plot_and_fit_var(x_values,x_err,rel_darkness_array,rel_darkness_err_array,top_index,xlim,coeff_guess=10)
    # plt.errorbar(x_values,rel_darkness_array,yerr=rel_darkness_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.scatter(x_values[top_index],rel_darkness_array[top_index],color='red',marker='o',zorder=6)
    plt.ylabel(r"$\overline{b_{r}}$",fontsize=fontsize)
    plt.xticks([])
    plt.yticks(fontsize=fontsize)
    # plt.xlim(xlim)
    # print(L_dark_coeff, L_dark_coeff_err, R_dark_coeff, R_dark_coeff_err)
    
    ax = fig.add_axes((0,-0.64,1,0.3))
    L_I_coeff, L_I_coeff_err, R_I_coeff, R_I_coeff_err = plot_and_fit_var(x_values,x_err,avg_I_peak_array,avg_I_peak_err_array,top_index,xlim,coeff_guess=10)
    
    # plt.errorbar(x_values,avg_I_peak_array,yerr=avg_I_peak_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.scatter(x_values[top_index],avg_I_peak_array[top_index],color='red',marker='o',zorder=6)
    plt.ylabel(r"$\overline{I_{p}}$",fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    # plt.xlim(xlim)
    
    ax = fig.add_axes((0,-0.96,1,0.3))
    L_R_coeff, L_R_coeff_err, R_R_coeff, R_R_coeff_err = plot_and_fit_var(x_values,x_err,R_array,R_err_array,top_index,xlim,coeff_guess=10)
    
    # plt.errorbar(x_values,R_array,yerr=R_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.scatter(x_values[top_index],R_array[top_index],color='red',marker='o',zorder=6)
    plt.ylabel(r"$R$ (mm)",fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    # plt.xlim(xlim)
    
    ax = fig.add_axes((0,-1.28,1,0.3))
    L_std_coeff, L_std_coeff_err, R_std_coeff, R_std_coeff_err = plot_and_fit_var(x_values,x_err,rel_avg_std_array,rel_avg_std_err_array,top_index,xlim,coeff_guess=10)
    
    # plt.errorbar(x_values,avg_std_array,yerr=avg_std_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.ylabel(r"$\overline{\sigma_{resid.}}$",fontsize=fontsize)
    # plt.errorbar(x_values,rel_avg_std_array,yerr=rel_avg_std_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.scatter(x_values[top_index],rel_avg_std_array[top_index],color='red',marker='o',zorder=6)
    plt.ylabel(r"$\overline{\sigma_{resid.,r}}$",fontsize=fontsize)
    # plt.errorbar(x_values,total_std_array,yerr=total_std_err_array, xerr=x_err,marker='x',color='blue',capsize=2,linestyle='',zorder=5)
    plt.xlabel(x_label,fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # plt.xlim(xlim)
    
    
    #OTher side
    ax = fig.add_axes((1.05,0,1,0.3))
    plot_and_fit_var(x_values,x_err,scl_m_std_array,scl_m_std_err_array,top_index,xlim,coeff_guess=10)
    
    # plt.errorbar(x_values,scl_m_std_array,yerr=scl_m_std_err_array,xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.scatter(x_values[top_index],scl_m_std_array[top_index],color='red',marker='o',zorder=6)
    plt.ylabel(r"$\sigma_{m}$",fontsize=fontsize)
    # plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
    #             labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    # plt.xlim(xlim)
    yticks = ax.get_yticks()
    plt.xticks(fontsize=fontsize)
    plt.yticks(yticks[1:],fontsize=fontsize)
    
    ax = fig.add_axes((1.05,-0.32,1,0.3))
    plot_and_fit_var(x_values,x_err,scl_b_std_array,scl_b_std_err_array,top_index,xlim,coeff_guess=10)
    
    # plt.errorbar(x_values,scl_b_std_array,yerr=scl_b_std_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.scatter(x_values[top_index],scl_b_std_array[top_index],color='red',marker='o',zorder=6)
    plt.ylabel(r"$\sigma_{b_{r}}$",fontsize=fontsize)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    # plt.xlim(xlim)
    
    ax = fig.add_axes((1.05,-0.64,1,0.3))
    plot_and_fit_var(x_values,x_err,scl_I_std_array,scl_I_std_err_array,top_index,xlim,coeff_guess=10)
    
    # plt.errorbar(x_values,scl_I_std_array,yerr=scl_I_std_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.scatter(x_values[top_index],scl_I_std_array[top_index],color='red',marker='o',zorder=6)
    plt.ylabel(r"$\sigma_{I_{p}}$",fontsize=fontsize)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    # plt.xlim(xlim)

    ax = fig.add_axes((1.05,-0.96,1,0.3))
    plot_and_fit_var(x_values,x_err,R_std_array,R_std_err_array,top_index,xlim,coeff_guess=10)
    
    scale_factor = 1
    # plt.errorbar(x_values,R_std_array/scale_factor,yerr=R_std_err_array/scale_factor, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.ylabel(r"$\sigma_{R} \times 10^{-5}$",fontsize=fontsize)
    # plt.scatter(x_values[top_index],(R_std_array/scale_factor)[top_index],color='red',marker='o',zorder=6)
    plt.ylabel(r"$\sigma_{R}$",fontsize=fontsize)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    # plt.xlim(xlim)
    
    ax = fig.add_axes((1.05,-1.28,1,0.3))
    plot_and_fit_var(x_values,x_err,std_std_array,std_std_err_array,top_index,xlim,coeff_guess=10)
    
    scale_factor = 1
    # plt.errorbar(x_values,R_std_array/scale_factor,yerr=R_std_err_array/scale_factor, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.ylabel(r"$\sigma_{R} \times 10^{-5}$",fontsize=fontsize)
    # plt.scatter(x_values[top_index],(R_std_array/scale_factor)[top_index],color='red',marker='o',zorder=6)
    plt.ylabel(r"$\sigma_{\sigma_{resid}}$",fontsize=fontsize)
    plt.xlabel(x_label,fontsize=fontsize)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.yticks(fontsize=fontsize)
    xticks = ax.get_xticks()
    plt.xticks(xticks[1:],fontsize=fontsize)
    # plt.xlim(xlim)
    plt.show()
    
    L_R_coeff /= R_array[top_index]
    L_R_coeff_err = np.sqrt(L_R_coeff_err**2 + (L_R_coeff*R_err_array[top_index])**2)/R_array[top_index]
    R_R_coeff /= R_array[top_index]
    R_R_coeff_err = np.sqrt(R_R_coeff_err**2 + (R_R_coeff*R_err_array[top_index])**2)/R_array[top_index]
    L_m_coeff /= avg_m_array[top_index]
    L_m_coeff_err = np.sqrt(L_m_coeff_err**2 + (L_m_coeff*avg_m_err_array[top_index])**2)/avg_m_array[top_index]
    R_m_coeff /= avg_m_array[top_index]
    R_m_coeff_err = np.sqrt(R_m_coeff_err**2 + (R_m_coeff*avg_m_err_array[top_index])**2)/avg_m_array[top_index]
    L_dark_coeff /= rel_darkness_array[top_index]
    L_dark_coeff_err = np.sqrt(L_dark_coeff_err**2 + (L_dark_coeff*rel_darkness_err_array[top_index])**2)/rel_darkness_array[top_index]
    R_dark_coeff /= rel_darkness_array[top_index]
    print(R_dark_coeff_err)
    R_dark_coeff_err =  np.sqrt(R_dark_coeff_err**2 + (R_dark_coeff*rel_darkness_err_array[top_index])**2)/rel_darkness_array[top_index]
    print(R_dark_coeff_err)
    L_I_coeff /= avg_I_peak_array[top_index]
    L_I_coeff_err = np.sqrt(L_I_coeff_err**2 + (L_I_coeff*avg_I_peak_err_array[top_index])**2)/avg_I_peak_array[top_index]
    R_I_coeff /= avg_I_peak_array[top_index]
    R_I_coeff_err = np.sqrt(R_I_coeff_err**2 + (R_I_coeff*avg_I_peak_err_array[top_index])**2)/avg_I_peak_array[top_index]
    L_std_coeff /= rel_avg_std_array[top_index]
    L_std_coeff_err = np.sqrt(L_std_coeff_err**2 + (L_std_coeff*rel_avg_std_err_array[top_index])**2)/rel_avg_std_array[top_index]
    R_std_coeff /= rel_avg_std_array[top_index]
    R_std_coeff_err = np.sqrt(R_std_coeff_err**2 + (R_std_coeff*rel_avg_std_err_array[top_index])**2)/rel_avg_std_array[top_index]
    
else:
    L_R_coeff = np.array([0])
    L_R_coeff_err = np.array([0])
    R_R_coeff = np.array([0])
    R_R_coeff_err = np.array([0])
    L_m_coeff = np.array([0])
    L_m_coeff_err = np.array([0])
    R_m_coeff = np.array([0])
    R_m_coeff_err = np.array([0])
    L_dark_coeff = np.array([0])
    L_dark_coeff_err = np.array([0])
    R_dark_coeff = np.array([0])
    R_dark_coeff_err = np.array([0])
    L_I_coeff = np.array([0])
    L_I_coeff_err = np.array([0])
    R_I_coeff = np.array([0])
    R_I_coeff_err = np.array([0])
    L_std_coeff = np.array([0])
    L_std_coeff_err = np.array([0])
    R_std_coeff = np.array([0])
    R_std_coeff_err = np.array([0])
          

results_folder = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Final_values'

#Create the folder with the final processed images
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

results_file_index = rootdir_unpr_1.find('New_col\\')+8
results_file_name = rootdir_unpr_1[results_file_index:].replace("\\", "-") + '.csv'
print(results_file_name)
results_path = results_folder + '\\' + results_file_name
print(results_path)

# d = {'Radius (mm)':R_array[top_index],'Radius error (mm)':R_err_array[top_index],
#      'Norm. radius std':R_std_array[top_index],'Norm. radius std error':R_std_err_array[top_index],
#      'Left radius variability coeff (/mm)': L_R_coeff, 'Left radius  variability coeff error (/mm)': L_R_coeff_err,
#      'Right radius  variability coeff (/mm)': R_R_coeff, 'Right radius  variability coeff error (/mm)': R_R_coeff_err,
#      'Exponent':avg_m_array[top_index],'Exponent error':avg_m_err_array[top_index],
#      'Exponent std':scl_m_std_array[top_index],'Exponent std error':scl_m_std_err_array[top_index], 
#      'Left exp. variability coeff (/mm)': L_m_coeff, 'Left exp.  variability coeff error (/mm)': L_m_coeff_err,
#      'Right exp.  variability coeff (/mm)': R_m_coeff, 'Right exp.  variability coeff error (/mm)': R_m_coeff_err,
#      'Rel. darkness':rel_darkness_array[top_index],'Rel. darkness err':rel_darkness_err_array[top_index],
#      'Norm. const. std':scl_b_std_array[top_index],'Norm. const. std error':scl_b_std_err_array[top_index],
#      'Left dark. variability coeff (/mm)': L_dark_coeff, 'Left dark.  variability coeff error (/mm)': L_dark_coeff_err,
#      'Right dark.  variability coeff (/mm)': R_dark_coeff, 'Right dark.  variability coeff error (/mm)': R_dark_coeff_err,
#      'Norm. peak intensity':avg_I_peak_array[top_index],'Norm. peak intensity error':avg_I_peak_err_array[top_index],
#      'Norm. peak intensity std':scl_I_std_array[top_index],'Norm. peak intensity std error':scl_I_std_err_array[top_index],
#      'Left peak variability coeff (/mm)': L_I_coeff, 'Left peak  variability coeff error (/mm)': L_I_coeff_err,
#      'Right peak  variability coeff (/mm)': R_I_coeff, 'Right peak  variability coeff error (/mm)': R_I_coeff_err,
#      'Avg. resid. std':avg_std_array[top_index],'Avg. resid. std error':avg_std_err_array[top_index],
#      'Left std variability coeff (/mm)': L_std_coeff, 'Left std  variability coeff error (/mm)': L_std_coeff_err,
#      'Right std  variability coeff (/mm)': R_std_coeff, 'Right std  variability coeff error (/mm)': R_std_coeff_err,
#       }

# R_std_array[top_index] *= R_array[top_index]
# R_std_err_array[top_index] *= R_array[top_index]
# scl_m_std_array = m_std_array/avg_m_array
# scl_m_std_err_array = np.sqrt(m_std_err_array**2 + (scl_m_std_array*avg_m_err_array)**2)/avg_m_array
# scl_b_std_array = b_std_array/avg_b_array
# scl_b_std_err_array = np.sqrt(b_std_err_array**2 + (scl_b_std_array*avg_b_err_array)**2)/avg_b_array
# b_std_array *= rel_darkness_array
# b_std_err_array *= rel_darkness_array

# scl_I_std_array = I_std_array/avg_I_peak_array
# scl_I_std_err_array = np.sqrt(I_std_err_array**2 + (scl_I_std_array*avg_I_peak_err_array)**2)/avg_I_peak_array


if save_results == True:

    d = {'Radius (mm)':R_array[top_index],'Radius error (mm)':R_err_array[top_index],
         'Norm. radius std':R_std_array[top_index],'Norm. radius std error':R_std_err_array[top_index],
         'Left radius variability coeff (/mm)': L_R_coeff, 'Left radius  variability coeff error (/mm)': L_R_coeff_err,
         'Right radius  variability coeff (/mm)': R_R_coeff, 'Right radius  variability coeff error (/mm)': R_R_coeff_err,
         'Exponent':avg_m_array[top_index],'Exponent error':avg_m_err_array[top_index],
         'Exponent std':scl_m_std_array[top_index],'Exponent std error':scl_m_std_err_array[top_index], 
         'Left exp. variability coeff (/mm)': L_m_coeff, 'Left exp.  variability coeff error (/mm)': L_m_coeff_err,
         'Right exp.  variability coeff (/mm)': R_m_coeff, 'Right exp.  variability coeff error (/mm)': R_m_coeff_err,
         'Rel. darkness':rel_darkness_array[top_index],'Rel. darkness err':rel_darkness_err_array[top_index],
         'Norm. const. std':scl_b_std_array[top_index],'Norm. const. std error':scl_b_std_err_array[top_index],
         'Left dark. variability coeff (/mm)': L_dark_coeff, 'Left dark.  variability coeff error (/mm)': L_dark_coeff_err,
         'Right dark.  variability coeff (/mm)': R_dark_coeff, 'Right dark.  variability coeff error (/mm)': R_dark_coeff_err,
         'Norm. peak intensity':avg_I_peak_array[top_index],'Norm. peak intensity error':avg_I_peak_err_array[top_index],
         'Norm. peak intensity std':scl_I_std_array[top_index],'Norm. peak intensity std error':scl_I_std_err_array[top_index],
         'Left peak variability coeff (/mm)': L_I_coeff, 'Left peak  variability coeff error (/mm)': L_I_coeff_err,
         'Right peak  variability coeff (/mm)': R_I_coeff, 'Right peak  variability coeff error (/mm)': R_I_coeff_err,
         'Avg. resid. std':rel_avg_std_array[top_index],'Avg. resid. std error':rel_avg_std_err_array[top_index],
         'Resid std std':scl_std_std_array[top_index], 'Resid std std error':scl_std_std_err_array[top_index],
         'Left std variability coeff (/mm)': L_std_coeff, 'Left std  variability coeff error (/mm)': L_std_coeff_err,
         'Right std  variability coeff (/mm)': R_std_coeff, 'Right std  variability coeff error (/mm)': R_std_coeff_err,
          }
    #'Norm. const.':avg_b_array[top_index],
    # 'Norm. const. error':avg_b_err_array[top_index],
    # 'Total resid. std':total_std_array[top_index],
    # 'Total resid. std error':total_std_err_array[top_index]
    
    dataframe = pd.DataFrame(d, index=[0])
    dataframe.to_csv(results_path)
    