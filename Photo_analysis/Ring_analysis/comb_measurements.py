# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:32:10 2023

@author: jochem langen
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
rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\Dist_range'
top_index = 12 #The index of the main image that is used to extract the parameters from / around
last_index = -1 #The last index of the analysed images, can be used to set the range of images looked at


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

#Generate relative avg std values
rel_avg_std_array = avg_std_array/avg_I_peak_array
rel_avg_std_err_array = np.sqrt(avg_std_err_array**2+(avg_I_peak_err_array*rel_avg_std_array)**2)/avg_I_peak_array

#Obtain fractional darkness
rel_darkness_array /= 100
rel_darkness_err_array /= 100

def chi_sqrd_red(y_fit,y,y_err,v):
    return np.sum(((y_fit - y)/y_err)**2)/v

def plot_and_fit_var(x_values,x_err,y_values,y_errs,top_index,xlim,coeff_guess=10):

    top_y = y_values[top_index]
    top_y_err = y_errs[top_index]
    top_x = x_values[top_index]
    
    x_values -= top_x
    
    def lin_simp(x,coeff,const):
        return coeff*x + const
    
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
    
    plt.errorbar(x_values,y_values,yerr=y_errs,xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    plt.scatter(top_x,top_y,color='red',marker='o',zorder=6)
    plt.plot(x_values,lin_comb_m(x_values,L_coeff,R_coeff),color='blue',zorder=0)
    plt.xlim(xlim)


    return L_coeff, L_coeff_err, R_coeff, R_coeff_err#, const, const_err


if distances == True:
    x_values = dist_array-dist_array[top_index]
    x_err = np.sqrt(2)*dist_err
    xlim = [x_values[0]-1,x_values[-1]+1]
    x_label = "d (mm)"

    #Big combined plot:
    print(dist_array[top_index])
    
        
    fig = plt.figure()
    
    ax = fig.add_axes((0,0,1,0.3))
    L_m_coeff, L_m_coeff_err, R_m_coeff, R_m_coeff_err = plot_and_fit_var(x_values,x_err,avg_m_array,avg_m_err_array,top_index,xlim,coeff_guess=10)
    plt.ylabel(r"$\overline{m}$",fontsize=fontsize)
    ax.xaxis.tick_top()
    yticks = ax.get_yticks()
    plt.xticks(fontsize=fontsize)
    plt.yticks(yticks[1:],fontsize=fontsize)
    
    ax = fig.add_axes((0,-0.32,1,0.3))
    L_dark_coeff, L_dark_coeff_err, R_dark_coeff, R_dark_coeff_err = plot_and_fit_var(x_values,x_err,rel_darkness_array,rel_darkness_err_array,top_index,xlim,coeff_guess=10)
    plt.ylabel(r"$\overline{b_{r}}$",fontsize=fontsize)
    plt.xticks([])
    plt.yticks(fontsize=fontsize)
    
    ax = fig.add_axes((0,-0.64,1,0.3))
    L_I_coeff, L_I_coeff_err, R_I_coeff, R_I_coeff_err = plot_and_fit_var(x_values,x_err,avg_I_peak_array,avg_I_peak_err_array,top_index,xlim,coeff_guess=10)
    
    plt.ylabel(r"$\overline{I_{p}}$",fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    
    ax = fig.add_axes((0,-0.96,1,0.3))
    L_R_coeff, L_R_coeff_err, R_R_coeff, R_R_coeff_err = plot_and_fit_var(x_values,x_err,R_array,R_err_array,top_index,xlim,coeff_guess=10)
    
    plt.ylabel(r"$R$ (mm)",fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    
    ax = fig.add_axes((0,-1.28,1,0.3))
    L_std_coeff, L_std_coeff_err, R_std_coeff, R_std_coeff_err = plot_and_fit_var(x_values,x_err,rel_avg_std_array,rel_avg_std_err_array,top_index,xlim,coeff_guess=10)
    plt.ylabel(r"$\overline{\sigma_{resid.,r}}$",fontsize=fontsize)
    plt.xlabel(x_label,fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    
    
    #OTher side
    ax = fig.add_axes((1.05,0,1,0.3))
    plot_and_fit_var(x_values,x_err,scl_m_std_array,scl_m_std_err_array,top_index,xlim,coeff_guess=10)
    
    plt.ylabel(r"$\sigma_{m}$",fontsize=fontsize)
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    yticks = ax.get_yticks()
    plt.xticks(fontsize=fontsize)
    plt.yticks(yticks[1:],fontsize=fontsize)
    
    ax = fig.add_axes((1.05,-0.32,1,0.3))
    plot_and_fit_var(x_values,x_err,scl_b_std_array,scl_b_std_err_array,top_index,xlim,coeff_guess=10)
    
    plt.ylabel(r"$\sigma_{b_{r}}$",fontsize=fontsize)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    
    ax = fig.add_axes((1.05,-0.64,1,0.3))
    plot_and_fit_var(x_values,x_err,scl_I_std_array,scl_I_std_err_array,top_index,xlim,coeff_guess=10)
    plt.ylabel(r"$\sigma_{I_{p}}$",fontsize=fontsize)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    
    ax = fig.add_axes((1.05,-0.96,1,0.3))
    plot_and_fit_var(x_values,x_err,R_std_array,R_std_err_array,top_index,xlim,coeff_guess=10)
    
    scale_factor = 1
    plt.ylabel(r"$\sigma_{R}$",fontsize=fontsize)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    
    ax = fig.add_axes((1.05,-1.28,1,0.3))
    plot_and_fit_var(x_values,x_err,std_std_array,std_std_err_array,top_index,xlim,coeff_guess=10)
    
    scale_factor = 1
    plt.ylabel(r"$\sigma_{\sigma_{resid}}$",fontsize=fontsize)
    plt.xlabel(x_label,fontsize=fontsize)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.yticks(fontsize=fontsize)
    xticks = ax.get_xticks()
    plt.xticks(xticks[1:],fontsize=fontsize)
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
    R_dark_coeff_err =  np.sqrt(R_dark_coeff_err**2 + (R_dark_coeff*rel_darkness_err_array[top_index])**2)/rel_darkness_array[top_index]
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

    dataframe = pd.DataFrame(d, index=[0])
    dataframe.to_csv(results_path)
    