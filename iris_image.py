# -*- coding: utf-8 -*-
"""
Created on Sun May 28 22:17:15 2023

@author: joche
"""
import os
from funcs_photos import *
import pandas as pd
pi = np.pi

fontsize = 18
last_index = 0 #In case any values at the end of the array should not be used, the last index should be set to what will be used or 0 if everything should be.
save_results = False
distances =  True #Set to True if the file contains data at more than 1 distance.
tickwidth = 1.5
ticklength = 4
mtickwidth = 1.5
mticklength = 2

rootdir_unpr_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\New_set\\Iris_range'
dist_err = 0.05
top_index = 4

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

        dist_array, exp_array, R_array, R_err_array, R_std_array, R_std_err_array, x_offset_array, x_offset_err_array, y_offset_array, y_offset_err_array, avg_m_array, avg_m_err_array, m_std_array, m_std_err_array, avg_b_array, avg_b_err_array, b_std_array, b_std_err_array, avg_I_peak_array, avg_I_peak_err_array, I_std_array, I_std_err_array, rel_darkness_array, rel_darkness_err_array, avg_std_array, avg_std_err_array, std_std_array, std_std_err_array, total_std_array, total_std_err_array = csv_file_1.values[:,1:].T

    else:
        #Extract data:
        dist_array, exp_array, R_array, R_err_array, R_std_array, R_std_err_array, x_offset_array, x_offset_err_array, y_offset_array, y_offset_err_array, avg_m_array, avg_m_err_array, m_std_array, m_std_err_array, avg_b_array, avg_b_err_array, b_std_array, b_std_err_array, avg_I_peak_array, avg_I_peak_err_array, I_std_array, I_std_err_array, rel_darkness_array, rel_darkness_err_array, avg_std_array, avg_std_err_array, std_std_array, std_std_err_array, total_std_array, total_std_err_array = csv_file_1.values[:last_index,1:].T

    
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
    
    # x_values -= top_x
    
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
    
    y_values /= top_y
    y_errs /= top_y
    top_y /= top_y
    plt.errorbar(x_values,y_values,yerr=y_errs,xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    ylim = np.asarray(plt.gca().get_ylim())
    plt.ylim(ylim)
    plt.plot([top_x,top_x],ylim,linestyle='--',color='red')
    plt.plot(xlim,[1,1],linestyle='--',color='red')
    # plt.scatter(top_x,top_y,color='red',marker='o',zorder=6)
    # L_x_plot_val = np.asarray([xlim[0],top_x])
    # R_x_plot_val = np.asarray([top_x,xlim[1]])
    # plt.plot(x_values,lin_comb_m(x_values,L_coeff,R_coeff),color='blue',zorder=0)
    # plt.plot(x_values,lin_comb(x_values,L_coeff,R_coeff,const),color='blue',zorder=0)
    # plt.scatter(top_x,const,color='red',marker='x')
    # plt.plot(L_x_plot_val,lin_simp(L_x_plot_val,L_coeff,L_const),color='blue',zorder=0)
    # plt.plot(R_x_plot_val,lin_simp(R_x_plot_val,R_coeff),color='blue',zorder=0)
    plt.xlim(xlim)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', width= tickwidth, length= ticklength)
    plt.tick_params(axis='both', which='minor', width= mtickwidth, length= mticklength)
    


    return L_coeff, L_coeff_err, R_coeff, R_coeff_err#, const, const_err

# x_values = R_array
# x_err = R_err_array
# x_label = "R (mm)"
# xlim = [R_array[0]-0.02,R_array[-1]+0.02]

#Everything below is in microW
P_err = 1.0
ax_1_P = 177.0
fib_tip_P = 183.0
ax_fib_tip_f = ax_1_P/fib_tip_P
ax_fib_tip_f_err = ax_fib_tip_f*np.sqrt((P_err/ax_1_P)**2 + (P_err/fib_tip_P)**2)


multhree_ax_ax1_P = 176.0
two_ax_onemml_P = 170.0
two_ax_onemml_f = two_ax_onemml_P/multhree_ax_ax1_P * ax_fib_tip_f
two_ax_onemml_f_err = two_ax_onemml_f*np.sqrt((P_err/two_ax_onemml_P)**2 + (P_err/multhree_ax_ax1_P)**2
                                                + (ax_fib_tip_f_err/ax_fib_tip_f)**2)

avg_I_peak_err_array = np.sqrt((two_ax_onemml_f*avg_I_peak_err_array)**2 + (two_ax_onemml_f_err*avg_I_peak_array)**2)
avg_I_peak_array *= two_ax_onemml_f


Gauss_beam_intensity = 98.31319273317742
Gauss_beam_err = 0.11716214411854588
avg_I_peak_array /= Gauss_beam_intensity
avg_I_peak_err_array = np.sqrt(avg_I_peak_err_array**2 + (Gauss_beam_err*avg_I_peak_array)**2)/Gauss_beam_intensity

print("Optimum iris diameter: {} +/- {} mm".format(dist_array[top_index],dist_err))

def ratio(a,b,a_err,b_err):
    y = a/b
    y_err = abs(y)*np.sqrt((a_err/a)**2 + (b_err/b)**2)
    return y, y_err

ratio_b
plt.figure()
plt.plot(avg_m_array)
plt.show()

if distances == True:
    x_values = dist_array#-dist_array[top_index]
    x_err = np.sqrt(2)*dist_err
    xlim = [x_values[0]-1,x_values[-1]+1]
    x_label = "d (mm)"
    # x_values = avg_I_peak_array
    # x_err = avg_I_peak_err_array
    # x_label = "I"
    # xlim = [avg_I_peak_array[0]-1,avg_I_peak_array[-1]+1]
    
    #Big combined plot:
    
        
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
    plt.yticks(yticks[1:-1],fontsize=fontsize)
    
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
    L_std_coeff, L_std_coeff_err, R_std_coeff, R_std_coeff_err = plot_and_fit_var(x_values,x_err,rel_avg_std_array,rel_avg_std_err_array,top_index,xlim,coeff_guess=10)
    
    # plt.errorbar(x_values,avg_std_array,yerr=avg_std_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.ylabel(r"$\overline{\sigma_{resid.}}$",fontsize=fontsize)
    # plt.errorbar(x_values,rel_avg_std_array,yerr=rel_avg_std_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
    # plt.scatter(x_values[top_index],rel_avg_std_array[top_index],color='red',marker='o',zorder=6)
    plt.ylabel(r"$\overline{\sigma_{resid.}}$",fontsize=fontsize)
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
    plt.yticks(yticks[1:-1],fontsize=fontsize)
    
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
    
    rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Final_values'
    results_file_name = rootdir + '\\' +  'iris.svg'
    plt.savefig(results_file_name,dpi=300,bbox_inches='tight')
    plt.show()
        
    
    pixel_size = np.average([6.14/1280,4.9/1024,4.8e-3],weights=[0.005/1280,0.05/1024,0.05e-3]) #In mm
    xticks = np.array([0,1.2,2.4,3.6,4.8,6])/pixel_size
    yticks = np.array([0,0.9,1.8,2.7,3.6,4.5])/pixel_size

    path = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\New_set\\Iris_range\\Processed\\Hollow_beam_49996us_097mm.csv'
    pixels = pd.read_csv(path).values[:,1:]
    pixels /= np.amax(pixels)
    fig = plt.figure()
    ax1 = fig.add_axes((0,0,1,1))
    pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',origin='lower',vmin=0,vmax=1)
    cbar = plt.colorbar(mappable=pixel_im)
    cbar.ax.tick_params(labelsize=fontsize, width= tickwidth, length= ticklength)
    # ticklabs = cbar.ax.get_yticklabels()
    # cbar.ax.set_yticklabels(ticklabs,   fontsize=fontsize)
    # plt.colorbar(mappable=pixel_im,label=r'Normalised intensity, $I\;/\;I_{max}$"')
    plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
    plt.xlabel("x (mm)",fontsize=fontsize)
    plt.ylabel("y (mm)",fontsize=fontsize)
    plt.gca().set_aspect('equal')
    plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
    # ax1.tick_params(width=4,length=3)
    plt.tick_params(axis='both', which='major', width= tickwidth, length= ticklength)
    plt.text(0.05, 0.9, '(b)',transform=ax1.transAxes,fontsize=fontsize,color='white')
    
    