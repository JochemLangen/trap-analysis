# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 21:59:08 2023

@author: joche
"""
import os
from funcs_photos import *
import pandas as pd
pi = np.pi

fontsize = 15
dist_err = 0.0005
last_index = -1


#Folder:
rootdir_unpr_1_1 = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flipped_ring_axicon\\Dist_range'
rootdir_1_1 = rootdir_unpr_1_1 + '\\Processed'
#File name with the results from the analysis
file_name_1_1 = rootdir_1_1+'\\'+rootdir_unpr_1_1[rootdir_unpr_1_1.rfind("\\")+1:]+'_analysed.csv'
csv_file_1_1 = pd.read_csv(file_name_1_1)

#Folder:
rootdir_unpr_2_1 = rootdir_unpr_1_1 + '\\Adj_cam'
rootdir_2_1 = rootdir_unpr_2_1 + '\\Processed'
#File name with the results from the analysis
file_name_2_1 = rootdir_2_1+'\\'+rootdir_unpr_2_1[rootdir_unpr_2_1.rfind("\\")+1:]+'_analysed.csv'
csv_file_2_1 = pd.read_csv(file_name_2_1)

if last_index == 0:
    #Extract data:
    dist_array_1_1, exp_array_1_1, R_array_1_1, R_err_array_1_1, R_std_array_1_1, R_std_err_array_1_1, x_offset_array_1_1, x_offset_err_array_1_1, y_offset_array_1_1, y_offset_err_array_1_1, avg_m_array_1_1, avg_m_err_array_1_1, m_std_array_1_1, m_std_err_array_1_1, avg_b_array_1_1, avg_b_err_array_1_1, b_std_array_1_1, b_std_err_array_1_1, avg_I_peak_array_1_1, avg_I_peak_err_array_1_1, I_std_array_1_1, I_std_err_array_1_1, rel_darkness_array_1_1, rel_darkness_err_array_1_1, avg_std_array_1_1, avg_std_err_array_1_1, total_std_array_1_1, total_std_err_array_1_1 = csv_file_1_1.values[:,1:].T

    #Extract data:
    dist_array_2_1, exp_array_2_1, R_array_2_1, R_err_array_2_1, R_std_array_2_1, R_std_err_array_2_1, x_offset_array_2_1, x_offset_err_array_2_1, y_offset_array_2_1, y_offset_err_array_2_1, avg_m_array_2_1, avg_m_err_array_2_1, m_std_array_2_1, m_std_err_array_2_1, avg_b_array_2_1, avg_b_err_array_2_1, b_std_array_2_1, b_std_err_array_2_1, avg_I_peak_array_2_1, avg_I_peak_err_array_2_1, I_std_array_2_1, I_std_err_array_2_1, rel_darkness_array_2_1, rel_darkness_err_array_2_1, avg_std_array_2_1, avg_std_err_array_2, total_std_array_2, total_std_err_array_2 = csv_file_2_1.values[:,1:].T
else:
    #Extract data:
    dist_array_1_1, exp_array_1_1, R_array_1_1, R_err_array_1_1, R_std_array_1_1, R_std_err_array_1_1, x_offset_array_1_1, x_offset_err_array_1_1, y_offset_array_1_1, y_offset_err_array_1_1, avg_m_array_1_1, avg_m_err_array_1_1, m_std_array_1_1, m_std_err_array_1_1, avg_b_array_1_1, avg_b_err_array_1_1, b_std_array_1_1, b_std_err_array_1_1, avg_I_peak_array_1_1, avg_I_peak_err_array_1_1, I_std_array_1_1, I_std_err_array_1_1, rel_darkness_array_1_1, rel_darkness_err_array_1_1, avg_std_array_1_1, avg_std_err_array_1_1, total_std_array_1_1, total_std_err_array_1_1 = csv_file_1_1.values[:last_index,1:].T

    #Extract data:
    dist_array_2_1, exp_array_2_1, R_array_2_1, R_err_array_2_1, R_std_array_2_1, R_std_err_array_2_1, x_offset_array_2_1, x_offset_err_array_2_1, y_offset_array_2_1, y_offset_err_array_2_1, avg_m_array_2_1, avg_m_err_array_2_1, m_std_array_2_1, m_std_err_array_2_1, avg_b_array_2_1, avg_b_err_array_2_1, b_std_array_2_1, b_std_err_array_2_1, avg_I_peak_array_2_1, avg_I_peak_err_array_2_1, I_std_array_2_1, I_std_err_array_2_1, rel_darkness_array_2_1, rel_darkness_err_array_2_1, avg_std_array_2_1, avg_std_err_array_2_1, total_std_array_2_1, total_std_err_array_2_1 = csv_file_2_1.values[:last_index,1:].T


#Folder:
rootdir_unpr_1_2 = rootdir_unpr_1_1 + '_2'
rootdir_1_2 = rootdir_unpr_1_2 + '\\Processed'
#File name with the results from the analysis
file_name_1_2 = rootdir_1_2+'\\'+rootdir_unpr_1_2[rootdir_unpr_1_2.rfind("\\")+1:]+'_analysed.csv'
csv_file_1_2 = pd.read_csv(file_name_1_2)

#Folder:
rootdir_unpr_2_2 = rootdir_unpr_1_2 + '\\Adj_cam'
rootdir_2_2 = rootdir_unpr_2_2 + '\\Processed'
#File name with the results from the analysis
file_name_2_2 = rootdir_2_2+'\\'+rootdir_unpr_2_2[rootdir_unpr_2_2.rfind("\\")+1:]+'_analysed.csv'
csv_file_2_2 = pd.read_csv(file_name_2_2)

if last_index == 0:
    #Extract data:
    dist_array_1_2, exp_array_1_2, R_array_1_2, R_err_array_1_2, R_std_array_1_2, R_std_err_array_1_2, x_offset_array_1_2, x_offset_err_array_1_2, y_offset_array_1_2, y_offset_err_array_1_2, avg_m_array_1_2, avg_m_err_array_1_2, m_std_array_1_2, m_std_err_array_1_2, avg_b_array_1_2, avg_b_err_array_1_2, b_std_array_1_2, b_std_err_array_1_2, avg_I_peak_array_1_2, avg_I_peak_err_array_1_2, I_std_array_1_2, I_std_err_array_1_2, rel_darkness_array_1_2, rel_darkness_err_array_1_2, avg_std_array_1_2, avg_std_err_array_1_2, total_std_array_1_2, total_std_err_array_1_2 = csv_file_1_2.values[:,1:].T

    #Extract data:
    dist_array_2_2, exp_array_2_2, R_array_2_2, R_err_array_2_2, R_std_array_2_2, R_std_err_array_2_2, x_offset_array_2_2, x_offset_err_array_2_2, y_offset_array_2_2, y_offset_err_array_2_2, avg_m_array_2_2, avg_m_err_array_2_2, m_std_array_2_2, m_std_err_array_2_2, avg_b_array_2_2, avg_b_err_array_2_2, b_std_array_2_2, b_std_err_array_2_2, avg_I_peak_array_2_2, avg_I_peak_err_array_2_2, I_std_array_2_2, I_std_err_array_2_2, rel_darkness_array_2_2, rel_darkness_err_array_2_2, avg_std_array_2_2, avg_std_err_array_2, total_std_array_2, total_std_err_array_2 = csv_file_2_2.values[:,1:].T
else:
    #Extract data:
    dist_array_1_2, exp_array_1_2, R_array_1_2, R_err_array_1_2, R_std_array_1_2, R_std_err_array_1_2, x_offset_array_1_2, x_offset_err_array_1_2, y_offset_array_1_2, y_offset_err_array_1_2, avg_m_array_1_2, avg_m_err_array_1_2, m_std_array_1_2, m_std_err_array_1_2, avg_b_array_1_2, avg_b_err_array_1_2, b_std_array_1_2, b_std_err_array_1_2, avg_I_peak_array_1_2, avg_I_peak_err_array_1_2, I_std_array_1_2, I_std_err_array_1_2, rel_darkness_array_1_2, rel_darkness_err_array_1_2, avg_std_array_1_2, avg_std_err_array_1_2, total_std_array_1_2, total_std_err_array_1_2 = csv_file_1_2.values[:last_index,1:].T

    #Extract data:
    dist_array_2_2, exp_array_2_2, R_array_2_2, R_err_array_2_2, R_std_array_2_2, R_std_err_array_2_2, x_offset_array_2_2, x_offset_err_array_2_2, y_offset_array_2_2, y_offset_err_array_2_2, avg_m_array_2_2, avg_m_err_array_2_2, m_std_array_2_2, m_std_err_array_2_2, avg_b_array_2_2, avg_b_err_array_2_2, b_std_array_2_2, b_std_err_array_2_2, avg_I_peak_array_2_2, avg_I_peak_err_array_2_2, I_std_array_2_2, I_std_err_array_2_2, rel_darkness_array_2_2, rel_darkness_err_array_2_2, avg_std_array_2_2, avg_std_err_array_2_2, total_std_array_2_2, total_std_err_array_2_2 = csv_file_2_2.values[:last_index,1:].T


#Append two data sets
dist_array_1 = np.append(dist_array_1_2,dist_array_1_1)
exp_array_1 = np.append(exp_array_1_2,exp_array_1_1) 
R_array_1 = np.append(R_array_1_2,R_array_1_1) 
R_err_array_1 = np.append(R_err_array_1_2,R_err_array_1_1)
R_std_array_1 = np.append(R_std_array_1_2,R_std_array_1_1)
R_std_err_array_1 = np.append(R_std_err_array_1_2,R_std_err_array_1_1)
avg_m_array_1 = np.append(avg_m_array_1_2,avg_m_array_1_1)
avg_m_err_array_1 = np.append(avg_m_err_array_1_2,avg_m_err_array_1_1)
m_std_array_1 = np.append(m_std_array_1_2,m_std_array_1_1) 
m_std_err_array_1 = np.append(m_std_err_array_1_2,m_std_err_array_1_1)
avg_b_array_1 = np.append(avg_b_array_1_2,avg_b_array_1_1)
avg_b_err_array_1 = np.append(avg_b_err_array_1_2,avg_b_err_array_1_1)
b_std_array_1 = np.append(b_std_array_1_2,b_std_array_1_1)
b_std_err_array_1 = np.append(b_std_err_array_1_2,b_std_err_array_1_1)
avg_I_peak_array_1 = np.append(avg_I_peak_array_1_2,avg_I_peak_array_1_1) 
avg_I_peak_err_array_1 = np.append(avg_I_peak_err_array_1_2,avg_I_peak_err_array_1_1) 
I_std_array_1 = np.append(I_std_array_1_2,I_std_array_1_1) 
I_std_err_array_1 = np.append(I_std_err_array_1_2,I_std_err_array_1_1) 
rel_darkness_array_1 = np.append(rel_darkness_array_1_2,rel_darkness_array_1_1) 
rel_darkness_err_array_1 = np.append(rel_darkness_err_array_1_2,rel_darkness_err_array_1_1) 
avg_std_array_1 = np.append(avg_std_array_1_2,avg_std_array_1_1)
avg_std_err_array_1 = np.append(avg_std_err_array_1_2,avg_std_err_array_1_1) 
total_std_array_1 = np.append(total_std_array_1_2,total_std_array_1_1)
total_std_err_array_1 = np.append(total_std_err_array_1_2,total_std_err_array_1_1)

dist_array_2 = np.append(dist_array_2_2,dist_array_2_1)
exp_array_2 = np.append(exp_array_2_2,exp_array_2_1) 
R_array_2 = np.append(R_array_2_2,R_array_2_1) 
R_err_array_2 = np.append(R_err_array_2_2,R_err_array_2_1)
R_std_array_2 = np.append(R_std_array_2_2,R_std_array_2_1)
R_std_err_array_2 = np.append(R_std_err_array_2_2,R_std_err_array_2_1)
avg_m_array_2 = np.append(avg_m_array_2_2,avg_m_array_2_1)
avg_m_err_array_2 = np.append(avg_m_err_array_2_2,avg_m_err_array_2_1)
m_std_array_2 = np.append(m_std_array_2_2,m_std_array_2_1) 
m_std_err_array_2 = np.append(m_std_err_array_2_2,m_std_err_array_2_1)
avg_b_array_2 = np.append(avg_b_array_2_2,avg_b_array_2_1)
avg_b_err_array_2 = np.append(avg_b_err_array_2_2,avg_b_err_array_2_1)
b_std_array_2 = np.append(b_std_array_2_2,b_std_array_2_1)
b_std_err_array_2 = np.append(b_std_err_array_2_2,b_std_err_array_2_1)
avg_I_peak_array_2 = np.append(avg_I_peak_array_2_2,avg_I_peak_array_2_1) 
avg_I_peak_err_array_2 = np.append(avg_I_peak_err_array_2_2,avg_I_peak_err_array_2_1) 
I_std_array_2 = np.append(I_std_array_2_2,I_std_array_2_1) 
I_std_err_array_2 = np.append(I_std_err_array_2_2,I_std_err_array_2_1) 
rel_darkness_array_2 = np.append(rel_darkness_array_2_2,rel_darkness_array_2_1) 
rel_darkness_err_array_2 = np.append(rel_darkness_err_array_2_2,rel_darkness_err_array_2_1) 
avg_std_array_2 = np.append(avg_std_array_2_2,avg_std_array_2_1)
avg_std_err_array_2 = np.append(avg_std_err_array_2_2,avg_std_err_array_2_1) 
total_std_array_2 = np.append(total_std_array_2_2,total_std_array_2_1)
total_std_err_array_2 = np.append(total_std_err_array_2_2,total_std_err_array_2_1)

#Combine data:
R_array, R_err_array = weighted_avg_2D_0(np.asarray([R_array_1,R_array_2]),np.asarray([R_err_array_1,R_err_array_2]))
R_std_array, R_std_err_array = weighted_avg_2D_0(np.asarray([R_std_array_1,R_std_array_2]),np.asarray([R_std_err_array_1,R_std_err_array_2]))
avg_m_array, avg_m_err_array = weighted_avg_2D_0(np.asarray([avg_m_array_1,avg_m_array_2]),np.asarray([avg_m_err_array_1,avg_m_err_array_2]))
m_std_array, m_std_err_array = weighted_avg_2D_0(np.asarray([m_std_array_1,m_std_array_2]),np.asarray([m_std_err_array_1,m_std_err_array_2]))
avg_b_array, avg_b_err_array = weighted_avg_2D_0(np.asarray([avg_b_array_1,avg_b_array_2]),np.asarray([avg_b_err_array_1,avg_b_err_array_2]))
b_std_array, b_std_err_array = weighted_avg_2D_0(np.asarray([b_std_array_1,b_std_array_2]),np.asarray([b_std_err_array_1,b_std_err_array_2]))
avg_I_peak_array, avg_I_peak_err_array = weighted_avg_2D_0(np.asarray([avg_I_peak_array_1,avg_I_peak_array_2]),np.asarray([avg_I_peak_err_array_1,avg_I_peak_err_array_2]))
I_std_array, I_std_err_array = weighted_avg_2D_0(np.asarray([I_std_array_1,I_std_array_2]),np.asarray([I_std_err_array_1,I_std_err_array_2]))
rel_darkness_array, rel_darkness_err_array = weighted_avg_2D_0(np.asarray([rel_darkness_array_1,rel_darkness_array_2]),np.asarray([rel_darkness_err_array_1,rel_darkness_err_array_2]))
avg_std_array, avg_std_err_array = weighted_avg_2D_0(np.asarray([avg_std_array_1,avg_std_array_2]),np.asarray([avg_std_err_array_1,avg_std_err_array_2]))
total_std_array, total_std_err_array = weighted_avg_2D_0(np.asarray([total_std_array_1,total_std_array_2]),np.asarray([total_std_err_array_1,total_std_err_array_2]))


#R is already scaled

scl_m_std_array = m_std_array/avg_m_array
scl_m_std_err_array = np.sqrt(m_std_err_array**2 + (scl_m_std_array*avg_m_err_array)**2)/avg_m_array
scl_b_std_array = b_std_array/avg_b_array
scl_b_std_err_array = np.sqrt(b_std_err_array**2 + (scl_b_std_array*avg_b_err_array)**2)/avg_b_array
scl_I_std_array = I_std_array/avg_I_peak_array
scl_I_std_err_array = np.sqrt(I_std_err_array**2 + (scl_I_std_array*avg_I_peak_err_array)**2)/avg_I_peak_array


pixel_size = np.average([6.14/1280,4.9/1024,4.8e-3],weights=[0.005/1280,0.05/1024,0.05e-3]) #In mm
pixel_size_err = 1/np.sqrt(1/(0.005/1280)**2 + 1/(0.05/1024)**2 + 1/(0.05e-3)**2)
R_array *= pixel_size
R_err_array = np.sqrt((R_err_array*pixel_size_err)**2 + (pixel_size_err*R_array/pixel_size)**2)

rel_avg_std_array = avg_std_array/avg_I_peak_array*100
rel_avg_std_err_array = np.sqrt(avg_std_err_array**2+(avg_I_peak_err_array*rel_avg_std_array)**2)/avg_I_peak_array




x_values = dist_array_1
x_err = dist_err
x_label = "d (mm)"
xlim = [dist_array_1[0]-1,dist_array_1[-1]+1]

# x_values = avg_I_peak_array
# x_err = avg_I_peak_err_array
# x_label = "I"
# xlim = [avg_I_peak_array[0]-1,avg_I_peak_array[-1]+1]

#Big combined plot:

    
fig = plt.figure()

ax = fig.add_axes((0,0,1,0.3))
plt.errorbar(x_values,avg_m_array,yerr=avg_m_err_array,xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel(r"$\overline{m}$",fontsize=fontsize)
# plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
#             labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
ax.xaxis.tick_top()
plt.xlim(xlim)
yticks = ax.get_yticks()
plt.xticks(fontsize=fontsize)
plt.yticks(yticks[1:],fontsize=fontsize)

# ax = fig.add_axes((0,-0.3,1,0.3))
# plt.errorbar(x_values,avg_b_array,yerr=avg_b_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
# plt.ylabel("Constant",fontsize=fontsize,labelpad=4)
# plt.xticks([])
# plt.xlim(xlim)

ax = fig.add_axes((0,-0.32,1,0.3))
plt.errorbar(x_values,rel_darkness_array,yerr=rel_darkness_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel(r"$\overline{b_{r}}$ (%)",fontsize=fontsize)
plt.xticks([])
plt.yticks(fontsize=fontsize)
plt.xlim(xlim)

ax = fig.add_axes((0,-0.64,1,0.3))
plt.errorbar(x_values,avg_I_peak_array,yerr=avg_I_peak_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel(r"$\overline{I_{p}}$",fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks([])
plt.xlim(xlim)

ax = fig.add_axes((0,-0.96,1,0.3))
plt.errorbar(x_values,R_array,yerr=R_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel(r"$R$ (mm)",fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks([])
plt.xlim(xlim)

ax = fig.add_axes((0,-1.28,1,0.3))
# plt.errorbar(x_values,avg_std_array,yerr=avg_std_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
# plt.ylabel(r"$\overline{\sigma_{resid.}}$",fontsize=fontsize)
plt.errorbar(x_values,rel_avg_std_array,yerr=rel_avg_std_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel(r"$\overline{\sigma_{resid.,r}}$ (%)",fontsize=fontsize)
# plt.errorbar(x_values,total_std_array,yerr=total_std_err_array, xerr=x_err,marker='x',color='blue',capsize=2,linestyle='',zorder=5)
plt.xlabel(x_label,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.xlim(xlim)


#OTher side
ax = fig.add_axes((1.05,0,1,0.3))
plt.errorbar(x_values,scl_m_std_array,yerr=scl_m_std_err_array,xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel(r"$\sigma_{m}$",fontsize=fontsize)
# plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
#             labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
ax.xaxis.tick_top()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.xlim(xlim)
yticks = ax.get_yticks()
plt.xticks(fontsize=fontsize)
plt.yticks(yticks[1:],fontsize=fontsize)

ax = fig.add_axes((1.05,-0.32,1,0.3))
plt.errorbar(x_values,scl_b_std_array,yerr=scl_b_std_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel(r"$\sigma_{b_{r}}$",fontsize=fontsize)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.yticks(fontsize=fontsize)
plt.xticks([])
plt.xlim(xlim)

ax = fig.add_axes((1.05,-0.64,1,0.3))
plt.errorbar(x_values,scl_I_std_array,yerr=scl_I_std_err_array, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel(r"$\sigma_{I_{p}}$",fontsize=fontsize)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.yticks(fontsize=fontsize)
plt.xticks([])
plt.xlim(xlim)

ax = fig.add_axes((1.05,-0.96,1,0.3))
scale_factor = 1
plt.errorbar(x_values,R_std_array/scale_factor,yerr=R_std_err_array/scale_factor, xerr=x_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
# plt.ylabel(r"$\sigma_{R} \times 10^{-5}$",fontsize=fontsize)
plt.ylabel(r"$\sigma_{R}$",fontsize=fontsize)
plt.xlabel(x_label,fontsize=fontsize)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.xlim(xlim)
plt.show()

    