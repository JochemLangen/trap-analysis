# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:43:20 2023

@author: joche
"""
import os
from funcs_photos import *
import pandas as pd
pi = np.pi

rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Final_values'
fontsize=18
tickwidth = 1.5
ticklength = 4
mtickwidth = 1.5
mticklength = 2
plot = True

def master_plot_part(R_array,R_err_array,R_std_array,R_std_err_array,L_R_coeff,L_R_coeff_err,R_R_coeff,R_R_coeff_err,markers,pos,size,fontsize,xlabel,letter=None,ylabels=None,scale_factor=1,legend=False,log_plot=''):
    ax = fig.add_axes((pos[0],pos[1],size[0],size[1]))
    no_files = len(R_array)
    for i in range(no_files):
        plt.errorbar(R_array[i],R_std_array[i]*scale_factor,yerr=R_std_err_array[i]*scale_factor, xerr=R_err_array[i],marker=markers[i],capsize=2,linestyle='',zorder=5,label=labels[i])
    
    xlim = np.asarray(ax.get_xlim())
    # plt.yscale('log')
    # xticks = np.round(np.linspace(xlim[0],xlim[1],8),1)
    plt.ylabel(ylabels[0],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    if log_plot == 'x':
        plt.xscale('log')
        # xlim[0] = 5*10**-3
    elif log_plot == 'y':
        plt.yscale('log')
    elif log_plot == 'xy':
        plt.xscale('log')
        plt.yscale('log')
        xlim[0]=8*10**-4
    plt.xlim(xlim)
    if legend == True:
        plt.legend(bbox_to_anchor=(1.1, 2.5),fontsize=fontsize)
    plt.tick_params(axis='both', which='major', width= tickwidth, length= ticklength)
    plt.tick_params(axis='both', which='minor', width= mtickwidth, length= mticklength)
    
    ax = fig.add_axes((pos[0],pos[1]+size[1]+0.02,size[0],size[1]*1.2))
    
    L_R_coeff *= -1 #To describe the gradient on both sides of the point of interest in both directions
    for i in range(no_files):
        if L_R_coeff_err[i] == 0:
            continue
        if labels[i] == '2 Axicons - 100mm lens':
            R_coeff = R_R_coeff[i].copy()
            R_R_coeff[i] = L_R_coeff[i].copy()
            L_R_coeff[i] = R_coeff
            #For this set-up, the translation stage was placed the other way around

        plt.errorbar(R_R_coeff[i]*scale_factor,L_R_coeff[i]*scale_factor,yerr=L_R_coeff_err[i]*scale_factor,xerr=R_R_coeff_err[i]*scale_factor,marker=markers[i],capsize=2,linestyle='',zorder=5,label=labels[i])
    
    xlim = np.asarray(ax.get_xlim())
    ylim = ax.get_ylim()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.ylabel(ylabels[1],fontsize=fontsize)
    plt.xlabel(ylabels[2],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    xlim[0] = 0.04*(xlim[0]//0.04)
    xlim[1] = 0.04*(xlim[1]//0.04+1)
    # xlim[1] = 0.01*(xlim[1]//0.01)
    
    plt.xticks(np.linspace(xlim[0],xlim[1],5),fontsize=fontsize)
    # xlim[0] = np.amin(np.append(L_R_coeff,xlim[0]))
    # xlim[1] = np.amax(np.append(L_R_coeff,xlim[1]))
    # if log_plot == 'xy':
    #     xlim[0] = -0.08
    plt.xlim(xlim)
    
    # plt.xticks(np.linspace(0.01*(xlim[0]//0.01),0.01*(xlim[-1]//0.01),5),fontsize=fontsize)
    
    plt.plot([xlim[0],xlim[1]],[0,0],color='grey',linestyle='--',zorder=0)
    plt.plot([0,0],[ylim[0],ylim[1]],color='grey',linestyle='--',zorder=0)
    
    
    
    plt.ylim(ylim)
    plt.tick_params(axis='both', which='major', width= tickwidth, length= ticklength)
    plt.text(0.895, 0.83, letter,transform=ax.transAxes,fontsize=fontsize)
    # for i in range(no_files):
    #     if L_R_coeff_err[i] == 0:
    #         continue
    #     plt.errorbar(R_array[i],L_R_coeff[i]*scale_factor,yerr=L_R_coeff_err[i]*scale_factor, xerr=R_err_array[i],marker=markers[i],capsize=2,linestyle='',zorder=5,label=labels[i])
    
    # plt.plot([xlim[0],xlim[1]],[0,0],color='grey',linestyle='--',zorder=0)
    # plt.ylabel(ylabels[1],fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # # ax.yaxis.tick_right()
    # # ax.yaxis.set_label_position("right")
    # # plt.xscale('log')
    # plt.xticks([])
    # plt.xlim(xlim)
    
    
    # ax = fig.add_axes((pos[0],pos[1]+size[1]*2,size[0],size[1]))
    # for i in range(no_files):
    #     if R_R_coeff_err[i] == 0:
    #         continue
    #     plt.errorbar(R_array[i],R_R_coeff[i]*scale_factor,yerr=R_R_coeff_err[i]*scale_factor, xerr=R_err_array[i],marker=markers[i],capsize=2,linestyle='',zorder=5,label=labels[i])
        
    # plt.plot([xlim[0],xlim[1]],[0,0],color='grey',linestyle='--',zorder=0)
    # plt.ylabel(ylabels[2],fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # # plt.xscale('log')
    # plt.xticks([])
    # plt.xlim(xlim)
    return

def simpl_master_plot(R_array,R_err_array,L_R_coeff,L_R_coeff_err,R_R_coeff,R_R_coeff_err,markers,pos,size,fontsize,xlabel,ylabels=None,scale_factor=1):
    ax = fig.add_axes((pos[0],pos[1],size[0],size[1]))
    no_files = len(R_array)
    for i in range(no_files):
        if L_R_coeff_err[i] == 0:
            continue
        plt.errorbar(R_array[i],L_R_coeff[i]*scale_factor,yerr=L_R_coeff_err[i]*scale_factor, xerr=R_err_array[i],marker=markers[i],capsize=2,linestyle='',zorder=5,label=labels[i])
    
    xlim = ax.get_xlim()
    plt.plot([xlim[0],xlim[1]],[0,0],color='grey',linestyle='--',zorder=0)
    xticks = np.round(np.linspace(xlim[0],xlim[1],6),3)
    plt.ylabel(ylabels[0],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(xticks,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.xlim(xlim)
    
    ax = fig.add_axes((pos[0],pos[1]+size[1],size[0],size[1]))
    for i in range(no_files):
        if R_R_coeff_err[i] == 0:
            continue
        plt.errorbar(R_array[i],R_R_coeff[i]*scale_factor,yerr=R_R_coeff_err[i]*scale_factor, xerr=R_err_array[i],marker=markers[i],capsize=2,linestyle='',zorder=5,label=labels[i])
        
    plt.plot([xlim[0],xlim[1]],[0,0],color='grey',linestyle='--',zorder=0)
    plt.ylabel(ylabels[1],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    plt.xlim(xlim)
    return


for subdir, dirs, files in os.walk(rootdir):
    directory_name = subdir[subdir[:-1].rfind("\\")+1:]
    no_files = len(files)-2
    R_array = np.empty(no_files)
    R_err_array = np.empty(no_files)
    R_std_array = np.empty(no_files)
    R_std_err_array = np.empty(no_files)
    L_R_coeff = np.empty(no_files)
    L_R_coeff_err = np.empty(no_files)
    R_R_coeff = np.empty(no_files)
    R_R_coeff_err = np.empty(no_files)
    avg_m_array = np.empty(no_files)
    avg_m_err_array = np.empty(no_files)
    scl_m_std_array = np.empty(no_files)
    scl_m_std_err_array = np.empty(no_files)
    L_m_coeff = np.empty(no_files)
    L_m_coeff_err = np.empty(no_files)
    R_m_coeff = np.empty(no_files)
    R_m_coeff_err = np.empty(no_files)
    rel_darkness_array = np.empty(no_files)
    rel_darkness_err_array = np.empty(no_files)
    scl_b_std_array = np.empty(no_files)
    scl_b_std_err_array = np.empty(no_files)
    L_dark_coeff = np.empty(no_files)
    L_dark_coeff_err = np.empty(no_files)
    R_dark_coeff = np.empty(no_files)
    R_dark_coeff_err = np.empty(no_files)
    avg_I_peak_array = np.empty(no_files)
    avg_I_peak_err_array = np.empty(no_files)
    scl_I_std_array = np.empty(no_files)
    scl_I_std_err_array = np.empty(no_files)
    L_I_coeff = np.empty(no_files)
    L_I_coeff_err = np.empty(no_files)
    R_I_coeff = np.empty(no_files)
    R_I_coeff_err = np.empty(no_files)
    rel_avg_std_array = np.empty(no_files)
    rel_avg_std_err_array = np.empty(no_files)
    L_std_coeff = np.empty(no_files)
    L_std_coeff_err = np.empty(no_files)
    R_std_coeff = np.empty(no_files)
    R_std_coeff_err = np.empty(no_files)
    scl_std_std_array = np.empty(no_files)
    scl_std_std_err_array = np.empty(no_files)
    
    i = 0
    for file in files:
        path = os.path.join(subdir,file)
        print(path)
        if path[-3:] == 'csv':
            R_array[i], R_err_array[i], R_std_array[i], R_std_err_array[i], L_R_coeff[i], L_R_coeff_err[i], R_R_coeff[i], R_R_coeff_err[i], avg_m_array[i], avg_m_err_array[i], scl_m_std_array[i], scl_m_std_err_array[i], L_m_coeff[i], L_m_coeff_err[i], R_m_coeff[i], R_m_coeff_err[i], rel_darkness_array[i], rel_darkness_err_array[i], scl_b_std_array[i], scl_b_std_err_array[i], L_dark_coeff[i], L_dark_coeff_err[i], R_dark_coeff[i], R_dark_coeff_err[i], avg_I_peak_array[i], avg_I_peak_err_array[i], scl_I_std_array[i], scl_I_std_err_array[i], L_I_coeff[i],     L_I_coeff_err[i],     R_I_coeff[i],     R_I_coeff_err[i],    rel_avg_std_array[i],     rel_avg_std_err_array[i],    scl_std_std_array[i], scl_std_std_err_array[i], L_std_coeff[i],     L_std_coeff_err[i],     R_std_coeff[i],    R_std_coeff_err[i] = pd.read_csv(path).values[:,1:].T
            i += 1

markers = np.array(['x','o','v','^','<','>','s','P'])
labels = np.array(['2 Axicons', '3 Axicons - 125mm lens','3 Axicons - 200mm lens', '3 Axicons - 100mm lens', '2 Axicons - 100mm lens', 'Axicon - 100mm lens', '100mm lens - axicon - disc', '100mm lens - axicon'])

#Everything below is in microW
P_err = 1.0
ax_1_P = 177.0
fib_tip_P = 183.0
ax_fib_tip_f = ax_1_P/fib_tip_P
ax_fib_tip_f_err = ax_fib_tip_f*np.sqrt((P_err/ax_1_P)**2 + (P_err/fib_tip_P)**2)

onemml_ax_P = 160.0
onemml_ax_disc_P = 113.3
onemml_ax_ax1_P = 168.0

onemml_ax_disc_f = onemml_ax_disc_P/onemml_ax_ax1_P * ax_fib_tip_f
onemml_ax_disc_f_err = onemml_ax_disc_f*np.sqrt((P_err/onemml_ax_disc_P)**2 + (P_err/onemml_ax_ax1_P)**2
                                                + (ax_fib_tip_f_err/ax_fib_tip_f)**2)
onemml_ax_f = onemml_ax_P/onemml_ax_ax1_P * ax_fib_tip_f
onemml_ax_f_err = onemml_ax_f*np.sqrt((P_err/onemml_ax_P)**2 + (P_err/onemml_ax_ax1_P)**2
                                                + (ax_fib_tip_f_err/ax_fib_tip_f)**2)

ax_onemml_P = 167.0
ax_onemml_ax1_P = 180.0

ax_onemml_f = ax_onemml_P/ax_onemml_ax1_P
ax_onemml_f_err = ax_onemml_f*np.sqrt((P_err/ax_onemml_P)**2 + (P_err/ax_onemml_ax1_P)**2)

multhree_ax_ax1_P = 176.0
three_ax_onemml_P = 168.0 
three_ax_onemml_f = three_ax_onemml_P/multhree_ax_ax1_P * ax_fib_tip_f
three_ax_onemml_f_err = three_ax_onemml_f*np.sqrt((P_err/three_ax_onemml_P)**2 + (P_err/multhree_ax_ax1_P)**2
                                                + (ax_fib_tip_f_err/ax_fib_tip_f)**2)

two_ax_onemml_P = 170.0
two_ax_onemml_f = two_ax_onemml_P/multhree_ax_ax1_P * ax_fib_tip_f
two_ax_onemml_f_err = two_ax_onemml_f*np.sqrt((P_err/two_ax_onemml_P)**2 + (P_err/multhree_ax_ax1_P)**2
                                                + (ax_fib_tip_f_err/ax_fib_tip_f)**2)
three_ax_twomml_P = 171.0 
three_ax_twomml_f = three_ax_twomml_P/multhree_ax_ax1_P * ax_fib_tip_f
three_ax_twomml_f_err = three_ax_twomml_f*np.sqrt((P_err/three_ax_twomml_P)**2 + (P_err/multhree_ax_ax1_P)**2
                                                + (ax_fib_tip_f_err/ax_fib_tip_f)**2)
three_ax_twofivemml_f = three_ax_onemml_f*3/4 + three_ax_twomml_f*1/4 #Average between the two, assuming a ~linear relation w.r.t. focal length
three_ax_twofivemml_f_err = np.sqrt((three_ax_onemml_f_err*3/4)**2 + (three_ax_onemml_f_err*1/4)**2)

two_ax_P = 175.0
two_ax_f = two_ax_P/multhree_ax_ax1_P * ax_fib_tip_f
two_ax_f_err = two_ax_onemml_f*np.sqrt((P_err/two_ax_P)**2 + (P_err/multhree_ax_ax1_P)**2
                                                + (ax_fib_tip_f_err/ax_fib_tip_f)**2)




eff_factors = np.array([two_ax_f,three_ax_twofivemml_f,three_ax_twomml_f,three_ax_onemml_f,two_ax_onemml_f,ax_onemml_f,
                       onemml_ax_disc_f,onemml_ax_f])
eff_factors_err = np.array([two_ax_f_err,three_ax_twofivemml_f_err,three_ax_twomml_f_err,three_ax_onemml_f_err,two_ax_onemml_f_err,
                           ax_onemml_f_err,onemml_ax_disc_f_err,onemml_ax_f_err])

avg_I_peak_err_array = np.sqrt((eff_factors*avg_I_peak_err_array)**2 + (eff_factors_err*avg_I_peak_array)**2)
avg_I_peak_array *= eff_factors


Gauss_beam_intensity = 98.31319273317742
Gauss_beam_err = 0.11716214411854588
avg_I_peak_array /= Gauss_beam_intensity
avg_I_peak_err_array = np.sqrt(avg_I_peak_err_array**2 + (Gauss_beam_err*avg_I_peak_array)**2)/Gauss_beam_intensity


# rel_darkness_err_array = np.sqrt((eff_factors*rel_darkness_err_array)**2 + (eff_factors_err*rel_darkness_array))
# rel_darkness_array *= eff_factors

def ratio(a,b,a_err,b_err):
    y = a/b
    y_err = abs(y)*np.sqrt((a_err/a)**2 + (b_err/b)**2)
    return y, y_err

def ratios(indices,R_array,R_err_array,R_std_array,R_std_err_array,L_R_coeff,L_R_coeff_err,R_R_coeff,R_R_coeff_err):
    R_array_y, R_array_y_err  = ratio(R_array[indices[0]],R_array[indices[1]],R_err_array[indices[0]],R_err_array[indices[1]])
    R_std_array_y, R_std_array_y_err = ratio(R_std_array[indices[0]],R_std_array[indices[1]],R_std_err_array[indices[0]],R_std_err_array[indices[1]])
    L_R_coeff_y, L_R_coeff_y_err = ratio(L_R_coeff[indices[0]],L_R_coeff[indices[1]],L_R_coeff_err[indices[0]],L_R_coeff_err[indices[1]])
    R_R_coeff_y, R_R_coeff_y_err = ratio(R_R_coeff[indices[0]],R_R_coeff[indices[1]],R_R_coeff_err[indices[0]],R_R_coeff_err[indices[1]])
    print("Ratio parameter: {} +/- {}".format(R_array_y,R_array_y_err))
    print("Ratio std: {} +/- {}".format(R_std_array_y,R_std_array_y_err))
    print("Ratio c_+: {} +/- {}".format(L_R_coeff_y,L_R_coeff_y_err))
    print("Ratio c_-: {} +/- {}".format(R_R_coeff_y,R_R_coeff_y_err))
    avg_c_y = np.average([abs(L_R_coeff_y),abs(R_R_coeff_y)],weights=[1/L_R_coeff_y_err,1/R_R_coeff_y_err])
    avg_c_y_err= 1/np.sqrt(1/R_R_coeff_y_err**2 + 1/L_R_coeff_y_err**2)
    print("Ratio avg. c: {} +/- {}".format(avg_c_y,avg_c_y_err))
    return

indices = [3,4]
print("Comparison {}/{}".format(labels[indices[0]],labels[indices[1]]))
print("R:")
ratios(indices,R_array,R_err_array,R_std_array,R_std_err_array,L_R_coeff,L_R_coeff_err,R_R_coeff,R_R_coeff_err)
print("m:")  
ratios(indices,avg_m_array,avg_m_err_array,scl_m_std_array,scl_m_std_err_array,L_m_coeff,L_m_coeff_err,R_m_coeff,R_m_coeff_err)
print("b:")  
ratios(indices,rel_darkness_array,rel_darkness_err_array,scl_b_std_array,scl_b_std_err_array,L_dark_coeff,L_dark_coeff_err,R_dark_coeff,R_dark_coeff_err)
print("I:")  
ratios(indices,avg_I_peak_array,avg_I_peak_err_array,scl_I_std_array,scl_I_std_err_array,L_I_coeff,L_I_coeff_err,R_I_coeff,R_I_coeff_err)
print("std:")  
ratios(indices,rel_avg_std_array,rel_avg_std_err_array,scl_std_std_array,scl_std_std_err_array,L_std_coeff,L_std_coeff_err,R_std_coeff,R_std_coeff_err)
 
avg_R_coeff = np.average([L_R_coeff[indices[0]],R_R_coeff[indices[0]]],weights=[1/L_R_coeff_err[indices[0]],1/R_R_coeff_err[indices[0]]])*R_array[indices[0]]
# print(L_R_coeff[indices[0]]*R_array[indices[0]],R_R_coeff[indices[0]]*R_array[indices[0]])
# print(avg_R_coeff,R_array[indices[0]])
L_R_coeff_err_non_rel = np.sqrt((L_R_coeff_err[indices[0]]*R_array[indices[0]])**2-(L_R_coeff[indices[0]]*R_err_array[indices[0]])**2) #Removing the error added previously
R_R_coeff_err_non_rel = np.sqrt((R_R_coeff_err[indices[0]]*R_array[indices[0]])**2-(R_R_coeff[indices[0]]*R_err_array[indices[0]])**2)
avg_R_coeff_err = 1/np.sqrt(1/L_R_coeff_err_non_rel**2 + 1/R_R_coeff_err_non_rel**2)
# print(L_R_coeff_err_non_rel/(L_R_coeff[indices[0]]*R_array[indices[0]]))
# print((L_R_coeff_err[indices[0]]/L_R_coeff[indices[0]]))
# print(L_R_coeff_err_non_rel, L_R_coeff_err[indices[0]]*R_array[indices[0]])
# print(np.arctan(L_R_coeff[indices[0]]*R_array[indices[0]]) * 180/np.pi)
beta = np.arctan(avg_R_coeff) * 180/np.pi
beta_err = abs(beta - np.arctan(avg_R_coeff + avg_R_coeff_err)* 180/np.pi)
print("Propagation angle: {} +/- {} degrees".format(beta,beta_err))
    
    
    
if plot == True:
    fig = plt.figure()
    
    
    
    pos = [0,0]
    size = [0.6,0.3]
    ylabels = [r"$\sigma_{R}$",r"$c_{-}$ ($mm^{-1}$)",r"$c_{+}$ ($mm^{-1}$)"]
    log_plot = 'y'
    letter='(a)'
    master_plot_part(R_array,R_err_array,R_std_array,R_std_err_array,L_R_coeff,L_R_coeff_err,R_R_coeff,R_R_coeff_err,markers,pos,size,fontsize,r'$R$ ($mm$)',letter,ylabels,log_plot=log_plot)
    
    ylabels = [r"$\sigma_{m}$",r"$c_{-}$ ($mm^{-1}$)",r"$c_{+}$ ($mm^{-1}$)"]
    pos = [size[0]+0.25,0]
    log_plot = ''
    letter='(b)'
    master_plot_part(avg_m_array,avg_m_err_array,scl_m_std_array,scl_m_std_err_array,L_m_coeff,L_m_coeff_err,R_m_coeff,R_m_coeff_err,markers,pos,size,fontsize,r"$\overline{m}$",letter,ylabels,log_plot=log_plot)
    
    ylabels = [r"$\sigma_{b_{r}}$",r"$c_{-}$ ($mm^{-1}$)",r"$c_{+}$ ($mm^{-1}$)"]
    pos = [2*(size[0]+0.25),0]
    scale_factor=1
    log_plot = 'x'
    letter='(c)'
    # print(rel_darkness_array*scale_factor,rel_darkness_err_array*scale_factor,scl_b_std_array,scl_b_std_err_array,L_I_coeff,L_I_coeff_err,R_I_coeff,R_I_coeff_err)
    master_plot_part(rel_darkness_array*scale_factor,rel_darkness_err_array*scale_factor,scl_b_std_array,scl_b_std_err_array,L_dark_coeff,L_dark_coeff_err,R_dark_coeff,R_dark_coeff_err,markers,pos,size,fontsize,r"$\overline{b_{r}}$",letter,ylabels,log_plot=log_plot)
    # plt.text(-90,90,r'(PPM)',fontsize=fontsize)
    
    ylabels = [r"$\sigma_{I_{p}}$",r"$c_{-}$ ($mm^{-1}$)",r"$c_{+}$ ($mm^{-1}$)"]
    # pos = [(3*size[0]+0.3*2)/3-(size[0]+0.1)/2,-(size[1]*2.5+0.25)]
    pos = [0.06,-(size[1]*2.5+0.28)]
    scale_factor=1
    log_plot = ''
    letter='(d)'
    master_plot_part(avg_I_peak_array,avg_I_peak_err_array,scl_I_std_array,scl_I_std_err_array,L_I_coeff,L_I_coeff_err,R_I_coeff,R_I_coeff_err,markers,pos,size,fontsize,r"$\overline{I_{p}}$",letter,ylabels,log_plot=log_plot)
    
    ylabels = [r"$\sigma_{\sigma_{resid.}}$",r"$c_{-}$ ($mm^{-1}$)",r"$c_{+}$ ($mm^{-1}$)"]
    pos = [0.06+size[0]+0.25,-(size[1]*2.5+0.28)]
    scale_factor=1
    log_plot = 'xy'
    letter='(e)'
    master_plot_part(rel_avg_std_array,rel_avg_std_err_array,scl_std_std_array,scl_std_std_err_array,L_std_coeff,L_std_coeff_err,R_std_coeff,R_std_coeff_err,markers,pos,size,fontsize,r"$\overline{\sigma_{resid.}}$",letter,ylabels,legend=True,log_plot=log_plot)
    yticks = plt.gca().get_yticks()
    plt.yticks(yticks[2:-1])
    
    
    results_file_name = rootdir + '\\' +  'master_plot.svg'
    plt.savefig(results_file_name,dpi=300,bbox_inches='tight')