# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:17:22 2023

@author: joche
"""

import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

def R_ax(theta,alpha,n=1.5):
    alpha = alpha/180 * pi
    theta = theta/180 * pi
    return (np.arcsin(n*np.sin(alpha - np.arcsin(np.sin(theta)/n)))-alpha)/pi * 180

def L_ax(theta,alpha,n=1.5):
    alpha = alpha/180 * pi
    theta = theta/180 * pi
    return (np.arcsin(n*np.arcsin(alpha - np.arcsin(np.sin(theta+alpha)/n))))/pi * 180

alpha_1 = 2 #degrees
theta = 0
beta_1 = R_ax(theta,alpha_1)
n = 1.5
fontsize = 13
# theta_range = np.array([0,5,10,15,20,25])
# line_styles = np.array(['-','--','-.',':','-','--'])

theta_range = np.array([0,5,10,15])
line_styles = np.array(['-','--','-.',':',])

max_alpha = np.arcsin(1/n)/pi * 180
# print(max_alpha/pi*180)
alpha_range = np.linspace(0,max_alpha,50)[:-1]




# print(R_ax(0,pi/4,n=1.5))
# for i in range(len(alpha_range)):
#     print(alpha_range[i])
#     beta_2_L = -L_ax(beta_1,alpha_range[i])
#     beta_2_R = -R_ax(beta_1,alpha_range[i])
#     print(beta_2_L,beta_2_R)



comb_alpha = np.arctan(np.tan(alpha_1/180 * pi)-np.tan(alpha_range/180 * pi))*180 / pi
comb_alpha = alpha_1 - alpha_range

theta_len = len(theta_range)
colour_min = 1#/theta_len

# colour_2_L = (0,colour_index,0)
# colour_2_R = (colour_index,0,0)
# colour_comb_2_L = (0,0,colour_index)
# colour_comb_2_R = (0,colour_index,colour_index)

fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
for i in range(theta_len):
    colour_index = colour_min #* (i+1)
    print(colour_index,i)
    beta_1 = L_ax(theta_range[i],alpha_1)
    beta_2_L = -L_ax(beta_1,alpha_range)
    beta_2_R = -R_ax(beta_1,alpha_range)
    beta_2_comb_L = L_ax(theta_range[i],comb_alpha)
    beta_2_comb_R = R_ax(theta_range[i],comb_alpha)
    
    plt.plot(alpha_range,beta_2_L,color='red',linestyle=line_styles[i])
    plt.plot(alpha_range,beta_2_R,color='blue',linestyle=line_styles[i])
    plt.plot(alpha_range,beta_2_comb_L,color='green',linestyle=line_styles[i])
    plt.plot(alpha_range,beta_2_comb_R,color='orange',linestyle=line_styles[i])
                  
# xlim = ax.get_xlim()
xlim = [0,alpha_range[-1]]
ylim = ax.get_ylim()
ylim = [-40,ylim[-1]]
plt.plot([alpha_1,alpha_1],[ylim[0],ylim[-1]],linestyle='--',color='grey',zorder=0)
plt.plot([xlim[0],xlim[-1]],[0,0],linestyle='--',color='grey',zorder=0)
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel(r"Conical angle, $\alpha_{2}$ (degrees)", fontsize=fontsize)
plt.ylabel(r"Propagation angle, $\beta$ (degrees)", fontsize=fontsize)
xticks = ax.get_xticks()
plt.xticks(xticks[:-1], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.text(8,-2.5,r'$\theta$ = 0',fontsize=fontsize)
plt.text(6,-6,r'$\theta$ = 5',fontsize=fontsize)
plt.text(4,-10.6,r'$\theta$ = 10',fontsize=fontsize)
plt.text(2,-14.7,r'$\theta$ = 15',fontsize=fontsize)    
plt.text(36,1.2,r'$\alpha_{1} = 2$',fontsize=fontsize)

ax = fig.add_axes((0.02,0.02,0.3,0.3))
scale_factor = 1/3600 #deg to arcsec
beta_1 = L_ax(theta_range[0],alpha_1)
beta_2_L = -L_ax(beta_1,alpha_range)/scale_factor
beta_2_R = -R_ax(beta_1,alpha_range)/scale_factor
beta_2_comb_L = L_ax(theta_range[0],comb_alpha)/scale_factor
beta_2_comb_R = R_ax(theta_range[0],comb_alpha)/scale_factor

alpha_range -= alpha_1
alpha_range /= scale_factor

plt.plot(alpha_range,beta_2_L,color='red',linestyle=line_styles[0])
plt.plot(alpha_range,beta_2_R,color='blue',linestyle=line_styles[0])
plt.plot(alpha_range,beta_2_comb_L,color='green',linestyle=line_styles[0])
plt.plot(alpha_range,beta_2_comb_R,color='orange',linestyle=line_styles[0])   

x_diff = 0.005/scale_factor
y_diff = 0.002/scale_factor
plt.xlim([-x_diff,x_diff])
plt.ylim([-y_diff,y_diff])
plt.plot([0,0],[-y_diff,y_diff],linestyle='--',color='grey',zorder=0)
plt.plot([-x_diff,+x_diff],[0,0],linestyle='--',color='grey',zorder=0)


plt.xlabel(r'$\delta \alpha_{2}$ (arcsec.)', fontsize=fontsize)
plt.ylabel(r'$\beta$ (arcsec.)', fontsize=fontsize)
# xticks = np.linspace(-x_diff,x_diff,4)[1:]
# plt.xticks(xticks,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)


ax = fig.add_axes((1.02,0,1,1))
alpha_range *= scale_factor
alpha_range += alpha_1

alpha_1 = 5
comb_alpha = np.arctan(np.tan(alpha_1/180 * pi)-np.tan(alpha_range/180 * pi))*180 / pi
comb_alpha = alpha_1 - alpha_range


for i in range(theta_len):
    colour_index = colour_min #* (i+1)
    print(colour_index,i)
    beta_1 = L_ax(theta_range[i],alpha_1)
    beta_2_L = -L_ax(beta_1,alpha_range)
    beta_2_R = -R_ax(beta_1,alpha_range)
    beta_2_comb_L = L_ax(theta_range[i],comb_alpha)
    beta_2_comb_R = R_ax(theta_range[i],comb_alpha)
    
    plt.plot(alpha_range,beta_2_L,color='red',linestyle=line_styles[i])
    plt.plot(alpha_range,beta_2_R,color='blue',linestyle=line_styles[i])
    plt.plot(alpha_range,beta_2_comb_L,color='green',linestyle=line_styles[i])
    plt.plot(alpha_range,beta_2_comb_R,color='orange',linestyle=line_styles[i])
                  
# xlim = ax.get_xlim()
# xlim = [0,alpha_range[-1]]
# ylim = ax.get_ylim()
plt.plot([alpha_1,alpha_1],[ylim[0],ylim[-1]],linestyle='--',color='grey',zorder=0)
plt.plot([xlim[0],xlim[-1]],[0,0],linestyle='--',color='grey',zorder=0)
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel(r"Conical angle, $\alpha_{2}$ (degrees)", fontsize=fontsize)
# plt.ylabel(r"Propagation angle, $\beta$ (degrees)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks([])
plt.text(10.1,-2,r'$\theta$ = 0',fontsize=fontsize)
plt.text(8.66,-6,r'$\theta$ = 5',fontsize=fontsize)
plt.text(7.33,-10.5,r'$\theta$ = 10',fontsize=fontsize)
plt.text(5,-14.7,r'$\theta$ = 15',fontsize=fontsize)   
plt.text(36,1.2,r'$\alpha_{1} = 5$',fontsize=fontsize)
    
ax = fig.add_axes((1.04,0.02,0.3,0.3))
scale_factor = 1/3600 #deg to arcsec
beta_1 = L_ax(theta_range[0],alpha_1)
beta_2_L = -L_ax(beta_1,alpha_range)/scale_factor
beta_2_R = -R_ax(beta_1,alpha_range)/scale_factor
beta_2_comb_L = L_ax(theta_range[0],comb_alpha)/scale_factor
beta_2_comb_R = R_ax(theta_range[0],comb_alpha)/scale_factor

alpha_range -= alpha_1
alpha_range /= scale_factor

plt.plot(alpha_range,beta_2_L,color='red',linestyle=line_styles[0])
plt.plot(alpha_range,beta_2_R,color='blue',linestyle=line_styles[0])
plt.plot(alpha_range,beta_2_comb_L,color='green',linestyle=line_styles[0])
plt.plot(alpha_range,beta_2_comb_R,color='orange',linestyle=line_styles[0])   

x_diff = 0.005/scale_factor
y_diff = 0.010/scale_factor
plt.xlim([-x_diff,x_diff])
plt.ylim([-y_diff,y_diff])
plt.plot([0,0],[-y_diff,y_diff],linestyle='--',color='grey',zorder=0)
plt.plot([-x_diff,+x_diff],[0,0],linestyle='--',color='grey',zorder=0)


plt.xlabel(r'$\delta \alpha_{2}$ (arcsec.)', fontsize=fontsize)
plt.ylabel(r'$\beta$ (arcsec.)', fontsize=fontsize)
# xticks = np.linspace(-x_diff,x_diff,4)[1:]
# plt.xticks(xticks,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.savefig('prop_angles'+'.svg',dpi=10,bbox_inches='tight')
plt.show()