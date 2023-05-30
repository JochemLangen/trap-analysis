# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:37:32 2023

@author: joche
"""
from funcs_photos import *
import pandas as pd

path = "Second_beam\Hollow_beam_bottom_left_corner.bmp"
im = Image.open(path)
pixels = np.array(im).T
im_shape = np.shape(pixels)
print(np.amax(pixels))
pixels = pixels/np.amax(pixels)
pixel_size = np.average([6.14/1280,4.9/1021]) #In mm
print(im_shape)
fontsize = 13

#Define Cartesian Coordinates: We take them as the centre of each pixel
corner_coords = [1278.5,1023.5] #Acting guess of centre to find the outer ring (you don't want it to be exactly on a pixel)
R_guess = 600
x_offset_guess = -400   #Offset from corner_coords
y_offset_guess = 0      #Offset from corner_coords
x = np.arange(im_shape[0])-corner_coords[0]
y = np.arange(im_shape[1])-corner_coords[1]
cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])

#Convert to Polar coordinate system:
r, theta, del_theta = CartPolar2(cart_coords)

#Setting the parameters for the outer ring determination
theta_convergence = np.linspace(4,100,20,dtype=int)
averaging_int_size = 10 #The number of points over which it is averaged to get a smooth curve
average_jump_lim = 0.14 #The minimum value to be considered as the outer peak
peak_size = 10 #The +/- area around the first value above this limit in which the max value is taken as the peak
R_array = np.empty_like(theta_convergence,dtype=float)
x_offset_array = np.empty_like(R_array)
y_offset_array = np.empty_like(R_array)

R_err_array = np.empty_like(R_array)
x_offset_err_array = np.empty_like(R_array)
y_offset_err_array = np.empty_like(R_array)

for i in range(len(theta_convergence)):
    print('no_theta_points = ',theta_convergence[i])
    no_theta_points = theta_convergence[i] #The number of angles in each fitting range
    theta_array = np.linspace(theta[0,-1],theta[-1,0],no_theta_points)
    R_array[i], R_err_array[i], x_offset_array[i], x_offset_err_array[i], y_offset_array[i], y_offset_err_array[i] = polar_find_centre(pixels, theta_array, r, theta, del_theta, cart_coords,corner_coords,averaging_int_size,average_jump_lim,peak_size,R_guess,x_offset_guess,y_offset_guess,plot=False,fontsize=fontsize)

    #Print the results:
    print("The outer radius is {} +/- {} mm".format(R_array[i]*pixel_size,R_err_array[i]*pixel_size))
    print("The outer radius is {} +/- {} pixels".format(R_array[i],R_err_array[i]))
    print("The x offset is {} +/- {} pixels".format(x_offset_array[i],x_offset_err_array[i]))
    print("The y offset is {} +/- {} pixels".format(y_offset_array[i],y_offset_err_array[i]))

fig = plt.figure()
fig.add_axes((0,0,1,0.3))
plt.plot([-10,theta_convergence[-1]+10],[R_array[-1],R_array[-1]],linestyle='--',color='red')
plt.errorbar(theta_convergence, R_array, yerr=R_err_array,marker='x',linestyle='',color='black',capsize=2)
plt.ylabel("Radius (pixels)",fontsize=fontsize,labelpad=10)
y_lim = [min(R_array)-max(R_err_array)*2,max(R_array)+max(R_err_array)*2]
y_diff = (y_lim[1]-y_lim[0])/8
y_step = np.ceil((y_lim[1]-y_lim[0]-y_diff)/3)
yticks = np.linspace(int(y_lim[0]+y_diff),int(y_lim[0]+y_diff)+3*y_step,4)
plt.xlim([theta_convergence[0]-5,theta_convergence[-1]+5])
plt.ylim([y_lim[0],y_lim[-1]])
plt.yticks(yticks,fontsize=fontsize)

fig.add_axes((0,-0.3,1,0.3))
plt.plot([-10,theta_convergence[-1]+10],[x_offset_array[-1],x_offset_array[-1]],linestyle='--',color='red')
plt.errorbar(theta_convergence, x_offset_array, yerr=x_offset_err_array,marker='x',linestyle='',color='black',capsize=2)
plt.ylabel("X-position (pixels)",fontsize=fontsize,labelpad=28)
y_lim = [min(x_offset_array)-max(x_offset_err_array)*2,max(x_offset_array)+max(x_offset_err_array)*2]
y_diff = (y_lim[1]-y_lim[0])/8
y_step = np.ceil((y_lim[1]-y_lim[0]-y_diff*2)/3)
yticks = np.linspace(int(y_lim[0]+y_diff),int(y_lim[0]+y_diff)+3*y_step,4)
plt.ylim([y_lim[0],y_lim[-1]])
plt.yticks(yticks,fontsize=fontsize)
plt.xlim([theta_convergence[0]-5,theta_convergence[-1]+5])

fig.add_axes((0,-0.6,1,0.3))
plt.plot([-10,theta_convergence[-1]+10],[y_offset_array[-1],y_offset_array[-1]],linestyle='--',color='red')
plt.errorbar(theta_convergence, y_offset_array, yerr=y_offset_err_array,marker='x',linestyle='',color='black',capsize=2)
plt.ylabel("Y-position (pixels)",fontsize=fontsize,labelpad=2)
plt.xlabel("No. of points used for fitting",fontsize=fontsize)
y_lim = [min(y_offset_array)-max(y_offset_err_array)*2,max(y_offset_array)+max(y_offset_err_array)*2]
y_diff = (y_lim[1]-y_lim[0])/8
y_step = np.ceil((y_lim[1]-y_lim[0]-y_diff)/3)
yticks = np.linspace(int(y_lim[0]),int(y_lim[0])+3*y_step,4)
plt.ylim([y_lim[0],y_lim[-1]])
plt.yticks(yticks,fontsize=fontsize)
plt.xlim([theta_convergence[0]-5,theta_convergence[-1]+5])
plt.xticks(fontsize=fontsize)

plt.savefig(path[:-4]+'_convergence'+'.svg',dpi=300,bbox_inches='tight')
plt.show()




print((R_array[:-1]-R_array[-1])/R_err_array[:-1])
print((R_array[:-1]-R_array[-1])/(R_err_array[:-1]+R_err_array[-1]))
print((y_offset_array[:-1]-y_offset_array[-1])/y_offset_err_array[:-1])
print((y_offset_array[:-1]-y_offset_array[-1])/(y_offset_err_array[:-1]+y_offset_err_array[-1]))
print((x_offset_array[:-1]-x_offset_array[-1])/x_offset_err_array[:-1])
print((x_offset_array[:-1]-x_offset_array[-1])/(x_offset_err_array[:-1]+x_offset_err_array[-1]))