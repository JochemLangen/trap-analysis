# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 13:00:45 2022

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
corner_coords = [1280.0,1024.0] #Acting guess of centre to find the outer ring (you don't want it to be exactly on a pixel)
R_guess = 600
x_offset_guess = 800
y_offset_guess = 1000
x = np.arange(im_shape[0])-corner_coords[0]
y = np.arange(im_shape[1])-corner_coords[1]
cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])

#Convert to Polar coordinate system:
r, theta, del_theta = CartPolar(cart_coords)

#Setting the parameters for the outer ring determination
no_theta_points = 15 #The number of angles in each fitting range
averaging_int_size = 10 #The number of points over which it is averaged to get a smooth curve
average_jump_lim = 0.14 #The minimum value to be considered as the outer peak
peak_size = 10 #The +/- area around the first value above this limit in which the max value is taken as the peak

plt.figure()
pixels_im = plt.imshow(pixels.T)
cbar = plt.colorbar(mappable=pixels_im)
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_label(label='Normalised intensity',fontsize=fontsize)
print(np.amin(theta),np.amax(theta))
theta_limits = [np.amin(theta[int(1279*7/8):,int(1023*7/8):]),np.arctan(im_shape[1]/im_shape[0]),np.amax(theta[int(1279*7/8):,int(1023*7/8):])]
#np.arctan(im_shape[1]/im_shape[0]) np.amax(theta)/3
theta_array = np.linspace(theta_limits[1],theta_limits[2],no_theta_points)
R1, R_err1, x_offset1, x_offset_err1, y_offset1, y_offset_err1 = find_centre(pixels, theta_array, r, theta, del_theta, cart_coords,corner_coords,'x',averaging_int_size,average_jump_lim,peak_size,R_guess,x_offset_guess,y_offset_guess,plot=True)

theta_array = np.linspace(theta_limits[0],theta_limits[1],no_theta_points)
R2, R_err2, x_offset2, x_offset_err2, y_offset2, y_offset_err2 = find_centre(pixels, theta_array, r, theta, del_theta, cart_coords,corner_coords,'y',averaging_int_size,average_jump_lim,peak_size,R_guess,x_offset_guess,y_offset_guess,plot=True)

#Calculate the mean of these two fits:
R = (R1 * 1/R_err1 + R2 * 1/R_err2)/(1/R_err1 + 1/R_err2)
R_err = np.sqrt(1/(1/R_err1**2 + 1/R_err2**2))
print("The outer radius is {} +/- {} mm".format(R*pixel_size,R_err*pixel_size))
print("The outer radius is {} +/- {} pixels".format(R,R_err))
x_offset = (x_offset1 * 1/x_offset_err1 + x_offset2 * 1/x_offset_err2)/(1/x_offset_err1 + 1/x_offset_err2)
x_offset_err = np.sqrt(1/(1/x_offset_err1**2 + 1/x_offset_err2**2))
print("The x offset is {} +/- {} pixels".format(x_offset,x_offset_err))
y_offset = (y_offset1 * 1/y_offset_err1 + y_offset2 * 1/y_offset_err2)/(1/y_offset_err1 + 1/y_offset_err2)
y_offset_err = np.sqrt(1/(1/y_offset_err1**2 + 1/y_offset_err2**2))
print("The y offset is {} +/- {} pixels".format(y_offset,y_offset_err))

plt.errorbar(x_offset,y_offset,xerr=x_offset_err,yerr=y_offset_err,marker='x',color='white',zorder=7,linestyle='')
x_values = np.linspace(x_offset-R+1,x_offset+R-1,100)
y_values = np.linspace(circle_func(x_offset-R+1,R, x_offset, y_offset),y_offset+R-1,100)
# y_values = np.linspace(y_offset-R+1,y_offset+R-1,100)
plt.plot(x_values,circle_func(x_values,R, x_offset, y_offset),color='white',zorder=6)
plt.plot(circle_func(y_values,R,y_offset,x_offset),y_values,color='white',zorder=6)
plt.ylim([0,1023])
plt.xlim([0,1280])
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('Image x-axis (pixels)',fontsize=fontsize)
plt.ylabel('Image y-axis (pixels)',fontsize=fontsize)
plt.show()

#Write results
dataframe = pd.DataFrame(np.array([[R, R_err, x_offset, x_offset_err, y_offset, y_offset_err]]),
                         columns=['Radius (pixels)','Radius error (pixels)','X offset (pixels)','X offset error (pixels)','Y offset (pixels)','Y offset error (pixels)'],
                         dtype=float)
dataframe.to_csv(path[:-4]+'.csv')


# theta_array = np.linspace(theta_limits[0],theta_limits[2],no_theta_points*2)
# plt.figure()
# x_offset = corner_coords[0]-x_offset
# y_offset = corner_coords[1]-y_offset
# print(x_offset,y_offset)
# plt.plot(theta_array,polar_circle_func(theta_array,R,x_offset,y_offset))
# plt.show()

# plt.figure()
# theta_im = plt.imshow(theta.T,interpolation=None,cmap='plasma',aspect='auto')
# plt.colorbar(mappable=theta_im,label='radians')
# plt.show()
