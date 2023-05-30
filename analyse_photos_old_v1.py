# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:48:22 2023

@author: joche
"""
from funcs_photos import *
import pandas as pd

path = "Second_beam\Hollow_beam_bottom_left_corner.bmp"
im = Image.open(path)
pixels = np.array(im).T
im_shape = np.shape(pixels)
pixels = pixels/np.amax(pixels)
pixel_size = np.average([6.14/1280,4.9/1024]) #In mm
print(im_shape)

csv_file = pd.read_csv(path[:-4]+'.csv')
R, R_err, x_offset, x_offset_err, y_offset, y_offset_err = csv_file.values[0,1:]

#Define Cartesian Coordinates: We take them as the centre of each pixel
x = np.arange(im_shape[0])-x_offset
y = np.arange(im_shape[1])-y_offset
cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])


#Convert to Polar coordinate system:
r, theta, del_theta = CartPolar2(cart_coords)

#Find the outer pattern:
outer_theta = theta[r >= R]
theta_limits = np.array([np.amin(outer_theta)+0.03,np.amax(outer_theta)-0.05])
# del_R = np.sqrt(0.5)/R
# no_theta_points = int(np.floor((theta_limits[1]-theta_limits[0])/del_R))
no_theta_points = 3

theta_array = np.linspace(theta_limits[0],theta_limits[1],no_theta_points)
outer_ring_pos = np.empty((no_theta_points,2))

# plt.figure()
# r_im = plt.imshow(r.T,interpolation=None,cmap='plasma',aspect='auto')
# plt.colorbar(mappable=r_im,label='pixels')
# plt.show()

plt.figure()
theta_im = plt.imshow(theta.T,interpolation=None,cmap='plasma',aspect='auto')
plt.colorbar(mappable=theta_im,label='radians')
plt.show()

averaging_int_size = 4 #The number of points over which it is averaged to get a smooth curve
average_jump_lim = 0.14 #The minimum value to be considered as the outer peak
peak_size =4 #The +/- area around the first value above this limit in which the max value is taken as the peak
trough_lim = 12
stacked_pixels = []
stacked_r = []
low_ref = 0.06
high_ref = 0.6
darkness_limit = 0.05

for i in range(no_theta_points):
    radials_bool = np.logical_and(np.logical_and(theta > theta_array[i]-del_theta,
                                              theta < theta_array[i]+del_theta), r >= R-10)
    #The -10 can be defined by the error on R (and the centre)

    r_radials = r[radials_bool]
    sorted_indices = np.argsort(r_radials) #sorting the pixels in order of radii
    r_radials = r_radials[sorted_indices]
    pixel_radials = pixels[radials_bool][sorted_indices]
    fl_pk_index = np.argmax(np.flip(pixel_radials))
    plt.figure()
    print("pixel radials",pixel_radials[0],pixel_radials[-1])
    plt.plot(r_radials,pixel_radials)    
    plt.show()
    
    print(len(pixel_radials))
    if r_radials[-fl_pk_index] == 1: #Cutting off the range we're looking at where the image saturates
        r_radials = 2*r_radials[-fl_pk_index+1]-r_radials[-fl_pk_index+1:] #Flipping the profile as if though a second axicon was in place
        pixel_radials = pixel_radials[-fl_pk_index+1:]
    else:
        #r_radials = r_radials[-fl_pk_index:]
        r_radials = 2*r_radials[-fl_pk_index]-r_radials[-fl_pk_index:] #Flipping the profile as if though a second axicon was in place
        pixel_radials = pixel_radials[-fl_pk_index:]
    plt.figure()
    print("pixel radials",pixel_radials[0],pixel_radials[-1])
    plt.plot(r_radials,pixel_radials)  
    plt.show()


    low_index = int(np.argwhere(pixel_radials<low_ref)[0])
    high_index = int(np.argwhere(pixel_radials>high_ref)[-1])
    print(low_index, high_index)
    #np.argmin(abs(pixel_radials-high_ref))


    pixel_avg = interval_avg(pixel_radials, averaging_int_size) #Take the averages over every -interval- points
    radial_avg = interval_avg(r_radials, averaging_int_size)
    log_avg_pixel = np.log(pixel_avg)
    
    avg_diff = -(pixel_avg[:-1]-pixel_avg[1:])#/(radial_avg[:-1]-radial_avg[1:]) #Find the difference between these points
    avg_peak_index = np.argmax(avg_diff)
    
    avg_trough_index = int(np.argwhere(pixel_avg[:avg_peak_index]<darkness_limit)[0]*averaging_int_size)
    avg_peak_index *= averaging_int_size
    
    #Log method:
    log_avg_diff = -(log_avg_pixel[:-1]-log_avg_pixel[1:])#/(log_avg_r[:-1]-log_avg_r[1:])
    log_avg_peak_index = np.argmax(log_avg_diff)
    
    log_avg_trough_index = int(np.argwhere(log_avg_diff[:log_avg_peak_index]<0)[0]*averaging_int_size)
    log_avg_peak_index *= averaging_int_size
    log_pixel = np.log(pixel_radials)
    # print(log_pixel)
    # log_diff = (log_pixel[:-1]-log_pixel[1:])#/(r_radials[:-1]-r_radials[1:])
    
    # log_avg_pixel = np.log(pixel_avg)
    log_avg_r = np.log(radial_avg)

    plt.figure()
    plt.plot(pixel_avg)    
    plt.show()

    plt.figure()
    plt.plot([radial_avg[0],radial_avg[-1]],[0,0],color='black',linestyle='--')
    plt.plot(radial_avg[:-1],log_avg_diff)
    plt.title('log_avg_diff')
    plt.scatter(radial_avg[int(log_avg_peak_index/averaging_int_size)],log_avg_diff[int(log_avg_peak_index/averaging_int_size)])
    plt.scatter(radial_avg[int(avg_peak_index/averaging_int_size)],log_avg_diff[int(avg_peak_index/averaging_int_size)])
    
    # plt.xscale('log')
    plt.show()
    
    # plt.figure()
    # plt.plot(r_radials,log_pixel)
    # plt.title('log_pixel over r_radials')
    # plt.show()
    
    # plt.figure()
    # plt.plot(r_radials[:-1],log_diff)
    # plt.title('log_diff over r_radials')
    # plt.show()
    
    plt.figure()
    plt.plot(radial_avg[:-1],pixel_avg[:-1])
    # print("peak index avg",avg_peak_index)
    plt.scatter(radial_avg[int(log_avg_peak_index/averaging_int_size)],pixel_avg[int(log_avg_peak_index/averaging_int_size)])
    plt.scatter(radial_avg[int(avg_peak_index/averaging_int_size)],pixel_avg[int(avg_peak_index/averaging_int_size)])
    plt.show()
    
    plt.figure()
    plt.plot([radial_avg[0],radial_avg[-1]],[0,0],color='black',linestyle='--')
    plt.plot(radial_avg[:-1],avg_diff)
    # print("peak index avg",avg_peak_index)
    plt.scatter(radial_avg[int(log_avg_peak_index/averaging_int_size)],avg_diff[int(log_avg_peak_index/averaging_int_size)])
    plt.scatter(radial_avg[int(avg_peak_index/averaging_int_size)],avg_diff[int(avg_peak_index/averaging_int_size)])
    plt.show()
    
    #Get the index of the point furthest out where this difference exceeds the reference value
    if avg_peak_index >= peak_size:
        peak_pixels = pixel_radials[avg_peak_index-peak_size:avg_peak_index+peak_size]
        
        # peak_r = r_radials[avg_peak_index-peak_size:avg_peak_index+peak_size]
        pixel_diff = peak_pixels[1:]-peak_pixels[:-1]#/(peak_r[:-1]-peak_r[1:])
        peak_index = avg_peak_index-peak_size + np.argmax(pixel_diff)+1
        
        plt.figure()
        plt.plot(r_radials[avg_peak_index-peak_size:avg_peak_index+peak_size],peak_pixels)
        plt.plot(r_radials[avg_peak_index-peak_size:avg_peak_index+peak_size][:-1],pixel_diff)
        plt.title('pixel_diff')
        plt.show()
        
    else:
        peak_pixels = pixel_radials[avg_peak_index-peak_size:]
        peak_r = r_radials[0:avg_peak_index+peak_size]
        pixel_diff = peak_pixels[1:]-peak_pixels[:-1]#/(peak_r[:-1]-peak_r[1:]) 
        peak_index = avg_peak_index-peak_size + np.argmax(pixel_diff)+1
        
        # plt.figure()
        # plt.plot(r_radials[0:avg_peak_index+peak_size][:-1],pixel_diff)
        # plt.title('pixel_diff')
        # plt.show()
        

    print("log trough index:",log_avg_trough_index, )
    
    if log_avg_peak_index >= peak_size:
        log_peak_pixels = log_pixel[log_avg_peak_index-peak_size:log_avg_peak_index+peak_size]
        log_diff = -(log_peak_pixels[:-1]-log_peak_pixels[1:])
        log_peak_index = log_avg_peak_index-peak_size + np.argmax(log_diff)
        
        # plt.figure()
        # plt.title('log_pixel_diff')
        # plt.plot(r_radials[log_avg_peak_index-peak_size:log_avg_peak_index+peak_size][:-1],log_diff)
        # plt.show()
    else:
        log_peak_pixels = log_pixel[0:log_avg_peak_index+peak_size]
        log_diff = -(log_peak_pixels[:-1]-log_peak_pixels[1:])
        log_peak_index = log_avg_peak_index-peak_size + np.argmax(log_diff)
        
        # plt.figure()
        # plt.title('log_pixel_diff')
        # plt.plot(r_radials[0:log_avg_peak_index+peak_size][:-1],log_diff)
        # plt.show()
        
    trough_pixels = pixel_radials[avg_trough_index-peak_size:avg_trough_index+peak_size]
    tr_pixel_diff = trough_pixels[1:]-trough_pixels[:-1]
    trough_index = avg_trough_index-peak_size + np.argmax(tr_pixel_diff)
    
    log_trough_pixels = log_pixel[log_avg_trough_index-peak_size:log_avg_trough_index+peak_size]
    log_tr_pixel_diff = log_trough_pixels[1:]-log_trough_pixels[:-1]
    log_trough_index = log_avg_trough_index-peak_size + np.argmax(log_tr_pixel_diff)
    # peak_index = avg_peak_index-peak_size+np.argmax(pixel_radials[avg_peak_index-peak_size:avg_peak_index+peak_size])
    #Get the index of the heighest point around this found peak, i.e. the actual peak value
    
    final_pixels = pixel_radials[peak_index:][:trough_index]
    log_final_pixels = pixel_radials[log_peak_index:][:log_trough_index]
    print(trough_index)
    stacked_pixels += list(pixel_radials)
    stacked_r += list(r_radials)
    # pixel_diff = pixel_radials[1:].astype(int)-pixel_radials[:-1].astype(int) #They are converted to integers as they were uint8 scalars before (mod 255)
    # print(r_radials)
    
    
    plt.figure()
    plt.scatter(r_radials,pixel_radials)
    plt.scatter(r_radials[peak_index],pixel_radials[peak_index],marker='x')
    plt.title(theta_array[i])
    plt.plot(r_radials[peak_index:][:trough_index],final_pixels, color='red',zorder=5)
    plt.plot(r_radials[log_peak_index:][:log_trough_index],log_final_pixels, color='green',zorder=7,alpha=0.5)
    plt.plot(r_radials[high_index:low_index],pixel_radials[high_index:low_index],color='black',zorder=10,alpha=0.5)
    plt.show()
    
    plt.figure()
    plt.scatter(r_radials,pixel_radials)
    plt.scatter(r_radials[peak_index],pixel_radials[peak_index],marker='x')
    plt.title(theta_array[i])
    plt.plot(r_radials[peak_index:][:trough_index],final_pixels, color='red',zorder=5)
    plt.plot(r_radials[log_peak_index:][:log_trough_index],log_final_pixels, color='green',zorder=7,alpha=0.5)
    plt.xlim([r_radials[high_index]-30,r_radials[low_index]+30])
    plt.plot(r_radials[high_index:low_index],pixel_radials[high_index:low_index],color='black',zorder=10,alpha=0.5)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    

    

    

    #Plot the line of points investigated on the image, with the found outer ring marked
    # plt.figure()
    # plt.imshow(im)
    # plt.errorbar(x_offset,y_offset,xerr=x_offset_err,yerr=y_offset_err,marker='x',color='green',zorder=7)
    # plt.plot(cart_coords[radials_bool,0]+x_offset,cart_coords[radials_bool,1]+y_offset,color='red',marker='.',markersize=0.01)
    # x_values = np.linspace(x_offset-R+1,x_offset+R-1,100)
    # y_values = np.linspace(y_offset-R+1,y_offset+R-1,100)
    # plt.plot(x_values,circle_func(x_values,R, x_offset, y_offset),color='green',zorder=6)
    # plt.plot(circle_func(y_values,R,y_offset,x_offset),y_values,color='green',zorder=6)
    # plt.xlim([0,1280])
    # plt.ylim([0,1024])
    # plt.show()
    
    # pixel_avg = interval_avg(pixel_radials, averaging_int_size) #Take the averages over every -interval- points
    # avg_diff = pixel_avg[:-1]-pixel_avg[1:] #Find the difference between these points
    # avg_peak_index = np.asarray((avg_diff > average_jump_lim).nonzero())[0,-1]*averaging_int_size 
    # #Get the index of the point furthest out where this difference exceeds the reference value
    # peak_index = avg_peak_index-peak_size+np.argmax(pixel_radials[avg_peak_index-peak_size:avg_peak_index+peak_size])
    # #Get the index of the heighest point around this found peak, i.e. the actual peak value
    # outer_ring_pos[i] = cart_coords[radials_bool,:][sorted_indices,:][peak_index,:]

# stacked_pixels = np.array(stacked_pixels)
# stacked_r = np.array(stacked_r)
# sorted_indices_sr = np.argsort(stacked_r) #sorting the pixels in order of radii
# stacked_r = stacked_r[sorted_indices_sr]
# stacked_pixels = stacked_pixels[sorted_indices_sr]

# plt.figure()
# plt.scatter(stacked_r,stacked_pixels)
# plt.xlim([0,50])
# plt.show()