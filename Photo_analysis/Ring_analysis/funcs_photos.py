# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 17:53:42 2023

@author: jochem langen
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

pi = np.pi

def interval_avg(y_data, interval):
    y_length = len(y_data)
    remainder = np.remainder(y_length,interval)
    
    dimension = int((y_length-remainder)/interval)-1
    used_shape = dimension*interval
    # print("The averages are taken over {} points, though the last one may be different and is here taken over {} points.".format(interval,interval+remainder))

    used_y = y_data[:used_shape]
    remainder_y = y_data[used_shape:]
    
    reshaped_y = np.reshape(used_y, (dimension,interval))
    averaged_y = np.average(reshaped_y, axis=1)
    return np.append(averaged_y, np.mean(remainder_y))

def interval_avg_weighted(y_data, y_err, interval):
    y_length = len(y_data)
    remainder = np.remainder(y_length,interval)
    
    dimension = int((y_length-remainder)/interval)-1
    used_shape = dimension*interval
    # print("The averages are taken over {} points, though the last one may be different and is here taken over {} points.".format(interval,interval+remainder))

    used_y = y_data[:used_shape]
    used_y_err = y_err[:used_shape]
    remainder_y = y_data[used_shape:]
    remainder_y_err = y_err[used_shape:]
    
    reshaped_y = np.reshape(used_y, (dimension,interval))
    reshaped_y_err = np.reshape(used_y_err, (dimension,interval))
    
    averaged_y, averaged_y_err = weighted_avg_2D_1(reshaped_y, reshaped_y_err)
    rem_avg_y, rem_avg_y_err = weighted_avg(remainder_y, remainder_y_err)
    
    return np.append(averaged_y, rem_avg_y), np.append(averaged_y_err, rem_avg_y_err)

def circle_func(x,R,A,B):
    # (x-A)^2 + (y-B)^2 = R^2
    # y = +/- sqrt(R^2 - (x-A)^2) + B
    return -np.sqrt(R**2 - (x-A)**2) + B

def polar_circle_func(theta,R,A,B):
    sin = np.sin(theta)
    cos = np.cos(theta)
    return A*cos + B*sin + np.sqrt(R**2-(A*sin-B*cos)**2)

def find_centre(pixels, theta_array, r, theta, del_theta, cart_coords, corner_coords, fit, averaging_int_size=10,average_jump_lim=0.14,peak_size=10,R_guess = 600,x_offset_guess = 800,y_offset_guess = 1000,plot=True):
    no_theta_points = len(theta_array)
    outer_ring_pos = np.empty((no_theta_points,2))
    outer_ring_polar = np.empty((no_theta_points,2))

    for i in range(no_theta_points):
        radials_bool = np.logical_and(theta > theta_array[i]-del_theta,
                                                 theta < theta_array[i]+del_theta)

        r_radials = r[radials_bool]
        sorted_indices = np.argsort(r_radials) #sorting the pixels in order of radii
        pixel_radials = pixels[radials_bool][sorted_indices]
        r_radials = r_radials[sorted_indices]
        
        pixel_avg = interval_avg(pixel_radials, averaging_int_size) #Take the averages over every -interval- points
        avg_diff = pixel_avg[:-1]-pixel_avg[1:] #Find the difference between these points
        avg_peak_index = np.asarray((avg_diff > average_jump_lim).nonzero())[0,-1]*averaging_int_size 
        #Get the index of the point furthest out where this difference exceeds the reference value
        peak_index = avg_peak_index-peak_size+np.argmax(pixel_radials[avg_peak_index-peak_size:avg_peak_index+peak_size+1])
        #Get the index of the heighest point around this found peak, i.e. the actual peak value
        outer_ring_pos[i] = cart_coords[radials_bool,:][sorted_indices,:][peak_index,:]
        outer_ring_polar[i,0] = theta[radials_bool][sorted_indices][peak_index]
        outer_ring_polar[i,1] = r_radials[peak_index]
        
        
        # print("Theta = {}".format(theta_array[i]))
        # #Plot the intensity profile over the distance from the corner with the found outer ring marked
        # plt.figure()
        # plt.plot(r_radials,pixel_radials)
        # plt.scatter(r_radials[peak_index],pixel_radials[peak_index],color='red')
        # plt.show()
        
        # #Plot the intensity step profile with the found outer ring marked before the top of the peak was used
        # plt.figure()
        # plt.plot(avg_diff)
        # plt.scatter(int(avg_peak_index/averaging_int_size),avg_diff[int(avg_peak_index/averaging_int_size)],color='red')
        
        #Plot the line of points investigated on the image, with the found outer ring marked
        # plt.figure()
        # path = "Second_beam\Hollow_beam_bottom_left_corner.bmp"
        # im = Image.open(path)
        # plt.imshow(im)
        # plt.gca().set_aspect('equal')
        # plt.xlim([0,1279])
        # plt.ylim([0,1023])
        # plt.plot(cart_coords[radials_bool,0]+corner_coords[0],cart_coords[radials_bool,1]+corner_coords[1],color='red',marker='.',markersize=0.01)
        # plt.scatter(outer_ring_pos[i,0]+corner_coords[0],outer_ring_pos[i,1]+corner_coords[1], marker='x',color='black')
        # plt.scatter(cart_coords[int(1279*7/8),int(1023*7/8),0]+1280,cart_coords[int(1279*7/8),int(1023*7/8),1]+1024)
        # plt.show()
    outer_ring_pos += corner_coords
    if fit == 'x':
        popt, pcov = curve_fit(circle_func,outer_ring_pos[:,0],outer_ring_pos[:,1],p0=[R_guess,x_offset_guess,y_offset_guess],bounds=(0,[2000,2000,2000]))
        R, x_offset, y_offset = popt
        R_err, x_offset_err, y_offset_err = np.sqrt(np.diagonal(pcov))
    elif fit == 'y':
        popt, pcov = curve_fit(circle_func,outer_ring_pos[:,1],outer_ring_pos[:,0],p0=[R_guess,y_offset_guess,x_offset_guess],bounds=(0,[2000,2000,2000]))
        R, y_offset, x_offset = popt
        R_err, y_offset_err, x_offset_err = np.sqrt(np.diagonal(pcov))
    elif fit == 'polar':
        popt, pcov = curve_fit(polar_circle_func,outer_ring_polar[:,0],outer_ring_polar[:,1],p0=[R_guess,x_offset_guess,y_offset_guess],bounds=([0,-1000,-1000],[2000,1000,1000]))
        R, x_offset, y_offset = popt
        x_offset = corner_coords[0]+x_offset
        y_offset = corner_coords[1]+y_offset
        R_err, x_offset_err, y_offset_err = np.sqrt(np.diagonal(pcov))
    else:
        print("No suitable fit type had been chosen!")
   
    if plot == True:

        plt.scatter(outer_ring_pos[:,0],outer_ring_pos[:,1], color='black',marker='x',s=2,zorder=15)
        if fit == 'x':
            # plt.scatter(outer_ring_pos[:,0],circle_func(outer_ring_pos[:,0],R_guess,x_offset_guess,y_offset_guess),marker='.',color='orange',zorder=3)
            plt.plot(outer_ring_pos[:,0],circle_func(outer_ring_pos[:,0],R, x_offset, y_offset),color='red',zorder=6)
        elif fit == 'y':
            # plt.scatter(circle_func(outer_ring_pos[:,1],R_guess,y_offset_guess,x_offset_guess),outer_ring_pos[:,1],marker='.',color='orange',zorder=3)
            plt.plot(circle_func(outer_ring_pos[:,1],R,y_offset,x_offset),outer_ring_pos[:,1],color='red',zorder=6)
        plt.errorbar(x_offset,y_offset,xerr=x_offset_err,yerr=y_offset_err,marker='x',color='red')

    return R, R_err, x_offset, x_offset_err, y_offset, y_offset_err

def find_inner_peak(r,pixels,theta,del_theta,theta_val,R,R_err,r_radial_err,max_intensity,ring_type,darkness_limit=0.05,averaging_int_size=10):
    radials_bool = np.logical_and(theta > theta_val-del_theta,
                                             theta < theta_val+del_theta)

    r_radials = r[radials_bool]
    sorted_indices = np.argsort(r_radials) #sorting the pixels in order of radii
    pixel_radials = pixels[radials_bool][sorted_indices]
    r_radials = r_radials[sorted_indices]
    
    pixel_avg = interval_avg(pixel_radials, averaging_int_size) #Take the averages over every -interval- points
    
    if ring_type == "outer":
        avg_diff = pixel_avg[:-1]-pixel_avg[1:] #Find the difference between these points
        
        beyond_dark_index = np.asarray((pixel_avg > darkness_limit).nonzero())[0,-1]
        avg_peak_index = (np.asarray((avg_diff[:beyond_dark_index] < 0).nonzero())[0,-1]+1)*averaging_int_size
    
        peak_index = avg_peak_index+np.argmax(pixel_radials[avg_peak_index:avg_peak_index+averaging_int_size+1])
    elif ring_type == "inner":
        avg_diff = pixel_avg[1:]-pixel_avg[:-1]
        beyond_dark_index = np.asarray((pixel_avg > darkness_limit).nonzero())[0,0]
        
        avg_peak_index = (np.asarray((avg_diff[beyond_dark_index:] < 0).nonzero())[0,0]+beyond_dark_index)*averaging_int_size
        peak_index = avg_peak_index-averaging_int_size+np.argmax(pixel_radials[avg_peak_index-averaging_int_size:avg_peak_index+averaging_int_size+1]) 
        #The range is made larger to include more in front of the beam as well as all that is averaged over (to negate the effect of the sampling)
        #This cannot be accounted for in the other case as that tends to include inner rings which may sometimes be brighter (if this is never the case -> incorporate)
    else:
        print("No valid ring type is chosen! It is either 'inner' or 'outer'.")
    
    peak_intensity = pixel_radials[peak_index]
    peak_intensity_err = peak_intensity/max_intensity*np.sqrt((0.5/peak_intensity)**2+0.5**2)
    if peak_intensity == 1: #Cutting off the range we're looking at where the image saturates
        peak_index -= 1
        
    r_radials = r_radials[:peak_index]/R
    pixel_radials = pixel_radials[:peak_index]
    peak_pos = r_radials[-1]
    r_radial_err /= R
    
    peak_pos_err = peak_pos/R*np.sqrt((r_radial_err/peak_pos)**2+R_err**2)
    return r_radials,pixel_radials,peak_intensity,peak_intensity_err,peak_pos,peak_pos_err
    

def find_outer_peak(r,pixels,theta,del_theta,theta_val,R,R_err,r_radial_err,max_intensity,darkness_limit=0.05,averaging_int_size=10):
    radials_bool = np.logical_and(theta > theta_val-del_theta,
                                             theta < theta_val+del_theta)

    r_radials = r[radials_bool]
    sorted_indices = np.argsort(r_radials) #sorting the pixels in order of radii
    pixel_radials = pixels[radials_bool][sorted_indices]
    r_radials = r_radials[sorted_indices]
    
    pixel_avg = interval_avg(pixel_radials, averaging_int_size) #Take the averages over every -interval- points
    avg_diff = pixel_avg[:-1]-pixel_avg[1:] #Find the difference between these points
    
    beyond_dark_index = np.asarray((pixel_avg > darkness_limit).nonzero())[0,-1]
    avg_peak_index = (np.asarray((avg_diff[:beyond_dark_index] < 0).nonzero())[0,-1]+1)*averaging_int_size
    peak_index = avg_peak_index+np.argmax(pixel_radials[avg_peak_index:avg_peak_index+averaging_int_size])
    
    peak_intensity = pixel_radials[peak_index]
    peak_intensity_err = peak_intensity/max_intensity*np.sqrt((0.5/peak_intensity)**2+0.5**2)
    
    if peak_intensity == 1: #Cutting off the range we're looking at where the image saturates
        peak_index += 1
        r_radials = 2*r_radials[peak_index]-r_radials[peak_index:] #Flipping the profile as if though a second axicon was in place
        pixel_radials = pixel_radials[peak_index:][r_radials > 0]
        r_radials = r_radials[r_radials > 0] #Removing the part that extends beyond the zero point
    else:
        #r_radials = r_radials[-fl_pk_index:]
        r_radials = 2*r_radials[peak_index]-r_radials[peak_index:] #Flipping the profile as if though a second axicon was in place
        pixel_radials = pixel_radials[peak_index:][r_radials > 0]
        r_radials = r_radials[r_radials > 0] #Removing the part that extends beyond the zero point   
    
    r_radials /= R
    peak_pos = r_radials[0]
    peak_pos_err = peak_pos/R*np.sqrt((r_radial_err/peak_pos)**2+R_err**2)
    return r_radials,pixel_radials,peak_intensity,peak_intensity_err,peak_pos,peak_pos_err
    
def find_outer_peak_alt(r,pixels,theta,del_theta,theta_val,R,R_err,r_radial_err,max_intensity):
    radials_bool = np.logical_and(np.logical_and(theta > theta_val-del_theta,
                                              theta < theta_val+del_theta), r >= R-R_err*20)
    #The -10 can be defined by the error on R (and the centre)

    r_radials = r[radials_bool]
    sorted_indices = np.argsort(r_radials) #sorting the pixels in order of radii
    r_radials = r_radials[sorted_indices]
    pixel_radials = pixels[radials_bool][sorted_indices]
    fl_pk_index = np.argmax(np.flip(pixel_radials))

    peak_intensity = pixel_radials[-fl_pk_index]
    peak_intensity_err = peak_intensity/max_intensity*np.sqrt((0.5/peak_intensity)**2+0.5**2)
    
    if peak_intensity == 1: #Cutting off the range we're looking at where the image saturates
        fl_pk_index += 1
        r_radials = 2*r_radials[-fl_pk_index]-r_radials[-fl_pk_index:] #Flipping the profile as if though a second axicon was in place
        pixel_radials = pixel_radials[-fl_pk_index:][r_radials > 0]
        r_radials = r_radials[r_radials > 0] #Removing the part that extends beyond the zero point
    else:
        #r_radials = r_radials[-fl_pk_index:]
        r_radials = 2*r_radials[-fl_pk_index]-r_radials[-fl_pk_index:] #Flipping the profile as if though a second axicon was in place
        pixel_radials = pixel_radials[-fl_pk_index:][r_radials > 0]
        r_radials = r_radials[r_radials > 0] #Removing the part that extends beyond the zero point   
    
    r_radials /= R
    peak_pos = r_radials[0]
    peak_pos_err = peak_pos/R*np.sqrt((r_radial_err/peak_pos)**2+R_err**2)
    return r_radials,pixel_radials,peak_intensity,peak_intensity_err,peak_pos,peak_pos_err
    
def polar_find_centre(pixels, theta_array, r, theta, del_theta, cart_coords, corner_coords, averaging_int_size=10,darkness_limit = 0.05,drop_limit = -0.05,peak_size=10,R_guess = 600,x_offset_guess = 800,y_offset_guess = 1000,ring_type="outer",plot=True,subplot=False,fontsize=13,path=None,save_figs=False):
    no_theta_points = len(theta_array)
    outer_ring_pos = np.empty((no_theta_points,2))
    outer_ring_polar = np.empty((no_theta_points,2))

    for i in range(no_theta_points):
        radials_bool = np.logical_and(theta-del_theta <= theta_array[i],
                                                 theta+del_theta >= theta_array[i])
        # r_range = np.linspace(0.1,600,100)
        # del_range = 0.5/r_range
        # fig = plt.figure()
        # fig.add_axes((0,0,1,1))
        # plt.imshow(pixels.T,interpolation=None)
        # plt.scatter(cart_coords[radials_bool,0]+corner_coords[0],cart_coords[radials_bool,1]+corner_coords[1],color='red',marker='x')
        # # print(r*(np.sin(theta+del_theta)-np.sin(theta)))
        # # plt.plot(cart_coords[radials_bool,0]+corner_coords[0],cart_coords[radials_bool,1]+corner_coords[1],color='red',marker='.',markersize=0.01)
        # plt.plot(r_range*np.cos(theta_array[i])+corner_coords[0],r_range*np.sin(theta_array[i])+corner_coords[1])
        # plt.plot(r_range*np.cos(theta_array[i]+del_range)+corner_coords[0],r_range*np.sin(theta_array[i]+del_range)+corner_coords[1])
        # plt.plot(r_range*np.cos(theta_array[i]-del_range)+corner_coords[0],r_range*np.sin(theta_array[i]-del_range)+corner_coords[1])
        # plt.scatter(r*np.cos(theta)+corner_coords[0],r*np.sin(theta)+corner_coords[1],marker='.')
        
        # plt.scatter(r*np.cos(theta)+corner_coords[0],r*np.sin(theta-del_theta)+corner_coords[1],marker='.')
        # plt.scatter(r*np.cos(theta)+corner_coords[0],r*np.sin(theta+del_theta)+corner_coords[1],marker='.')
        
        # plt.xlim(850,860)
        # plt.gca().set_aspect('equal')

        # plt.ylim(corner_coords[1]-5,corner_coords[1]+5)
        # plt.show()
        
        
        r_radials = r[radials_bool]
        # print(np.shape(r_radials),'shape')
        sorted_indices = np.argsort(r_radials) #sorting the pixels in order of radii
        pixel_radials = pixels[radials_bool][sorted_indices]
        r_radials = r_radials[sorted_indices]
        pixel_avg = interval_avg(pixel_radials, averaging_int_size) #Take the averages over every -interval- points
        
        
        
        if ring_type == "outer":
            avg_diff = pixel_avg[:-1]-pixel_avg[1:] #Find the difference between these points
            
            #These are two unused methods of finding the peak:
                
            # avg_peak_index = np.asarray((avg_diff > average_jump_lim).nonzero())[0,-1]*averaging_int_size 
            # avg_peak_index_a = np.asarray((pixel_avg > 0.1).nonzero())[0,-1]*averaging_int_size 
            # #Get the index of the point furthest out where this difference exceeds the reference value
            # peak_index = avg_peak_index-peak_size+np.argmax(pixel_radials[avg_peak_index-peak_size:avg_peak_index+peak_size])
            # peak_index_a = avg_peak_index_a-peak_size+np.argmax(pixel_radials[avg_peak_index_a-peak_size:avg_peak_index_a+peak_size])
            # #Get the index of the heighest point around this found peak, i.e. the actual peak value

            beyond_dark_index = np.asarray((pixel_avg > darkness_limit).nonzero())[0,-1]
            avg_peak_index = (np.asarray((avg_diff[:beyond_dark_index] < 0).nonzero())[0,-1]+1)*averaging_int_size
        
            peak_index = avg_peak_index+np.argmax(pixel_radials[avg_peak_index:avg_peak_index+peak_size+1])
            #avg_peak_index-peak_size+np.argmax(pixel_radials[avg_peak_index-peak_size:avg_peak_index+peak_size])
        elif ring_type == "inner":
            avg_diff = pixel_avg[1:]-pixel_avg[:-1]
            beyond_dark_index = np.asarray((pixel_avg > darkness_limit).nonzero())[0,0]
            
            avg_peak_index = (np.asarray((avg_diff[beyond_dark_index:] < drop_limit).nonzero())[0,0]+beyond_dark_index)*averaging_int_size
            peak_index = avg_peak_index-peak_size+np.argmax(pixel_radials[avg_peak_index-peak_size:avg_peak_index+peak_size+1]) 
            #The range is made larger to include more in front of the beam as well as all that is averaged over (to negate the effect of the sampling)
            #This cannot be accounted for in the other case as that tends to include inner rings which may sometimes be brighter (if this is never the case -> incorporate)
        else:
            print("No valid ring type is chosen! It is either 'inner' or 'outer'.")
        
        
        outer_ring_pos[i] = cart_coords[radials_bool,:][sorted_indices,:][peak_index,:]
        outer_ring_polar[i,0] = theta[radials_bool][sorted_indices][peak_index]
        outer_ring_polar[i,1] = r_radials[peak_index]
        
    
        if subplot == True:
            r_avg = interval_avg(r_radials, averaging_int_size)
            avg_peak_indices = (np.asarray((avg_diff[beyond_dark_index:] < 0).nonzero())[0]+1)*averaging_int_size
            #Plot the intensity step profile with the found outer ring marked before the top of the peak was used
            fig = plt.figure()
            ax = fig.add_axes((0,0,1,1))
            plt.plot(r_avg, pixel_avg)
            plt.plot(r_avg[:-1],avg_diff)
            plt.scatter(r_avg[int(avg_peak_index/averaging_int_size)],pixel_avg[int(avg_peak_index/averaging_int_size)],color='black',zorder=5,marker='x')
            # plt.scatter(r_avg[beyond_dark_index],pixel_avg[beyond_dark_index],color='red',zorder=5,marker='x')
            plt.xticks([])
            plt.yticks(fontsize=fontsize)
            plt.ylabel('(Difference in) norm. intensity',fontsize=fontsize)
            xlim = ax.get_xlim()
        
            #Plot the intensity profile over the distance from the corner with the found outer ring marked
            fig.add_axes((0,-1,1,1))
            plt.plot(r_radials,pixel_radials)
            plt.scatter(r_radials[peak_index],pixel_radials[peak_index],color='black',marker='x')
            plt.xlim(xlim)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel('Distance, $r$ (pixels)',fontsize=fontsize)
            plt.ylabel('Normalised intensity',fontsize=fontsize)
            if save_figs == True:
                plt.savefig(path[:-4]+'_subplot'+'.svg',dpi=300,bbox_inches='tight')
            plt.show()
            
            #Plot the line of points investigated on the image, with the found outer ring marked
            plt.figure()
            plt.imshow(pixels.T)
            plt.plot(cart_coords[radials_bool,0]+corner_coords[0],cart_coords[radials_bool,1]+corner_coords[1],color='red',marker='.',markersize=0.01)
            plt.scatter(outer_ring_pos[i,0]+corner_coords[0],outer_ring_pos[i,1]+corner_coords[1], marker='x',color='black')
            plt.gca().set_aspect('equal')
            plt.xlim([0,1279])
            plt.ylim([0,1023])
            # plt.scatter(cart_coords[int(1279*7/8),int(1023*7/8),0]+1280,cart_coords[int(1279*7/8),int(1023*7/8),1]+1024)
            if save_figs == True:
                plt.savefig(path[:-4]+'_subplot_line'+'.svg',dpi=300,bbox_inches='tight')
            plt.show()
        

    outer_ring_pos += corner_coords

    popt, pcov = curve_fit(polar_circle_func,outer_ring_polar[:,0],outer_ring_polar[:,1],p0=[R_guess,x_offset_guess,y_offset_guess],bounds=([0,-1000,-1000],[2000,1000,1000]))
    R, x_offset_or, y_offset_or = popt
    x_offset = corner_coords[0]+x_offset_or
    y_offset = corner_coords[1]+y_offset_or
    R_err, x_offset_err, y_offset_err = np.sqrt(np.diagonal(pcov))
    #As True_Sigma is turned off, the errors are determined in the fitting process to give a reduced chi-squared value of 1.
    #So, these errors could be extracted and put on the plots below
   
    if plot == True:
        # theta_fit_array = np.linspace(np.amin(theta),np.amax(theta),200)
        theta_fit_array = np.linspace(0,2*pi,200)
        r_fit_array = polar_circle_func(theta_fit_array,R, x_offset_or, y_offset_or)
        cart_fit_array = PolarCart(theta_fit_array,r_fit_array,corner_coords[0],corner_coords[1])
        
        #Results plotted on top of the image
        plt.figure()
        pixels_im = plt.imshow(pixels.T)
        cbar = plt.colorbar(mappable=pixels_im)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(label='Normalised intensity',fontsize=fontsize)
        
        plt.scatter(outer_ring_pos[:,0],outer_ring_pos[:,1], color='black',marker='x',s=2,zorder=15)
        # plt.scatter(outer_ring_pos[:,0],circle_func(outer_ring_pos[:,0],R_guess,x_offset_guess,y_offset_guess),marker='.',color='orange',zorder=3)
        plt.plot(cart_fit_array[:,0],cart_fit_array[:,1],color='white',zorder=6)
        plt.errorbar(x_offset,y_offset,xerr=x_offset_err,yerr=y_offset_err,marker='x',color='white')
        
        plt.ylim([0,1023])
        plt.xlim([0,1280])
        plt.gca().set_aspect('equal')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel('Image x-axis (pixels)',fontsize=fontsize)
        plt.ylabel('Image y-axis (pixels)',fontsize=fontsize)
        if save_figs == True:
            plt.savefig(path[:-4]+'_centre'+'.svg',dpi=300,bbox_inches='tight')
        plt.show()

        #Graph showing the fitting function
        plt.figure()
        plt.plot(theta_fit_array,r_fit_array,zorder=0)
        plt.scatter(outer_ring_polar[:,0],outer_ring_polar[:,1],marker='x',color='black',zorder=1)
        plt.xlim([0,2*pi])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
                   labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
        plt.xlabel(r'Angle, $\theta$ (radians)',fontsize=fontsize,labelpad=-5)
        plt.ylabel('Distance, $r$ (pixels)',fontsize=fontsize)
        if save_figs == True:
            plt.savefig(path[:-4]+'_function'+'.svg',dpi=300,bbox_inches='tight')
        plt.show()
    return R, R_err, x_offset, x_offset_err, y_offset, y_offset_err

def avg_arrays(low_arr,low_arr_err, high_arr,high_arr_err, r, corner_coords, R, safety_frac=0.1):
    new_arr = low_arr.copy()
    new_arr_err = low_arr_err.copy()
    
    saturated_r = r[high_arr_err >= np.inf]
    sat_r_len = len(saturated_r)
    sat_limit = int(np.floor(2*np.pi*R)) #The number of pixels in one ring at the fitted radius
    
    if sat_r_len <= sat_limit:
        
        # new_arr, new_arr_err = weighted_avg(np.array([low_arr,high_arr]),np.array([low_arr_err,high_arr_err]),axis=0)
        #Attempted method but it does not work when two subsequent images have the exact same value for a pixel bringing the error to zero
        
        new_arr = np.average([low_arr,high_arr],weights=[1/low_arr_err,1/high_arr_err],axis=0)
        #Note, averaging may flatten the profile if there is a spatial time variance. This is assumed not to be the case from comparisons of the data
        new_arr_err = 1/np.sqrt(1/low_arr_err**2+1/high_arr_err**2)
        
        # plt.figure()
        # plt.errorbar(np.linspace(0,1280,1280),high_arr[:,512],yerr=high_arr_err[:,512])
        # plt.ylim([0,0.05*40000])
        # plt.xlim([corner_coords[0]-10,corner_coords[0]+10])
        # plt.show()
        inner_r = np.amax(r)
    else:
        inner_r = np.min(saturated_r)*(1-safety_frac) #The radius of the ring to be averaged

        inner_indices = r <= inner_r
        # new_arr[inner_indices], new_arr_err[inner_indices] = weighted_avg(np.array([low_arr[inner_indices],high_arr[inner_indices]]),np.array([low_arr_err[inner_indices],high_arr_err[inner_indices]]),axis=0)
        
        new_arr[inner_indices] = np.average([low_arr[inner_indices],high_arr[inner_indices]],weights=[1/low_arr_err[inner_indices],1/high_arr_err[inner_indices]],axis=0)
        #Note, averaging may flatten the profile if there is a spatial time variance. This is assumed not to be the case from comparisons of the data
        new_arr_err[inner_indices] = 1/np.sqrt(1/low_arr_err[inner_indices]**2+1/high_arr_err[inner_indices]**2)

        # plt.figure()
        # plt.errorbar(np.linspace(0,1280,1280),high_arr[:,512],yerr=high_arr_err[:,512])
        # plt.plot([corner_coords[0]+inner_r,corner_coords[0]+inner_r],[0,0.01*40000],color='orange')
        # plt.plot([corner_coords[0]-inner_r,corner_coords[0]-inner_r],[0,0.01*40000],color='orange')
        # plt.ylim([0,0.05*40000])
        # plt.xlim([corner_coords[0]-10,corner_coords[0]+10])
        # plt.show()
    
    return new_arr, new_arr_err, inner_r

def avg_area(pixels,pixels_err,r,r_err,theta,theta_err,theta_low,theta_high,delta_r):
    #Select and sort the pixels in the relevant area
    if theta_low > theta_high: #If the range spans the 2*pi bounday
        area_bool = np.logical_or(theta + theta_err >= theta_low, theta - theta_err < theta_high)
    else:
        area_bool = np.logical_and(theta + theta_err >= theta_low, theta - theta_err < theta_high)

    area_r = r[area_bool]
    sorted_indices = np.argsort(area_r)
    area_r = area_r[sorted_indices]
    area_r_err = r_err[area_bool][sorted_indices]
    area_pixels = pixels[area_bool][sorted_indices]
    area_pixels_err = pixels_err[area_bool][sorted_indices]
    
    #Average the pixels into the right sampling array
    r_intervals = np.arange(np.min(area_r),np.max(area_r),delta_r)
    r_len = len(r_intervals)-1
    
    new_r = np.empty(r_len)
    new_r_err = np.empty(r_len)
    new_pixels = np.empty(r_len)
    new_pixels_err = np.empty(r_len)
    
    for k in range(r_len):
        indices = np.logical_and(area_r >= r_intervals[k], area_r < r_intervals[k+1])
        unique_pix, unique_counts = np.unique(area_pixels[indices],return_counts=True)
        #Tests to make sure the averaged array is longer than 1 and all values are not the same -> otherwise we get 0 errors
        if indices.sum() > 1 and len(unique_counts) > 1:
            new_pixels[k], new_pixels_err[k] = weighted_avg(area_pixels[indices], area_pixels_err[indices])
            new_r[k], new_r_err[k] = weighted_avg(area_r[indices], area_r_err[indices])
        else:
            new_pixels[k] = np.average(area_pixels[indices],weights=1/area_pixels_err[indices])
            new_pixels_err[k] = 1/np.sqrt(np.sum(1/area_pixels_err[indices]**2))
            new_r[k] = np.average(area_r[indices],weights=1/area_r_err[indices])
            new_r_err[k] = 1/np.sqrt(np.sum(1/area_r_err[indices]**2))
            
    return new_r, new_r_err, new_pixels, new_pixels_err, area_r, area_r_err, area_pixels, area_pixels_err

def avg_sections(pixels,pixels_err,r,r_err,theta,theta_err,theta_low,theta_high,R,R_err,delta_r,darkness_limit=0.05,drop_limit=0,averaging_int_size=10):
    #Select and sort the pixels in the relevant area
    new_r, new_r_err, new_pixels, new_pixels_err, area_r, area_r_err, area_pixels, area_pixels_err = avg_area(pixels,pixels_err,r,r_err,theta,theta_err,theta_low,theta_high,delta_r)


    #Average the pixels to get the general profile
    pixel_avg = interval_avg(new_pixels, averaging_int_size) #Take the averages over every -interval- points
    
    avg_diff = pixel_avg[1:]-pixel_avg[:-1]
    beyond_dark_index = np.asarray((pixel_avg > darkness_limit).nonzero())[0,0]
    
    avg_peak_index = (np.asarray((avg_diff[beyond_dark_index:] < drop_limit).nonzero())[0,0]+beyond_dark_index)*averaging_int_size
    peak_index = avg_peak_index-averaging_int_size+np.argmax(new_pixels[avg_peak_index-averaging_int_size:avg_peak_index+averaging_int_size+1]) 
    #The range is made larger to include more in front of the beam as well as all that is averaged over (to negate the effect of the sampling)
    
    #Determine the useful parameters:
    peak_intensity = new_pixels[peak_index]
    peak_intensity_err = new_pixels_err[peak_index] #A better error would be including the error due to the max finding algorithm -> requires Monte Carlo methods
    peak_pos = new_r[peak_index]/R
    peak_pos_err = np.sqrt(new_r_err[peak_index]**2 + (R_err*peak_pos)**2)/R
    
    if peak_intensity == 1: #Cutting off the range we're looking at where the image saturates
        peak_index -= 1
        
    new_r = new_r[:peak_index]/R
    new_r_err = np.sqrt(new_r_err[:peak_index]**2 + (R_err*new_r)**2)/R
    new_pixels = new_pixels[:peak_index]
    new_pixels_err = new_pixels_err[:peak_index]
    
    area_r /= R
    area_r_err = np.sqrt(area_r_err**2 + (R_err*area_r)**2)/R
    

    return new_r, new_r_err, new_pixels, new_pixels_err, peak_intensity, peak_intensity_err, peak_pos, peak_pos_err, area_r, area_r_err, area_pixels, area_pixels_err
    

def CartPolar(cart_coords):
    r = np.sqrt(cart_coords[:,:,0]**2+cart_coords[:,:,1]**2)
    theta=np.arctan(cart_coords[:,:,1]/cart_coords[:,:,0])
    del_theta = np.sqrt(0.5)/r #this is half the diameter of a pixel as an interval in theta at a certain distance
    #Half the size of a pixel in cartesian coordinates is sqrt(0.5^2+0.5^2), half the diagonal
    if np.any(np.isnan(theta).ravel()):
        print("There are 'NaN' values in the array, change the origin coords so that it doesn't fall exactly on a pixel")
    return r, theta, del_theta

def CartPolar2(cart_coords):
    r = np.sqrt(cart_coords[:,:,0]**2+cart_coords[:,:,1]**2)
    theta=np.arctan2(cart_coords[:,:,1],cart_coords[:,:,0])
    theta[theta < 0] = theta[theta < 0] + 2*pi
    del_theta = np.sqrt(0.5)/r #this is half the diameter of a pixel as an interval in theta at a certain distance
    #Half the size of a pixel in cartesian coordinates is sqrt(0.5^2+0.5^2), half the diagonal
    if np.any(np.isnan(theta).ravel()):
        print("There are 'NaN' values in the array, change the origin coords so that it doesn't fall exactly on a pixel")
    return r, theta, del_theta

def CartPolar3(cart_coords,xerr,yerr):
    r = square_fn(cart_coords[:,:,0],cart_coords[:,:,1])
    r_err = np.sqrt((square_fn(cart_coords[:,:,0]+xerr,cart_coords[:,:,1])-r)**2 + (square_fn(cart_coords[:,:,0],cart_coords[:,:,1]+yerr)-r)**2)
    
    theta=np.arctan2(cart_coords[:,:,1],cart_coords[:,:,0])
    theta_err = np.sqrt((np.arctan2(cart_coords[:,:,1]+yerr,cart_coords[:,:,0])-theta)**2 + (np.arctan2(cart_coords[:,:,1],cart_coords[:,:,0]+xerr)-theta)**2)
    theta[theta < 0] = theta[theta < 0] + 2*pi
    
    
    # del_theta = np.sqrt(0.5)/r #this is half the diameter of a pixel as an interval in theta at a certain distance
    # #Half the size of a pixel in cartesian coordinates is sqrt(0.5^2+0.5^2), half the diagonal
    
    # plt.figure()
    # plt.plot(r[:,512],del_theta[:,512])
    # plt.plot(r[:,512],theta_err[:,512])
    # plt.plot(r[:,512],0.5/r[:,512])
    # plt.ylim([0,0.005])
    # plt.show()
    if np.any(np.isnan(theta).ravel()):
        print("There are 'NaN' values in the array, change the origin coords so that it doesn't fall exactly on a pixel")
    return r, r_err, theta, theta_err

def square_fn(x,y):
    return np.sqrt(x**2+y**2)

def PolarCart(theta,r,x_offset,y_offset):
    #This inverses the process of CartPolar2
    return np.array([x_offset+r*np.cos(theta),y_offset+r*np.sin(theta)]).T

def linear(x,m,a):
    return m*x+a

def power_law(x,m,a,b):
    return a*(x**m)+b

def inv_power_law(y,m,a,b):
    return ((y-b)/a)**(1/m)

def weighted_avg(param_array,err_array):
    N = len(param_array)
    weights = 1/err_array
    avg_param = np.average(param_array,weights=weights)
    weighted_SE = np.sqrt(np.average((param_array-avg_param)**2,weights=weights)/(N-1))
    #std = sqrt(sum(weight*(x-x_mean)**2)/((N-1)*sum(weight)/N))
    #SE = std/sqrt(N)
    return avg_param, weighted_SE

def weighted_avg_2D_1(param_array,err_array):
    N = np.shape(param_array)[1]
    weights = 1/err_array
    avg_param = np.average(param_array,weights=weights,axis=1)
    weighted_SE = np.sqrt(np.average(((param_array.T-avg_param).T)**2,weights=weights,axis=1)/(N-1))
    return avg_param, weighted_SE

def weighted_avg_2D_0(param_array,err_array):
    N = np.shape(param_array)[0]
    weights = 1/err_array
    avg_param = np.average(param_array,weights=weights,axis=0)
    weighted_SE = np.sqrt(np.average(((param_array-avg_param))**2,weights=weights,axis=0)/(N-1))
    return avg_param, weighted_SE

def SE_to_std(param_array, SE):
    N = len(param_array)
    std = SE * np.sqrt(N)
    std_err = std/np.sqrt(2*N-2) #Error in SE is SE/sqrt(2*N-2) -> so error in std is std/sqrt(2*N-2)
    return std, std_err

def weighted_std(param_array,err_array):
    N = len(param_array)
    weights = 1/err_array
    avg_param = np.average(param_array,weights=weights)
    std = np.sqrt(np.average((param_array-avg_param)**2,weights=weights)/((N-1)/N))
    std_err = std/np.sqrt(2*N-2)
    return std, std_err

def weighted_resid_std(residuals,residual_errs):
    N = len(residuals)
    weights = 1/residual_errs
    std = np.sqrt(np.average(residuals**2,weights=weights)/((N-1)/N))
    std_err = std/np.sqrt(2*N - 2)
    return std, std_err
                  
def chauvenet(param_array,err_array):
    N = len(param_array)
    weights = 1/err_array
    mean = np.mean(param_array)
    std = np.std(param_array)
    
    mean = np.average(param_array,weights=weights)
    std = np.sqrt(np.average((param_array-mean)**2,weights=weights)/((N-1)/N))
    n_out = norm.sf(param_array,loc=mean,scale=std)*N
    reduced_indices = n_out >= 0.5
    plt.figure()
    plt.plot(n_out)
    plt.ylim(0,10)
    plt.show()
    return reduced_indices

def plot_data(theta_array,m_array,m_err_array,avg_m,avg_m_err,xlim,line_colour,dotted_colour):
    plt.errorbar(theta_array,m_array,yerr=m_err_array,marker='x',color='black',capsize=2,linestyle='',zorder=5)

    avg_line_x = [xlim[0],xlim[-1]]
    avg_line_y = [avg_m,avg_m]
    avg_line_y_high = [avg_m+avg_m_err,avg_m+avg_m_err]
    avg_line_y_low = [avg_m-avg_m_err,avg_m-avg_m_err]
    plt.plot(avg_line_x,avg_line_y,linestyle='--', color=line_colour, alpha=0.8)

    plt.plot(avg_line_x,avg_line_y_high, linestyle=':', color=dotted_colour)
    plt.plot(avg_line_x,avg_line_y_low, linestyle=':', color=dotted_colour)
    plt.fill_between(avg_line_x,avg_line_y_low,avg_line_y_high, color=line_colour,alpha=0.2)
    return

def plot_resid(norm_resid, norm_resid_err, x, plot, D, xmin, xmax, font_size, resid_y_lim):
    plot.plot([xmin,xmax],[D, D],[xmin,xmax],[-D,-D],[xmin,xmax],[0,0], color =(0.1,0.1,0.1), linestyle = '--', linewidth = 1, zorder = 1)
    plt.fill_between([xmin,xmax],[D, D],[-D, -D], facecolor=(0.9,0.9,0.9), interpolate=True, zorder = 0)
    plot.errorbar(x, norm_resid, yerr= norm_resid_err, capsize = 2, fmt='x', ms = 6, color = 'black', zorder = 3)
    
    plt.xlim([xmin,xmax])
    plt.ylim([-resid_y_lim,resid_y_lim])
    
    yticks = np.linspace(-resid_y_lim*0.6,resid_y_lim*0.6,3)
    # plt.yticks(yticks[:-1],fontsize = font_size)
    plt.yticks(yticks,fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plot.tick_params(axis='both', which='major', width= 1.3, length= 5)
    plot.minorticks_on()
    plt.ylabel("Normalised Residuals", fontsize = font_size) 
    return
    
