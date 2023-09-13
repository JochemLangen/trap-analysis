# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:48:22 2023

@author: joche
"""
from funcs_photos import *
import pandas as pd

path = "Second_beam\Hollow_beam_bottom_left_corner.bmp"
# path = "New_col\\Flipped_ring_axicon\\Hollow_beam_39577_2.pgm"
im = Image.open(path)
pixels = np.array(im).T
im_shape = np.shape(pixels)
max_intensity = np.amax(pixels)
pixels = pixels/max_intensity
pixel_err = 0.5/max_intensity
r_radial_err = 0.5 #Pixel width
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
# theta_limits = np.array([np.amin(outer_theta)+0.03,np.amax(outer_theta)-0.05])
theta_limits= [theta[0,-1],theta[-1,0]]
# theta_ind = [theta[-1,512],theta[-1,513]]
del_R = 0.5/R

no_theta_points = int(np.floor((theta_limits[1]-theta_limits[0])/(2*del_R)))
print(no_theta_points)
no_theta_points = 5

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

#Setting up the arrays for all the results
m_array = np.empty(no_theta_points)
m_err_array = np.empty(no_theta_points)
a_array = np.empty(no_theta_points)
a_err_array = np.empty(no_theta_points)
std_array = np.empty(no_theta_points)
std_err_array = np.empty(no_theta_points)
peak_intensity = np.empty(no_theta_points)
peak_intensity_err = np.empty(no_theta_points)
peak_pos = np.empty(no_theta_points)
peak_pos_err = np.empty(no_theta_points)
peak_diff_pos = np.empty(no_theta_points)
peak_diff_pos_err = np.empty(no_theta_points)
trough_diff_pos = np.empty(no_theta_points)
trough_diff_pos_err = np.empty(no_theta_points)


averaging_int_size = 4 #The number of points over which it is averaged to get a smooth curve
peak_size =averaging_int_size #The +/- area around the first value above this limit in which the max value is taken as the peak
subplots = True
plots = True
prints = False
difference_limit = 0.01
darkness_limit = 0.05
method = 'difference' #The method used to determine where the trough is
fontsize=10

for i in range(no_theta_points):
    r_radials,pixel_radials,peak_intensity[i],peak_intensity_err[i],peak_pos[i],peak_pos_err[i]=find_outer_peak(r,pixels,theta,del_theta,theta_array[i],R,R_err,r_radial_err,max_intensity)
    
    pixel_avg = interval_avg(pixel_radials, averaging_int_size) #Take the averages over every -interval- points
    radial_avg = interval_avg(r_radials, averaging_int_size)
    
    avg_diff = pixel_avg[:-1]-pixel_avg[1:] #Find the difference between these points
    avg_peak_index = np.argmax(avg_diff)
    
    if method == 'darkness':
        low_diff_indices = np.argwhere(pixel_avg[avg_peak_index:]<darkness_limit)
    elif method == 'difference':
        low_diff_indices = np.argwhere(avg_diff[avg_peak_index:]<difference_limit)
    else:
        print("No valid trough index selection method has been chosen!")
    
    if len(low_diff_indices) == 0:
        avg_trough_index = len(pixel_radials)-1-peak_size
    else:
        avg_trough_index = int((low_diff_indices[0]+avg_peak_index)*averaging_int_size)
    avg_peak_index *= averaging_int_size
    
    

    if subplots == True:
        plt.figure()
        plt.plot([radial_avg[0],radial_avg[-1]],[difference_limit,difference_limit],color='black',linestyle='--',label='Difference limit')
        plt.plot(radial_avg[:-1],avg_diff)
        plt.scatter(radial_avg[int(avg_peak_index/averaging_int_size)],avg_diff[int(avg_peak_index/averaging_int_size)],color='orange',label='Peak index')
        plt.scatter(radial_avg[int(avg_trough_index/averaging_int_size)],avg_diff[int(avg_trough_index/averaging_int_size)],color='black',label='Trough index')
        plt.legend()
        plt.ylabel("Difference in averaged normalised pixel intensities",fontsize=fontsize)
        plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
        plt.show()

    
        plt.figure()
        plt.plot([radial_avg[0],radial_avg[-1]],[darkness_limit,darkness_limit],color='black',linestyle='--',label='Darkness limit')
        
        plt.plot(radial_avg,pixel_avg)
        plt.scatter(radial_avg[int(avg_peak_index/averaging_int_size)],pixel_avg[int(avg_peak_index/averaging_int_size)],color='orange',label='Peak index')
        plt.scatter(radial_avg[int(avg_trough_index/averaging_int_size)],pixel_avg[int(avg_trough_index/averaging_int_size)],color='black',label='Trough index')
        plt.legend()
        plt.ylabel("Averaged normalised pixel intensities",fontsize=fontsize)
        plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
        plt.ylim([0,np.amax(pixel_avg)])
        plt.show()
    


    #Get the index of the point furthest out where this difference exceeds the reference value
    if avg_peak_index >= peak_size:
        peak_pixels = pixel_radials[avg_peak_index:avg_peak_index+peak_size+1]
        
        # peak_r = r_radials[avg_peak_index-peak_size:avg_peak_index+peak_size]
        pixel_diff = peak_pixels[:-1]-peak_pixels[1:]#/(peak_r[:-1]-peak_r[1:])
        peak_index = avg_peak_index + np.argmax(pixel_diff)
        peak_diff_pos[i] = r_radials[peak_index]
        peak_diff_pos_err[i] = peak_diff_pos[i]/R*np.sqrt((r_radial_err/peak_diff_pos[i])**2+R_err**2)
        
        if subplots == True:
            r_peak = r_radials[avg_peak_index:avg_peak_index+peak_size]
            plt.figure()
            plt.plot(r_radials,pixel_radials)
            plt.scatter(r_peak,peak_pixels[:-1],label='Peak pixels')
            plt.plot(r_peak,pixel_diff,label='Peak pixel differences')
            plt.scatter(peak_diff_pos[i],pixel_radials[peak_index],marker='x',color='black')
            plt.xlim([r_peak[0]-0.03,r_peak[-1]+0.03])
            plt.legend()
            plt.ylabel("(Difference in) Normalised pixel intensities",fontsize=fontsize)
            plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
            plt.show()
    else:
        peak_pixels = pixel_radials[0:avg_peak_index+peak_size+1]
        peak_r = r_radials[0:avg_peak_index+peak_size]
        pixel_diff = peak_pixels[:-1]-peak_pixels[1:]#/(peak_r[:-1]-peak_r[1:]) 
        peak_index = avg_peak_index + np.argmax(pixel_diff)
        peak_diff_pos[i] = r_radials[peak_index]
        peak_diff_pos_err[i] = peak_diff_pos[i]/R*np.sqrt((r_radial_err/peak_diff_pos[i])**2+R_err**2)
        
        if subplots == True:
            r_peak = r_radials[0:avg_peak_index+peak_size]
            
            plt.figure()
            plt.plot(r_radials,pixel_radials)
            plt.scatter(r_peak,peak_pixels[:-1],label='Peak pixels')
            plt.plot(r_peak,pixel_diff,label='Peak pixel differences')
            plt.scatter(peak_diff_pos[i],pixel_radials[peak_index],marker='x',color='black')
            plt.xlim([r_peak[0]-0.03,r_peak[-1]+0.03])
            plt.legend()
            plt.ylabel("(Difference in) Normalised pixel intensities",fontsize=fontsize)
            plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
            plt.show()
     
    trough_pixels = pixel_radials[avg_trough_index:avg_trough_index+peak_size+1]
    trough_r = r_radials[avg_trough_index:avg_trough_index+peak_size]
    tr_pixel_diff = trough_pixels[:-1]-trough_pixels[1:]
    trough_index = avg_trough_index + np.argmin(tr_pixel_diff)
    trough_diff_pos[i] = r_radials[trough_index]
    trough_diff_pos_err[i] = trough_diff_pos[i]/R*np.sqrt((r_radial_err/trough_diff_pos[i])**2+R_err**2)
    
    if subplots == True:
        trough_r = r_radials[avg_trough_index:avg_trough_index+peak_size]
        plt.figure()
        plt.plot(r_radials,pixel_radials)
        plt.scatter(trough_r,trough_pixels[:-1],label='Trough pixels')
        plt.plot(trough_r,tr_pixel_diff,label='Trough pixel differences')
        plt.scatter(trough_diff_pos[i],pixel_radials[trough_index],marker='x',color='black')
        plt.xlim([trough_r[0]-0.03,trough_r[-1]+0.03])
        plt.legend()
        plt.ylabel("(Difference in) Normalised pixel intensities",fontsize=fontsize)
        plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
        plt.show()
        
    
    final_pixels = pixel_radials[peak_index:trough_index+1]
    final_r = r_radials[peak_index:trough_index+1]
    #Remove infs and nans in log
    finite_indices = final_pixels > 0
    final_pixels = final_pixels[finite_indices]
    final_r = final_r[finite_indices]
    
    log_final_pixels = np.log(final_pixels)
    log_final_r = np.log(final_r)
    
    
    
    
    m_guess = 50
    a_guess = 0
    popt, pcov = curve_fit(linear,log_final_r,log_final_pixels,p0=[m_guess,a_guess],bounds=([0,-np.inf],np.inf))
    m_array[i], a_array[i] = popt
    m_err_array[i], a_err_array[i] = np.sqrt(np.diagonal(pcov))

    
    hollow_beam_pixels = pixel_radials[trough_index+1:]
    hollow_beam_r = r_radials[trough_index+1:]
    fitted_intensities = np.exp(linear(np.log(hollow_beam_r),m_array[i],a_array[i]))

    hollow_difference = hollow_beam_pixels - fitted_intensities
    
    # std_array[i] = np.std(hollow_beam_pixels)
    std_array[i] = np.std(hollow_difference)
    std_err_array[i] = std_array[i]/np.sqrt(2*len(hollow_beam_pixels)-2)
    
    if peak_pos[i] > 1.02 or peak_pos[i] < 0.98:
        xlim = [final_r[-1]-0.1,r_radials[0]+0.03]
        ylim = [final_pixels[-1]*0.05,1]
        r_values = np.linspace(xlim[0],xlim[-1],100)
        fitted_intensities = np.exp(linear(np.log(r_values),m_array[i],a_array[i]))
        
        plt.figure()
        plt.plot(r_values,fitted_intensities,color='purple',label="Power law fit")
        plt.scatter(r_radials,pixel_radials,label='All data')
        plt.scatter([r_radials[trough_index],r_radials[peak_index]],[pixel_radials[trough_index],pixel_radials[peak_index]],marker='x',color='black')
        plt.title(theta_array[i])
        plt.plot(final_r,final_pixels, color='red',zorder=5, label='Fitting pixels')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.ylabel("Normalised pixel intensities",fontsize=fontsize)
        plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
        plt.show()
        
        
    if prints == True:
        print("m = {} +/- {}".format(m_array[i],m_err_array[i]))
        print("a = {} +/- {}".format(a_array[i],a_err_array[i]))
        print("Standard deviation around fitted power law = {}".format(std_array[i]))
    
    if plots == True:
        
        xlim = [final_r[-1]-0.06,r_radials[0]+0.03]
        ylim = [final_pixels[-1]*0.2,1]
        r_values = np.linspace(xlim[0],xlim[-1],100)
        fitted_intensities = np.exp(linear(np.log(r_values),m_array[i],a_array[i]))
        if subplots == True:
            plt.figure()
            plt.plot(r_values,fitted_intensities,color='purple',label="Power law fit",zorder=0)
            plt.scatter(r_radials,pixel_radials,label='All data',zorder=1)
            plt.title(theta_array[i])
            plt.scatter(final_r,final_pixels, color='red',label='Fitting pixels',zorder=2)
            plt.scatter([r_radials[trough_index],r_radials[peak_index]],[pixel_radials[trough_index],pixel_radials[peak_index]],marker='x',color='black',zorder=3)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend()
            plt.ylabel("Normalised pixel intensities",fontsize=fontsize)
            plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
            plt.show()
            
            #Plot the histogram of the inside of the hollow beam
            bin_no = 60

            plt.figure()
            plt.hist(hollow_difference,bins=bin_no,range=[-0.1,0.2],density=True)
            plt.ylabel("Normalised probability density",fontsize=fontsize)
            plt.xlabel("Normalised pixel intensity residuals",fontsize=fontsize)
            plt.show()
            
        #Plot the log plot of the data
        plt.figure()
        plt.plot(r_values,fitted_intensities,color='purple',label="Power law fit",zorder=0)
        plt.scatter(r_radials,pixel_radials,label='All data',zorder=1)
        plt.title(theta_array[i])
        plt.scatter(final_r,final_pixels, color='red', label='Fitting pixels',zorder=2)
        plt.scatter([r_radials[trough_index],r_radials[peak_index]],[pixel_radials[trough_index],pixel_radials[peak_index]],marker='x',color='black',zorder=3)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel("(Difference in) Normalised pixel intensities",fontsize=fontsize)
        plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
        plt.show()
        
        


avg_m, avg_m_err = weighted_avg(m_array,m_err_array)
print("Average power law exponent: {} +/- {}".format(avg_m,avg_m_err))
avg_R, avg_R_err = weighted_avg(peak_pos,peak_pos_err)
print("Average normalised ring radius: {} +/- {}".format(avg_R,avg_R_err))
avg_grad_peak, avg_grad_peak_err = weighted_avg(peak_diff_pos,peak_diff_pos_err)
print("Average normalised gradient peak radius: {} +/- {}".format(avg_grad_peak,avg_grad_peak_err))
avg_trough_peak, avg_trough_peak_err = weighted_avg(trough_diff_pos,trough_diff_pos_err)
print("Average normalised gradient trough radius: {} +/- {}".format(avg_trough_peak,avg_trough_peak_err))

plt.figure()
plt.errorbar(theta_array,m_array,yerr=m_err_array,marker='x',color='black',capsize=2,linestyle='')
plt.ylabel("Fitted power law exponent",fontsize=fontsize)
plt.xlabel("Angle (radians)",fontsize=fontsize)
plt.show()


plt.figure()
plt.errorbar(theta_array,a_array,yerr=a_err_array,marker='x',color='black',capsize=2,linestyle='')
plt.ylabel("Fitted power law coefficient",fontsize=fontsize)
plt.xlabel("Angle (radians)",fontsize=fontsize)
plt.show()

plt.figure()
plt.errorbar(theta_array,std_array,yerr=std_err_array,marker='x',color='black',capsize=2,linestyle='')
plt.ylabel("Residual standard deviation",fontsize=fontsize)
plt.xlabel("Angle (radians)",fontsize=fontsize)
plt.show()

plt.figure()
plt.errorbar(theta_array,peak_intensity,yerr=peak_intensity_err,marker='x',color='black',capsize=2,linestyle='')
plt.ylabel("Normalised peak intensity",fontsize=fontsize)
plt.xlabel("Angle (radians)",fontsize=fontsize)
plt.show()

plt.figure()
plt.errorbar(theta_array,peak_pos,yerr=peak_pos_err,marker='x',color='black',capsize=2,linestyle='')
plt.ylabel("Normalised peak position, r/R",fontsize=fontsize)
plt.xlabel("Angle (radians)",fontsize=fontsize)
plt.show()

plt.figure()
plt.errorbar(theta_array,peak_pos,yerr=peak_pos_err,marker='x',color='black',capsize=2,linestyle='',label='Ring peak')
plt.errorbar(theta_array,peak_diff_pos,yerr=peak_diff_pos_err,marker='x',color='blue',capsize=2,linestyle='',label='Gradient peak')
plt.errorbar(theta_array,trough_diff_pos,yerr=trough_diff_pos_err,marker='x',color='orange',capsize=2,linestyle='',label='Gradient trough')

plt.ylabel("Normalised position, r/R",fontsize=fontsize)
plt.xlabel("Angle (radians)",fontsize=fontsize)
plt.show()




# reduced_indices = chauvenet(peak_pos,peak_pos_err)
# peak_pos = peak_pos[reduced_indices]
# peak_pos_err = peak_pos_err[reduced_indices]
# theta_array = theta_array[reduced_indices]

# plt.figure()
# plt.errorbar(theta_array,peak_pos,yerr=peak_pos_err,marker='x',color='black',capsize=2,linestyle='')
# plt.ylabel("Normalised peak position, r/R",fontsize=fontsize)
# plt.xlabel("Angle (radians)",fontsize=fontsize)
# plt.show()
