# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 23:01:59 2023

@author: joche
"""
from funcs_photos import *
import pandas as pd
pi = np.pi

# path = "Second_beam\Hollow_beam_bottom_left_corner.bmp"
path = "New_col\\Flipped_ring_axicon\\Hollow_beam_39577_2.pgm"
# path = "New_col\\Ring_axicon_lens\\Hollow_beam_39577us.pgm"
# path = "New_col\\Ring_lens_axicon\\axicon_2\\Hollow_beam_84370us.pgm"
# path = "New_col\\Ring_lens_axicon\\axicon_3\\Hollow_beam_84370us.pgm"
# path = "New_col\\Ring_lens_axicon\\Hollow_beam_84370us.pgm"
path = "New_col\\Flipped_ring_axicon\\Hollow_beam_84370_2.pgm"
path = "New_col\\Flip_focus_ring_axicon\\Hollow_beam_10413us_7000mm.pgm"
path = "New_col\\Flip_focus_ring_axicon\\Collimated\\moved_camera\\Hollow_beam_4166us_8000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_1000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_2000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_3000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_4000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_6000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_8000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_10000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_12000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_14000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_16000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_18000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_20000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_0000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_2000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_4000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_6000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_8000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_10000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_12000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_14000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_16000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_18000mm.pgm"
# path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_20000mm.pgm"
# path="New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\Hollow_beam_24998us_10000mm.pgm"
im = Image.open(path)
pixels = np.array(im).T
im_shape = np.shape(pixels)
max_intensity = np.amax(pixels)
pixels = pixels/max_intensity
pixel_err = 2**6/max_intensity #4 for the 16-bit images with 10-bit accuracy
r_radial_err = 0.5 #Pixel width
pixel_size = np.average([6.14/1280,4.9/1024]) #In mm
print(im_shape)
ring_type = 'inner'

csv_file = pd.read_csv(path[:-4]+'.csv')
R, R_err, x_offset, x_offset_err, y_offset, y_offset_err = csv_file.values[0,1:7]

#Define Cartesian Coordinates: We take them as the centre of each pixel
x = np.arange(im_shape[0])-x_offset
y = np.arange(im_shape[1])-y_offset
cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])


#Convert to Polar coordinate system:
r, theta, del_theta = CartPolar2(cart_coords)

#Find the outer pattern:
ring_arg = np.logical_and(r <= R+R_err+r_radial_err, r <= R - R_err - r_radial_err)
outer_theta = theta[ring_arg]
outer_del_theta = del_theta[ring_arg]
min_theta_arg = np.argmin(outer_theta)
max_theta_arg = np.argmax(outer_theta)
theta_limits = np.array([outer_theta[min_theta_arg]+outer_del_theta[min_theta_arg],outer_theta[max_theta_arg]-outer_del_theta[max_theta_arg]])
# theta_limits= [theta[0,-1],theta[-1,0]]
# theta_limits = [theta[-1,512],theta[-1,513]]
del_R = 0.5/R
# r_radial_err /= R -> this is now done in the function itself

no_theta_points = int(np.floor((theta_limits[1]-theta_limits[0])/(2*del_R)))
print(no_theta_points)
no_theta_points = 100

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
b_array = np.empty(no_theta_points)
b_err_array = np.empty(no_theta_points)
std_array = np.empty(no_theta_points)
std_err_array = np.empty(no_theta_points)
peak_intensity = np.empty(no_theta_points)
peak_intensity_err = np.empty(no_theta_points)
peak_pos = np.empty(no_theta_points)
peak_pos_err = np.empty(no_theta_points)
peak_diff_pos = np.empty(no_theta_points)
peak_diff_pos_err = np.empty(no_theta_points)
# trough_diff_pos = np.empty(no_theta_points)
# trough_diff_pos_err = np.empty(no_theta_points)


averaging_int_size = 4 #The number of points over which it is averaged to get a smooth curve
peak_size =averaging_int_size #The +/- area around the first value above this limit in which the max value is taken as the peak
subplots = False
plots = False
prints = False
slice_plot = False
bad_fit_prints = False
# difference_limit = 0.01
darkness_limit = 0.2
method = 'difference' #The method used to determine where the trough is
fontsize=12

for i in range(no_theta_points):
    
    r_radials,pixel_radials,peak_intensity[i],peak_intensity_err[i],peak_pos[i],peak_pos_err[i]=find_inner_peak(r,pixels,theta,del_theta,theta_array[i],R,R_err,r_radial_err,max_intensity,ring_type,darkness_limit)
    
    pixel_avg = interval_avg(pixel_radials, averaging_int_size) #Take the averages over every -interval- points
    radial_avg = interval_avg(r_radials, averaging_int_size)
    
    avg_diff = pixel_avg[1:]-pixel_avg[:-1] #Find the difference between these points
    avg_peak_index = np.argmax(avg_diff)*averaging_int_size    
    

    if subplots == True:
        plt.figure()
        plt.plot(radial_avg[:-1],avg_diff)
        plt.scatter(radial_avg[int(avg_peak_index/averaging_int_size)],avg_diff[int(avg_peak_index/averaging_int_size)],color='orange',label='Peak index')
        plt.legend()
        plt.ylabel("Difference in averaged normalised pixel intensities",fontsize=fontsize)
        plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
        plt.show()

    
        plt.figure()
        plt.plot(radial_avg,pixel_avg)
        plt.scatter(radial_avg[int(avg_peak_index/averaging_int_size)],pixel_avg[int(avg_peak_index/averaging_int_size)],color='orange',label='Peak index')
        plt.legend()
        plt.ylabel("Averaged normalised pixel intensities",fontsize=fontsize)
        plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
        plt.ylim([0,np.amax(pixel_avg)])
        plt.show()
    


    #Get the index of the point furthest out where this difference exceeds the reference value
    if avg_peak_index >= peak_size:
        peak_pixels = pixel_radials[avg_peak_index-peak_size:avg_peak_index+peak_size+1]
        
        # peak_r = r_radials[avg_peak_index-peak_size:avg_peak_index+peak_size]
        pixel_diff = peak_pixels[1:]-peak_pixels[:-1]#/(peak_r[:-1]-peak_r[1:])
        peak_index = avg_peak_index-peak_size + np.argmax(pixel_diff)+1
        peak_diff_pos[i] = r_radials[peak_index]
        peak_diff_pos_err[i] = peak_diff_pos[i]/R*np.sqrt((r_radial_err/peak_diff_pos[i])**2+R_err**2)
        
        if subplots == True:
            r_peak = r_radials[avg_peak_index-peak_size+1:avg_peak_index+peak_size+1]
            plt.figure()
            plt.plot(r_radials,pixel_radials)
            plt.scatter(r_peak,peak_pixels[1:],label='Peak pixels')
            plt.plot(r_peak,pixel_diff,label='Peak pixel differences')
            plt.scatter(peak_diff_pos[i],pixel_radials[peak_index],marker='x',color='black')
            plt.xlim([r_peak[0]-0.03,r_peak[-1]+0.03])
            plt.legend()
            plt.ylabel("(Difference in) Normalised pixel intensities",fontsize=fontsize)
            plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
            plt.show()
    else:
        peak_pixels = pixel_radials[avg_peak_index-peak_size:]
        # peak_r = r_radials[avg_peak_index-peak_size:]
        pixel_diff = peak_pixels[1:]-peak_pixels[:-1]#/(peak_r[:-1]-peak_r[1:])
        peak_index = avg_peak_index-peak_size + np.argmax(pixel_diff)+1
        peak_diff_pos[i] = r_radials[peak_index]
        peak_diff_pos_err[i] = peak_diff_pos[i]/R*np.sqrt((r_radial_err/peak_diff_pos[i])**2+R_err**2)
        
        if subplots == True:
            r_peak = r_radials[avg_peak_index-peak_size+1:]
            
            plt.figure()
            plt.plot(r_radials,pixel_radials)
            plt.scatter(r_peak,peak_pixels[1:],label='Peak pixels')
            plt.plot(r_peak,pixel_diff,label='Peak pixel differences')
            plt.scatter(peak_diff_pos[i],pixel_radials[peak_index],marker='x',color='black')
            plt.xlim([r_peak[0]-0.03,r_peak[-1]+0.03])
            plt.legend()
            plt.ylabel("(Difference in) Normalised pixel intensities",fontsize=fontsize)
            plt.xlabel("Averaged normalised pixel radii, r/R",fontsize=fontsize)
            plt.show()
    
    final_pixels = pixel_radials[:peak_index]
    final_r = r_radials[:peak_index]
    #Remove infs and nans in log
    finite_indices = final_pixels > 0
    log_final_pixels = np.log(final_pixels[finite_indices])
    log_final_r = np.log(final_r[finite_indices])
    
    
    
    
    m_guess = 50
    a_guess = 0.001
    b_guess = 0.001
    try:
        popt, pcov = curve_fit(power_law,final_r,final_pixels,p0=[m_guess,a_guess,b_guess],bounds=(0,np.inf))
        m_array[i], a_array[i], b_array[i] = popt
        m_err_array[i], a_err_array[i], b_err_array[i] = np.sqrt(np.diagonal(pcov))
    except: 
        print("The fitting went wrong! This value has been removed.")
        m_array[i], a_array[i], b_array[i] = m_guess,a_guess,b_guess
        m_err_array[i], a_err_array[i], b_err_array[i] = (0,0,0)
    
    # darkness_r_fit = inv_power_law(darkness_limit,m_array[i], a_array[i], b_array[i])
    # trough_index = np.argmin(abs(r_radials-darkness_r_fit))
    # trough_diff_pos[i] = r_radials[trough_index]
    # trough_diff_pos_err[i] = trough_diff_pos[i]/R*np.sqrt((r_radial_err/trough_diff_pos[i])**2+R_err**2)
    
    # hollow_beam_pixels = pixel_radials[trough_index:]
    # hollow_beam_r = r_radials[trough_index:]
    hollow_beam_pixels = pixel_radials
    hollow_beam_r = r_radials
    
    fitted_intensities = power_law(hollow_beam_r,m_array[i], a_array[i], b_array[i])


    hollow_difference = hollow_beam_pixels - fitted_intensities
    
    # std_array[i] = np.std(hollow_beam_pixels)
    std_array[i] = np.std(hollow_difference)
    std_err_array[i] = std_array[i]/np.sqrt(2*len(hollow_beam_pixels)-2)
    
    if bad_fit_prints == True:
        if peak_pos[i] > 1.02 or peak_pos[i] < 0.98:
            xlim = [max(r_radials[0]*0.95,0.1),r_radials[-1]+0.03]
            ylim = [pixel_radials[0]*0.05,pixel_radials[-1]+0.1]
            r_values = np.linspace(xlim[0],xlim[-1],100)
    
            fitted_intensities = power_law(r_values,m_array[i], a_array[i], b_array[i])
    
                
            plt.figure()
            plt.plot(r_values,fitted_intensities,color='purple',label="Power law fit",zorder=0)
            plt.errorbar(r_radials,pixel_radials,xerr=r_radial_err,yerr=pixel_err,marker='x',capsize=2,linestyle='',label='All data',zorder=1)
            plt.title(theta_array[i])
            plt.errorbar(final_r,final_pixels,xerr=r_radial_err,yerr=pixel_err,marker='x',capsize=2,linestyle='', color='red',label='Fitting pixels',zorder=2)
            plt.scatter(r_radials[peak_index],pixel_radials[peak_index],marker='x',color='black',zorder=3)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend()
            plt.ylabel("Normalised pixel intensities",fontsize=fontsize)
            plt.xlabel("Normalised pixel radii, r/R",fontsize=fontsize)
            plt.show()
        
        
    if prints == True:
        print("m = {} +/- {}".format(m_array[i],m_err_array[i]))
        print("a = {} +/- {}".format(a_array[i],a_err_array[i]))
        print("b = {} +/- {}".format(b_array[i],b_err_array[i]))
        print("Standard deviation around fitted power law = {}".format(std_array[i]))
    
    if plots == True:
        
        xlim = [max(r_radials[0]*0.95,0.1),r_radials[-1]+0.03]
        ylim = [pixel_radials[0]*0.05,pixel_radials[-1]+0.1]
        r_values = np.linspace(xlim[0],xlim[-1],100)

        fitted_intensities = power_law(r_values,m_array[i], a_array[i], b_array[i])

        if subplots == True:
            plt.figure()
            plt.plot(r_values,fitted_intensities,color='purple',label="Power law fit",zorder=0)
            plt.errorbar(r_radials,pixel_radials,xerr=r_radial_err,yerr=pixel_err,marker='x',capsize=2,linestyle='',label='All data',zorder=1)
            plt.title(theta_array[i])
            plt.errorbar(final_r,final_pixels,xerr=r_radial_err,yerr=pixel_err,marker='x',capsize=2,linestyle='', color='red',label='Fitting pixels',zorder=2)
            plt.scatter(r_radials[peak_index],pixel_radials[peak_index],marker='x',color='black',zorder=3)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend()
            plt.ylabel("Normalised pixel intensities",fontsize=fontsize)
            plt.xlabel("Normalised pixel radii, r/R",fontsize=fontsize)
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
        plt.errorbar(r_radials,pixel_radials,xerr=r_radial_err,yerr=pixel_err,marker='x',capsize=2,linestyle='',label='All data',zorder=1)
        plt.title(theta_array[i])
        plt.errorbar(final_r,final_pixels,xerr=r_radial_err,yerr=pixel_err,marker='x',capsize=2,linestyle='', color='red',label='Fitting pixels',zorder=2)
        plt.scatter(r_radials[peak_index],pixel_radials[peak_index],marker='x',color='black',zorder=3)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel("(Difference in) Normalised pixel intensities",fontsize=fontsize)
        plt.xlabel("Normalised pixel radii, r/R",fontsize=fontsize)
        plt.show()
    
    if slice_plot == True:
        radials_bool = np.logical_and(theta > theta_array[i]-del_theta,
                                                 theta < theta_array[i]+del_theta)
        theta_val_2 = (theta_array[i] - pi)%(2*pi)
        radials_bool_2 = np.logical_and(theta > theta_val_2-del_theta,
                                                 theta < theta_val_2+del_theta)
        
        
        r_radials_slice = np.append(r[radials_bool],-r[radials_bool_2])
        sorted_indices = np.argsort(r_radials_slice) #sorting the pixels in order of radii
        pixel_radials_slice = np.append(pixels[radials_bool],pixels[radials_bool_2])[sorted_indices]
        r_radials_slice = r_radials_slice[sorted_indices]/R
        
        
        
        
        xlim = [r_radials_slice[0]-0.03,r_radials_slice[-1]+0.03]
        ylim = [0,np.amax(pixel_radials_slice)+0.05]
        r_values = np.linspace(0,xlim[-1],100)
        
        fitted_intensities = power_law(r_values,m_array[i], a_array[i], b_array[i])
        
        plt.figure()
        plt.plot(r_values,fitted_intensities,color='red',label="Power law fit",zorder=2)
        plt.errorbar(r_radials_slice,pixel_radials_slice,xerr=r_radial_err,yerr=pixel_err,color='black',marker='x',capsize=2,linestyle='',label='All data',zorder=1)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.ylabel(r"Normalised intensity, $I\;/\;I_{max}$",fontsize=fontsize)
        plt.xlabel("Normalised radius, r/R",fontsize=fontsize)
        plt.show()
        
        
valid_m = m_err_array.nonzero()
m_array = m_array[valid_m]
m_err_array = m_err_array[valid_m]
b_array = b_array[valid_m]
b_err_array = b_err_array[valid_m]
a_array = a_array[valid_m]
a_err_array = a_err_array[valid_m]
std_array = std_array[valid_m]
std_err_array = std_err_array[valid_m]
peak_intensity = peak_intensity[valid_m]
peak_intensity_err =peak_intensity_err[valid_m]
peak_pos= peak_pos[valid_m]
peak_pos_err=peak_pos_err[valid_m]
peak_diff_pos = peak_diff_pos[valid_m]
peak_diff_pos_err =peak_diff_pos_err[valid_m]
theta_array = theta_array[valid_m]

# print(m_array,m_err_array)
avg_m, avg_m_err = weighted_avg(m_array,m_err_array)
print("Average power law exponent: {} +/- {}".format(avg_m,avg_m_err))
m_std, m_std_err = weighted_std(m_array,m_err_array)
print("Standard deviation on power law exponent: {} +/- {}".format(m_std,m_std_err))

avg_b, avg_b_err = weighted_avg(b_array,b_err_array)
print("Average constant: {} +/- {}".format(avg_b,avg_b_err))
b_std, b_std_err = weighted_std(b_array,b_err_array)
print("Standard deviation on constant: {} +/- {}".format(b_std,b_std_err))

avg_I_peak, avg_I_peak_err = weighted_avg(peak_intensity,peak_intensity_err)
print("Average peak intensity: {} +/- {}".format(avg_I_peak,avg_I_peak_err))
I_std, I_std_err = weighted_std(peak_intensity,peak_intensity_err)
print("Standard deviation on peak intensity: {} +/- {}".format(I_std,I_std_err))

rel_darkness = avg_b/avg_I_peak*100
rel_darkness_err = rel_darkness*np.sqrt((avg_b_err/avg_b)**2+(avg_I_peak_err/avg_I_peak)**2)
print("Average darkness is {} +/- {}% of the peak".format(rel_darkness,rel_darkness_err))

avg_std, avg_std_err = weighted_avg(std_array,std_err_array)
print("Average residual standard deviation: {} +/- {}".format(avg_std,avg_std_err))

avg_R, avg_R_err = weighted_avg(peak_pos,peak_pos_err)
print("Average normalised ring radius: {} +/- {}".format(avg_R,avg_R_err))
R_std, R_std_err = weighted_std(peak_pos,peak_pos_err)
print("Standard deviation on peak radius: {} +/- {}".format(R_std,R_std_err))

avg_grad_peak, avg_grad_peak_err = weighted_avg(peak_diff_pos,peak_diff_pos_err)
print("Average normalised gradient peak radius: {} +/- {}".format(avg_grad_peak,avg_grad_peak_err))
# avg_trough_peak, avg_trough_peak_err = weighted_avg(trough_diff_pos,trough_diff_pos_err)
# print("Average normalised gradient trough radius: {} +/- {}".format(avg_trough_peak,avg_trough_peak_err))

#Write results
dataframe = pd.DataFrame(np.array([[R, R_err, x_offset, x_offset_err, y_offset, y_offset_err, avg_m, avg_m_err, m_std, m_std_err, avg_b, avg_b_err, b_std, b_std_err, avg_I_peak, avg_I_peak_err, avg_std, avg_std_err, avg_R, avg_R_err, avg_grad_peak, avg_grad_peak_err]]),
                         columns=['Radius (pixels)','Radius error (pixels)','X offset (pixels)','X offset error (pixels)','Y offset (pixels)','Y offset error (pixels)', 'Exponent', 
                                  'Exponent error', 'Exponent std.', 'Exponent std. error', 'Constant', 'Constant error', 'Constant std.', 'Constant std. error', 'Avg. intensity', 'Avg. intensity error', 'Fit std.', 'Fit std. error', 'Avg. radius', 'Avg. radius error', 'Avg. grad. peak','Avg. grad. peak error'],
                         dtype=float)
dataframe.to_csv(path[:-4]+'.csv')

line_colour = (0.6,0,0)
dotted_colour = (0.9,0,0)

fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
plt.errorbar(theta_array,m_array,yerr=m_err_array,marker='x',color='black',capsize=2,linestyle='',zorder=5)
xlim = ax.get_xlim()

avg_line_x = [xlim[0],xlim[-1]]
avg_line_y = [avg_m,avg_m]
avg_line_y_high = [avg_m+avg_m_err,avg_m+avg_m_err]
avg_line_y_low = [avg_m-avg_m_err,avg_m-avg_m_err]
plt.plot(avg_line_x,avg_line_y,linestyle='--', color=line_colour, alpha=0.8)

ax.plot(avg_line_x,avg_line_y_high, linestyle=':', color=dotted_colour)
ax.plot(avg_line_x,avg_line_y_low, linestyle=':', color=dotted_colour)
ax.fill_between(avg_line_x,avg_line_y_low,avg_line_y_high, color=line_colour,alpha=0.2)
plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
           labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
plt.xlim(xlim)
plt.ylabel("Fitted power law exponent",fontsize=fontsize)
plt.xlabel("Angle (radians)",fontsize=fontsize)
plt.show()

fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
plt.errorbar(theta_array,b_array,yerr=b_err_array,marker='x',color='black',capsize=2,linestyle='',zorder=5)
xlim = ax.get_xlim()

avg_line_x = [xlim[0],xlim[-1]]
avg_line_y = [avg_b,avg_b]
avg_line_y_high = [avg_b+avg_b_err,avg_b+avg_b_err]
avg_line_y_low = [avg_b-avg_b_err,avg_b-avg_b_err]
plt.plot(avg_line_x,avg_line_y,linestyle='--', color=line_colour, alpha=0.8)

ax.plot(avg_line_x,avg_line_y_high, linestyle=':', color=dotted_colour)
ax.plot(avg_line_x,avg_line_y_low, linestyle=':', color=dotted_colour)
ax.fill_between(avg_line_x,avg_line_y_low,avg_line_y_high, color=line_colour,alpha=0.2)
plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
           labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
plt.xlim(xlim)
plt.ylabel("Fitted power law constant",fontsize=fontsize)
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
plt.ylabel(r"Normalised peak intensity, $I\;/\;I_{max}$",fontsize=fontsize)
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
# plt.errorbar(theta_array,trough_diff_pos,yerr=trough_diff_pos_err,marker='x',color='orange',capsize=2,linestyle='',label='Gradient trough')

plt.ylabel("Normalised position, r/R",fontsize=fontsize)
plt.xlabel("Angle (radians)",fontsize=fontsize)
plt.show()

#Big combined plot:
xlim = [-0.1,2*pi+0.1]
    
fig = plt.figure()

ax = fig.add_axes((0,0,1,0.3))
plot_data(theta_array,m_array,m_err_array,avg_m,avg_m_err,xlim)
plt.ylabel("Fitted exponent",fontsize=fontsize,labelpad=34)
plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
           labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
ax.xaxis.tick_top()
plt.xlim(xlim)
yticks = ax.get_yticks()
plt.yticks(yticks[1:])


ax = fig.add_axes((0,-0.3,1,0.3))
plot_data(theta_array,b_array,b_err_array,avg_b,avg_b_err,xlim)
# plt.yaxis.set_label_position('right')
# ax.yaxis.tick_right()
plt.ylabel("Fitted constant",fontsize=fontsize,labelpad=2)
plt.xticks([])
plt.xlim(xlim)

ax = fig.add_axes((0,-0.6,1,0.3))
plot_data(theta_array,std_array,std_err_array,avg_std,avg_std_err,xlim)
plt.ylabel("Residual std",fontsize=fontsize,labelpad=28)
plt.xticks([])
plt.xlim(xlim)

ax = fig.add_axes((0,-0.9,1,0.3))
plot_data(theta_array,peak_intensity,peak_intensity_err,avg_I_peak,avg_I_peak_err,xlim)
# ax.yaxis.set_label_position('right')
# ax.yaxis.tick_right()
plt.ylabel(r"Norm. peak intensity",fontsize=fontsize,labelpad=6)
plt.xticks([])
plt.xlim(xlim)
yticks = ax.get_yticks()
plt.yticks(yticks[2:-1])

ax = fig.add_axes((0,-1.2,1,0.3))
plot_data(theta_array,peak_pos,peak_pos_err,avg_R,avg_R_err,xlim)
plt.ylabel("Norm. peak radius",fontsize=fontsize,labelpad=26)
plt.xlim(xlim)

plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
           labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
plt.xlabel("Angle (radians)",fontsize=fontsize)
# plt.savefig(path[:-4]+'_comb_param'+'.svg',dpi=300,bbox_inches='tight')
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

