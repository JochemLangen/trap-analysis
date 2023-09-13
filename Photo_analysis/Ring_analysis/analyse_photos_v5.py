# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:04:05 2023

@author: jochem langen

In a broad sense the programme does the following:
1. Import and prepare all the data (sorting, generating coordinate systems, determining the right angles).
2. It splits the ring profile into a set number of sections and bins all datapoints onto a grid defined by the radius.
3. Identify the features of the profile.
4. Fit a power law to the inner ring profile.
5. Generate all the output parameters (see below) by comparing the different sections and parameter post analysis.
6. Provide various options of plot generation to show the profile parameters.

This programme takes the images from the centre_photos and analyses each of them to generate the following parameters and their errors:
- The parameters it imported from "centre_photos": ring radius and centre x and y position from the image centre.
- The standard deviation of the ring radius along the ring.
- The average exponent of the power law that is fitted to the inside of the ring.
- The standard deviation in the power law exponent along the ring.
- The average normalised constant of the power law fit (normalised to the profile mean intensity).
- The standard deviation in the normalised power law constant.
- The average normalised peak intensity of the ring (normalised to the profile mean intensity).
- The standard deviation in the normalised peak intensity.
- The average relative power law constant, i.e. darkness (relative to the peak intensity).
- The average residual standard deviation around the fit.
- The standard deviation of the residual std. around the fit.
- The standard deviation of all residuals together.

How to use this programme:
1. Change rootdir to the same folder as was used in "center_photos". 
Note: use the original folder, not the "processed" folder.
3. Top index determines the image that you want to set as the main image for plotting. If the report_plot parameter (see below) is set to true, this is the only image that is analysed.
2. dist_float_factor is the number by which the total distance number integer should be divided to obtain the correct float value.
3. The dist_err (the error of the distance measurement) should be updated.
4. Exposure_err and exp_ms_err, the exposure time errors are not used in this file anymore but may be put here for personal future reference.
4. pixel_pos_err, is the pixel error. As the pixel size is the unit size of distance, this error is 0.5.
5. max_pixel_value should be adjusted to reflect the number of bits of the pixels and describes the saturation value.
6. The following parameters are used within the peak finding algorithm and may be adjusted if the algorithm does not manage to consistently find the centre. They have been determined to provide good results for all of the different set-ups used previously:
- averaging_int_size: The number of points over each radial line of pixels is averaged to get a smooth curve which is used here to find the trap profile shape and the fitting area.
- peak_size: When the the right fitting positions have been found from the averaged points, this parameter determines the +/- area around this point within which the individual cut-off pixel might fall. This is taken to be the same as the averaging interval.
- darkness_limit: The minimum value of any pixel to be considered as part of the peak in the peak finding algorithm, separate from the two parameters above.
- drop_limit: To be the drop-off from the first peak, the intensity difference between points must be below this value in the peak finding algorithm, separate from the first 2 parameters.
- peak_finding_avg_int_size: The averaging number of points used to find the peak in the peak finding algorithm, separate from the first 2 parameters. This corresponds to averaging_int_size in the "centre_photos" programme

7. The parameters: plots, subplots, prints, slice_plot, report_plot, bad_fit_prints, image_result & save_info determine what output information to generate.
8. distances sets whether a distance parameter is used (which would also be reflected in the file naming).
9. Guesses for the power law fit parameters: m_guess, a_guess, b_guess.
10. Plotting parameters: fontsize, tickwidth, ticklength.
11. no_theta_areas, the number of areas to divide the ring up in. All the pixels in one of these areas will be binned onto a grid and the results from these section are compared to find the ring symmetry through the standard deviation
12. delta_r is the interval over which the pixels are averaged (in pixels) when creating the sections
13. There is the option further down to set single_dir to false. If this is done here then the programme will also explore subfolders.
14. If save_info is set to true, the results will be in a file with the same name as the one created from "centre_photos" though with "_analysed" added to the end.

"""
import os
import sys
from funcs_photos import *
import pandas as pd
import matplotlib.mlab as mlab
from scipy.stats import norm
pi = np.pi

#The location of the folder with all the original images in the set
rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\Dist_range' #
top_index = 12 #The index of the main image that is used to extract the main parameters from / around when looking at a distance range (used for plotting in this file)


new_rootdir = rootdir + '\\Processed'

print(rootdir)

pixel_array = []
pixel_err_array = []
file_dist_array = []
file_err_dist_array = []
dist_float_factor = 1000 #The position of the floating point in the distance number in the file_name in mm
dist_err = 0.0005 #The error in mm

exposure_err = 0.005 #In us, not used in this file
exp_ms_err = 5e-6 #In ms, not used in this file
pixel_pos_err = 0.5 #Pixel width
name_array = []
max_pixel_value = 2**16-2**4

averaging_int_size = 4 #The number of points over which it is averaged to get a smooth curve
peak_size = averaging_int_size #The +/- area around the first value above this limit in which the max value is taken as the of the relevant parameter
darkness_limit = 0.05 #The minimum value to be considered as the peak
drop_limit = -0.05 #The value below which the difference between points must be to be the drop-off from the first peak
peak_finding_avg_int_size = 10 #The averaging number of points used to find the peak

#The parameters that determine what to do with the analysis: plot, print and/or save.
subplots = True
plots = True
prints = True
slice_plot = False
report_plot = False
bad_fit_prints = False
image_results = True
save_info = False
distances = True #Whether or not there are images taken over multiple distances
if distances == False:
    dist_err = 0

#Power law parameter guesses
m_guess = 20 #Exponent
a_guess = 3 #Coefficient
b_guess = 0.001 #Constant

fontsize=15
tickwidth = 2
ticklength = 4

no_theta_areas = 6 #The number of areas the image is averaged in
delta_r = 1 #The interval over which the pixels are averaged (in pixels) when creating the sections

#Read the results from the centring finding algorithm
file_name = new_rootdir+'\\'+rootdir[rootdir.rfind("\\")+1:]+'.csv'
results_path = file_name[:-4] + '_analysed.csv'
csv_file = pd.read_csv(file_name)

dist_array, exp_array, R_array, R_err_array, x_offset_array, x_offset_err_array, y_offset_array, y_offset_err_array = csv_file.values[:,1:].T
dist_len = len(dist_array)
exp_ms_array = exp_array / 1000

single_dir = True

dir_index = 0
for subdir, dirs, files in os.walk(new_rootdir):
    if single_dir == True:
        if dir_index > 0:
            break
    dir_index += 1
    directory_name = subdir[subdir[:-1].rfind("\\")+1:]
    for file in files:
        path = os.path.join(subdir,file)
        if path != file_name and path != results_path:
            
            dist_last_index = path.rfind("mm")
            dreduced_path = path[:dist_last_index]
            
            if path.find('err') == -1:
                pixel_array += [pd.read_csv(path).values[:,1:]]
                if distances == True:
                    file_dist_array += [float(dreduced_path[dreduced_path.rfind("_")+1:dist_last_index])/dist_float_factor]
                name_array += [path[path.rfind("\\")+1:]]
            else:
                pixel_err_array += [pd.read_csv(path).values[:,1:]]
                if distances == True:
                    file_err_dist_array += [float(dreduced_path[dreduced_path.rfind("_")+1:dist_last_index])/dist_float_factor]
                name_array += [path[path.rfind("\\")+1:]]
            
          
file_dist_array = np.asarray(file_dist_array)
file_err_dist_array = np.asarray(file_err_dist_array)
pixel_array = np.asarray(pixel_array,dtype=float)
pixel_err_array = np.asarray(pixel_err_array,dtype=float)
name_array = np.asarray(name_array,dtype=str)

if distances == True:  
    sorted_indices = np.argsort(file_dist_array)
    pixel_array = pixel_array[sorted_indices]
    name_array = name_array[sorted_indices]

    sorted_indices = np.argsort(file_err_dist_array)
    pixel_err_array = pixel_err_array[sorted_indices]

im_shape = np.shape(pixel_array[0])

#Creating the arrays used for normalisation
max_I = np.empty_like(dist_array)
max_I_err = np.empty_like(dist_array)
mean_intensity = np.average(pixel_array,axis=(1,2))
mean_intensity_err = np.std(mean_intensity)/np.sqrt(len(mean_intensity))
#As the intensity is supposed to be more or less constant throughout the experiment, the SE across the intensities can form an estimate of the error.
#This incorporates changes in the laser intensity but also the variation of the area imaged (in particular area that is not imaged anymore at larger radii) 

#Creating the arrays with the results:
avg_m_array = np.empty(dist_len)
avg_m_err_array = np.empty(dist_len)
m_std_array = np.empty(dist_len)
m_std_err_array = np.empty(dist_len)
avg_b_array = np.empty(dist_len)
avg_b_err_array = np.empty(dist_len)
b_std_array = np.empty(dist_len)
b_std_err_array = np.empty(dist_len)
avg_I_peak_array = np.empty(dist_len)
avg_I_peak_err_array = np.empty(dist_len)
I_std_array = np.empty(dist_len)
I_std_err_array = np.empty(dist_len)
rel_darkness_array = np.empty(dist_len)
rel_darkness_err_array = np.empty(dist_len)
avg_std_array = np.empty(dist_len)
avg_std_err_array = np.empty(dist_len)
std_std_array = np.empty(dist_len)
std_std_err_array = np.empty(dist_len)
total_std_array = np.empty(dist_len)
total_std_err_array = np.empty(dist_len)
avg_R_array = np.empty(dist_len)
avg_R_err_array = np.empty(dist_len)
R_std_array = np.empty(dist_len)
R_std_err_array = np.empty(dist_len)


for j in range(dist_len):
    
    if report_plot == True:
        j = top_index
    #Normalising the images
    ind = np.unravel_index(np.argmax(pixel_array[j], axis=None), pixel_array[j].shape)
    max_I[j] = pixel_array[j][ind]
    max_I_err[j] = pixel_err_array[j][ind]
    pixels = pixel_array[j] / max_I[j]
    if max_I_err[j] == np.inf: #The error of saturated values was set to inf previously to average out saturated values. This does not make sense for this purpose
        pixels_err = pixel_err_array[j] / max_I[j]
    else:
        pixels_err = np.sqrt(pixel_err_array[j]**2 + (max_I_err[j]*pixels)**2)/max_I[j]

    #Define Cartesian Coordinates: We take them as the centre of each pixel
    x = np.arange(im_shape[0])-x_offset_array[j]
    x_err = np.sqrt(pixel_pos_err**2 + x_offset_err_array[j]**2)
    y = np.arange(im_shape[1])-y_offset_array[j]
    y_err = np.sqrt(pixel_pos_err**2 + y_offset_err_array[j]**2)
    cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])

    #Convert to Polar coordinate system:
    r, r_err, theta, theta_err = CartPolar3(cart_coords,x_err,y_err)

    #Find the angles of the points along the ring to set the angle limits:
    ring_arg = np.logical_and(r <= R_array[j] + R_err_array[j] + r_err, r <= R_array[j] - R_err_array[j] - r_err)
    ring_theta = theta[ring_arg]
    ring_del_theta = theta_err[ring_arg]
    min_theta_arg = np.argmin(ring_theta)
    max_theta_arg = np.argmax(ring_theta)
    theta_limits = np.array([ring_theta[min_theta_arg]+ring_del_theta[min_theta_arg],ring_theta[max_theta_arg]-ring_del_theta[max_theta_arg]])
   

    theta_array = np.linspace(theta_limits[0],theta_limits[1],no_theta_areas+1)
    
    #Setting up the arrays for all the results
    m_array = np.empty(no_theta_areas)
    m_err_array = np.empty(no_theta_areas)
    a_array = np.empty(no_theta_areas)
    a_err_array = np.empty(no_theta_areas)
    b_array = np.empty(no_theta_areas)
    b_err_array = np.empty(no_theta_areas)
    std_array = np.empty(no_theta_areas)
    std_err_array = np.empty(no_theta_areas)
    peak_intensity = np.empty(no_theta_areas)
    peak_intensity_err = np.empty(no_theta_areas)
    peak_pos = np.empty(no_theta_areas)
    peak_pos_err = np.empty(no_theta_areas)
    peak_diff_pos = np.empty(no_theta_areas)
    peak_diff_pos_err = np.empty(no_theta_areas)
    residuals_array = np.array([])
    residuals_err_array = np.array([])
    
    for i in range(no_theta_areas):

        #Average the section and find the peak position
        r_radials, r_radials_err, pixel_radials, pixel_radials_err, peak_intensity[i],peak_intensity_err[i],peak_pos[i],peak_pos_err[i], sect_r, sect_r_err, sect_pixels, sect_pixels_err = avg_sections(pixels,pixels_err,r,r_err,theta,theta_err,theta_array[i],theta_array[i+1],R_array[j],R_err_array[j],delta_r,darkness_limit,drop_limit,peak_finding_avg_int_size)

        #Average the pixels to get a smoother curve
        pixel_avg, pixel_err_avg = interval_avg_weighted(pixel_radials, pixel_radials_err, averaging_int_size) #Take the averages over every -interval- points
        radial_avg, radial_err_avg = interval_avg_weighted(r_radials, r_radials_err, averaging_int_size)
        
        
        #Determine the point with the steepest wall
        avg_diff = pixel_avg[1:]-pixel_avg[:-1] #Find the difference between these points
        avg_peak_index = (np.argmax(avg_diff)+1)*averaging_int_size   
        
        if subplots == True:
            avg_diff_err = np.sqrt(pixel_err_avg[1:]**2 + pixel_err_avg[:-1]**2)
            
            plt.figure()
            plt.errorbar(radial_avg[:-1],avg_diff,yerr=avg_diff_err,xerr=radial_err_avg[:-1],capsize = 2)
            avg_index = int(avg_peak_index/averaging_int_size)
            plt.errorbar(radial_avg[avg_index-1],avg_diff[avg_index-1],yerr=avg_diff_err[avg_index-1],xerr=radial_err_avg[avg_index-1],marker='x',linestyle='',color='black',label='Peak index',capsize = 2)
            plt.legend()
            plt.ylabel("Difference in averaged normalised intensities",fontsize=fontsize)
            plt.xlabel("Averaged normalised radii, r/R",fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.show()

            plt.figure()
            plt.errorbar(radial_avg,pixel_avg,yerr=pixel_err_avg,xerr=radial_err_avg,capsize = 2)
            plt.errorbar(radial_avg[avg_index],pixel_avg[avg_index],yerr=pixel_err_avg[avg_index],xerr=radial_err_avg[avg_index],marker='x',linestyle='',color='black',label='Peak index',capsize = 2)
            plt.legend()
            plt.ylabel("Averaged normalised intensities",fontsize=fontsize)
            plt.xlabel("Averaged normalised radii, r/R",fontsize=fontsize)
            plt.ylim([0,np.amax(pixel_avg)])
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.show()
        
        #Get the index of the point furthest out where this difference exceeds the reference value
        if (len(r_radials)-1)-avg_peak_index >= peak_size:
            peak_pixels = pixel_radials[avg_peak_index-peak_size:avg_peak_index+peak_size+1]
            
            # peak_r = r_radials[avg_peak_index-peak_size:avg_peak_index+peak_size]
            pixel_diff = peak_pixels[1:]-peak_pixels[:-1]#/(peak_r[:-1]-peak_r[1:])
            peak_index = avg_peak_index-peak_size + np.argmax(pixel_diff)+1
            peak_diff_pos[i] = r_radials[peak_index]
            peak_diff_pos_err[i] = r_radials_err[peak_index]
            
            if subplots == True:
                r_peak = r_radials[avg_peak_index-peak_size+1:avg_peak_index+peak_size+1]
                r_peak_err = r_radials_err[avg_peak_index-peak_size+1:avg_peak_index+peak_size+1]
                peak_pixels_err = pixel_radials_err[avg_peak_index-peak_size:avg_peak_index+peak_size+1]
                pixel_diff_err = np.sqrt(peak_pixels_err[1:]**2 + peak_pixels_err[:-1]**2)
                plt.figure()
                plt.scatter(r_peak,peak_pixels[1:],label='Peak pixels',color='orange')
                plt.errorbar(r_radials,pixel_radials,yerr=pixel_radials_err, xerr=r_radials_err, capsize = 2)
                plt.errorbar(r_peak,pixel_diff,yerr=pixel_diff_err,xerr=r_peak_err,label='Peak pixel differences', capsize = 2)
                plt.errorbar(peak_diff_pos[i],pixel_radials[peak_index],yerr=pixel_radials_err[peak_index],xerr=peak_diff_pos_err[i],marker='x',color='black', capsize = 2,zorder=5)
                plt.xlim([r_peak[0]-0.03,r_peak[-1]+0.03])
                plt.legend()
                plt.ylabel("(Difference in) Normalised intensities",fontsize=fontsize)
                plt.xlabel("Averaged normalised radii, r/R",fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.show()
        else:
            peak_pixels = pixel_radials[avg_peak_index-peak_size:]
            pixel_diff = peak_pixels[1:]-peak_pixels[:-1]#/(peak_r[:-1]-peak_r[1:])
            peak_index = avg_peak_index-peak_size + np.argmax(pixel_diff)+1
            peak_diff_pos[i] = r_radials[peak_index]
            peak_diff_pos_err[i] = r_radials_err[peak_index]
            
            if subplots == True:
                r_peak = r_radials[avg_peak_index-peak_size+1:]
                r_peak_err = r_radials_err[avg_peak_index-peak_size+1:]
                peak_pixels_err = pixel_radials_err[avg_peak_index-peak_size:]
                pixel_diff_err = np.sqrt(peak_pixels_err[1:]**2 + peak_pixels_err[:-1]**2)
                plt.figure()
                plt.scatter(r_peak,peak_pixels[1:],label='Peak pixels',color='orange')
                plt.errorbar(r_radials,pixel_radials,yerr=pixel_radials_err, xerr=r_radials_err, capsize = 2)
                plt.errorbar(r_peak,pixel_diff,yerr=pixel_diff_err,xerr=r_peak_err,label='Peak pixel differences', capsize = 2)
                plt.errorbar(peak_diff_pos[i],pixel_radials[peak_index],yerr=pixel_radials_err[peak_index],xerr=peak_diff_pos_err[i],marker='x',color='black', capsize = 2,zorder=5)
                plt.xlim([r_peak[0]-0.03,r_peak[-1]+0.03])
                plt.legend()
                plt.ylabel("(Difference in) Normalised intensities",fontsize=fontsize)
                plt.xlabel("Averaged normalised radii, r/R",fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.show()
        
        final_pixels = pixel_radials[:peak_index+1]
        final_pixels_err = pixel_radials_err[:peak_index+1]
        final_r = r_radials[:peak_index+1]
        final_r_err = r_radials_err[:peak_index+1]

        try:
            popt, pcov = curve_fit(power_law,final_r,final_pixels,sigma=final_pixels_err,absolute_sigma=False,p0=[m_guess,a_guess,b_guess],bounds=(0,np.inf))
            #Absolute sigma has been set to false as this provided a larger and thus more realistic error
            m_array[i], a_array[i], b_array[i] = popt
            m_err_array[i], a_err_array[i], b_err_array[i] = np.sqrt(np.diagonal(pcov))
        except Exception as ex: 
            print("The fitting went wrong! You likely need to choose a smaller number of areas")
            print("The error is: ",ex)
            sys.exit()
        
        #Generate the array of non-averaged points to later find the homogeneity:
        sect_peak_index = np.argmin(abs(sect_r-peak_diff_pos[i]))
        sect_r = sect_r[:sect_peak_index+1]
        sect_r_err = sect_r_err[:sect_peak_index+1]
        sect_pixels = sect_pixels[:sect_peak_index+1]
        sect_pixels_err = sect_pixels_err[:sect_peak_index+1]
        
        
        fitted_intensities = power_law(sect_r,m_array[i], a_array[i], b_array[i])
        fitted_intensity_err = np.sqrt((power_law(sect_r+sect_r_err,m_array[i], a_array[i], b_array[i])-fitted_intensities)**2
                                       + (power_law(sect_r,m_array[i]+m_err_array[i], a_array[i], b_array[i])-fitted_intensities)**2
                                       + (power_law(sect_r,m_array[i], a_array[i] + a_err_array[i], b_array[i])-fitted_intensities)**2
                                       + (power_law(sect_r,m_array[i], a_array[i], b_array[i] + b_err_array[i])-fitted_intensities)**2)

        residuals = sect_pixels - fitted_intensities
        residuals_array = np.append(residuals_array, residuals)
        residual_err = np.sqrt(sect_pixels_err**2 + fitted_intensity_err**2)
        residuals_err_array = np.append(residuals_err_array, residual_err)
        
        std_array[i], std_err_array[i] = weighted_resid_std(residuals,residual_err)

        
        if bad_fit_prints == True:
            if peak_pos[i] > 1.02 or peak_pos[i] < 0.98:
                xlim = [max(r_radials[0]*0.95,0.1),r_radials[-1]+0.03]
                ylim = [pixel_radials[0]*0.05,pixel_radials[-1]+0.1]
                r_values = np.linspace(xlim[0],xlim[-1],100)
        
                fitted_intensities = power_law(r_values,m_array[i], a_array[i], b_array[i])
        
                    
                plt.figure()
                plt.plot(r_values,fitted_intensities,color='purple',label="Power law fit",zorder=0)
                plt.errorbar(r_radials,pixel_radials,xerr=r_radials_err,yerr=pixel_radials_err,marker='x',capsize=2,linestyle='',label='Averaged data',zorder=1)
                plt.title(theta_array[i])
                plt.errorbar(final_r,final_pixels,xerr=final_r_err,yerr=final_pixels_err,marker='x',capsize=2,linestyle='', color='red',label='Fitting data',zorder=2)
                plt.errorbar(r_radials[peak_index],pixel_radials[peak_index],xerr=r_radials_err[peak_index],yerr=pixel_radials_err[peak_index],marker='x',capsize=2,color='black',zorder=3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend()
                plt.ylabel("Normalised intensities",fontsize=fontsize)
                plt.xlabel("Normalised radii, r/R",fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.show()
            
            
        if prints == True:
            print("m = {} +/- {}".format(m_array[i],m_err_array[i]))
            print("a = {} +/- {}".format(a_array[i],a_err_array[i]))
            print("b = {} +/- {}".format(b_array[i],b_err_array[i]))
            print("Standard deviation around fitted power law = {} +/- {}".format(std_array[i], std_err_array[i]))
        
        if plots == True:
            
            xlim = [max(r_radials[0]*0.95,0.1),r_radials[-1]+0.01]
            ylim = [np.amin(pixel_radials[pixel_radials.nonzero()]),pixel_radials[-1]+0.1]
            r_values = np.linspace(xlim[0],xlim[-1],100)

            fitted_intensities = power_law(r_values,m_array[i], a_array[i], b_array[i])

            if subplots == True:
                plt.figure()
                plt.errorbar(sect_r,sect_pixels,xerr=sect_r_err,yerr=sect_pixels_err,marker='x',capsize=2,linestyle='',label='All data',zorder=0)
                plt.plot(r_values,fitted_intensities,color='purple',label="Power law fit",zorder=41)
                plt.errorbar(r_radials,pixel_radials,xerr=r_radials_err,yerr=pixel_radials_err,marker='x',capsize=2,linestyle='',label='Averaged data',zorder=2)
                plt.title(theta_array[i])
                plt.errorbar(final_r,final_pixels,xerr=final_r_err,yerr=final_pixels_err,marker='x',capsize=2,linestyle='', color='red',label='Fitting data',zorder=3)
                plt.errorbar(r_radials[peak_index],pixel_radials[peak_index],xerr=r_radials_err[peak_index],yerr=pixel_radials_err[peak_index],marker='x',capsize=2,color='black',zorder=4)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend()
                plt.ylabel("Normalised intensities",fontsize=fontsize)
                plt.xlabel("Normalised radii, r/R",fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.show()
                
                #Plot the histogram of the inside of the hollow beam
                #The Freedman-Diaconis rule is used to obtain the bin width:
                bin_width = 2*(1.349*std_array[i])/np.cbrt(len(residuals))
                bin_lim = np.amax(abs(residuals))
                bin_no = int(bin_lim / bin_width)
                bin_no = 100
                print("No. of bins used: {}".format(bin_no))
                
                plt.figure()
                plt.hist(residuals,bins=bin_no,weights=residual_err,range=[-bin_lim,bin_lim],density=True)
                plt.ylabel("Normalised probability density",fontsize=fontsize)
                plt.xlabel("Normalised intensity residuals",fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.show()
                
            #Plot the log plot of the data
            plt.figure()
            plt.plot(r_values,fitted_intensities,color='purple',label="Power law fit",zorder=0)
            plt.errorbar(r_radials,pixel_radials,xerr=r_radials_err,yerr=pixel_radials_err,marker='x',capsize=2,linestyle='',label='Averaged data',zorder=1)
            plt.title(theta_array[i])
            plt.errorbar(final_r,final_pixels,xerr=final_r_err,yerr=final_pixels_err,marker='x',capsize=2,linestyle='', color='red',label='Fitting data',zorder=2)
            plt.errorbar(r_radials[peak_index],pixel_radials[peak_index],xerr=r_radials_err[peak_index],yerr=pixel_radials_err[peak_index],marker='x',capsize=2,color='black',zorder=3)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend()
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel("Normalised intensities",fontsize=fontsize)
            plt.xlabel("Normalised radii, r/R",fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.show()
        
        if slice_plot == True:
            #This produces a slice of a single line of pixels halfway down the averaged section as well as a line exactly opposit
            theta_val = (theta_array[i]+theta_array[i+1])/2
            radials_bool = np.logical_and(theta + theta_err >= theta_val, theta - theta_err < theta_val)
            
            theta_val_2 = (theta_val - pi)%(2*pi)
            radials_bool_2 = np.logical_and(theta + theta_err >= theta_val_2, theta - theta_err < theta_val_2)
            
            
            r_radials_slice = np.append(r[radials_bool],-r[radials_bool_2])
            sorted_indices = np.argsort(r_radials_slice) #sorting the pixels in order of radii
            pixel_radials_slice = np.append(pixels[radials_bool],pixels[radials_bool_2])[sorted_indices]
            pixel_radials_err_slice = np.append(pixels_err[radials_bool],pixels_err[radials_bool_2])[sorted_indices]
            r_radials_slice = r_radials_slice[sorted_indices]/R_array[j]
            r_radials_err_slice = np.sqrt(np.append(r_err[radials_bool],-r_err[radials_bool_2])[sorted_indices]**2 + (R_err_array[j]*r_radials_slice)**2)/R_array[j]
            
            
            
            xlim = [-1.5,1.5]
            ylim = [0,np.amax(pixel_radials_slice)+0.05]
            r_values = np.linspace(0,xlim[-1],100)
            
            fitted_intensities = power_law(r_values,m_array[i], a_array[i], b_array[i])
            
            plt.figure()
            plt.plot(r_values,fitted_intensities,color='red',label="Power law fit",zorder=2)
            plt.errorbar(r_radials_slice,pixel_radials_slice,xerr=r_radials_err_slice,yerr=pixel_radials_err_slice,color='black',marker='x',capsize=2,linestyle='',label='Averaged data',zorder=1)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.ylabel(r"Normalised intensity, $I\;/\;I_{max}$",fontsize=fontsize)
            plt.xlabel("Normalised radius, r/R",fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.show()
            
            
            #This produces a slice of the averaged section
                
            #The part of the slice used in the fitting above
            slice_r, slice_r_err, slice_pixels, slice_pixels_err = avg_area(pixels,pixels_err,r,r_err,theta,theta_err,theta_array[i],theta_array[i+1],delta_r)[:4]
            #The opposite slice:
            theta_low = (theta_array[i] - pi)%(2*pi)
            theta_high = (theta_array[i+1] - pi)%(2*pi)
            opp_slice_r, opp_slice_r_err, opp_slice_pixels, opp_slice_pixels_err = avg_area(pixels,pixels_err,r,r_err,theta,theta_err,theta_low,theta_high,delta_r)[:4]
            
            #Combining the slices:
            slice_r = np.append(slice_r,-opp_slice_r)/R_array[j]
            slice_r_err = np.sqrt(np.append(slice_r_err, opp_slice_r_err)**2 + (R_err_array[j]*slice_r)**2)/R_array[j]
            slice_pixels = np.append(slice_pixels, opp_slice_pixels)
            slice_pixels_err = np.append(slice_pixels_err, opp_slice_pixels_err)
            
            
            plt.figure()
            plt.plot(r_values,fitted_intensities,color='red',label="Power law fit",zorder=2)
            plt.errorbar(slice_r,slice_pixels,xerr=slice_r_err,yerr=slice_pixels_err,color='black',marker='x',capsize=2,linestyle='',label='Averaged data',zorder=1)
            plt.xlim(xlim)
            plt.ylim(ylim)

            plt.ylabel(r"Normalised intensity, $I\;/\;I_{max}$",fontsize=fontsize)
            plt.xlabel("Normalised radius, r/R",fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.show()
        
        if report_plot == True:
            
            pixel_size = np.average([6.14/1280,4.9/1024,4.8e-3],weights=[0.005/1280,0.05/1024,0.05e-3]) #In mm
            xticks = np.array([0,1.2,2.4,3.6,4.8,6])/pixel_size
            yticks = np.array([0,0.9,1.8,2.7,3.6,4.5])/pixel_size

            fig = plt.figure()
            ax1 = fig.add_axes((0,0,1,1))
            pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',origin='lower',vmin=0,vmax=1)
            cbar = plt.colorbar(mappable=pixel_im)
            cbar.ax.tick_params(labelsize=fontsize, width= tickwidth, length= ticklength)
            plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
            plt.xlabel("x (mm)",fontsize=fontsize)
            plt.ylabel("y (mm)",fontsize=fontsize)
            plt.gca().set_aspect('equal')
            plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
            plt.tick_params(axis='both', which='major', width= tickwidth, length= ticklength)
            plt.text(0.05, 0.9, '(b)',transform=ax1.transAxes,fontsize=fontsize,color='white')
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            
            #This produces a slice of a single line of pixels halfway down the averaged section as well as a line exactly opposit
            theta_val = (theta_array[i]+theta_array[i+1])/2
            radials_bool = np.logical_and(theta + theta_err >= theta_val, theta - theta_err < theta_val)
            
            theta_val_2 = (theta_val - pi)%(2*pi)
            radials_bool_2 = np.logical_and(theta + theta_err >= theta_val_2, theta - theta_err < theta_val_2)
            
            
            r_radials_slice = np.append(r[radials_bool],-r[radials_bool_2])
            sorted_indices = np.argsort(r_radials_slice) #sorting the pixels in order of radii
            pixel_radials_slice = np.append(pixels[radials_bool],pixels[radials_bool_2])[sorted_indices]
            pixel_radials_err_slice = np.append(pixels_err[radials_bool],pixels_err[radials_bool_2])[sorted_indices]
            r_radials_slice = r_radials_slice[sorted_indices]/R_array[j]
            r_radials_err_slice = np.sqrt(np.append(r_err[radials_bool],-r_err[radials_bool_2])[sorted_indices]**2 + (R_err_array[j]*r_radials_slice)**2)/R_array[j]
            
            
            xlim = [-1.5,1.5]
            inner_sect_pix = final_pixels[abs(final_r) < 1]
            
            ylim = [np.amin(inner_sect_pix[inner_sect_pix.nonzero()])*0.3,np.amax(pixel_radials_slice)+0.3]
            r_values = np.linspace(0,xlim[-1],100)
            
            fitted_intensities = power_law(r_values,m_array[i], a_array[i], b_array[i])
            
            offset_axes = 0.23
            ax2 = fig.add_axes((1+offset_axes,0,1,1))
            plt.plot(r_values,fitted_intensities,color='red',label="Power law fit",linewidth=3,zorder=2)
            non_zero_ind = pixel_radials_slice.nonzero()
            plt.errorbar(r_radials_slice[non_zero_ind],pixel_radials_slice[non_zero_ind],xerr=r_radials_err_slice[non_zero_ind],yerr=pixel_radials_err_slice[non_zero_ind],color='black',marker='x',capsize=2,linestyle='',label='Averaged data',zorder=1)
            
            plt.xlim(xlim)
            plt.yscale('log')
            ylim = [10**-4,1]
            plt.ylim(ylim)
            plt.ylabel(r"Normalised intensity, $I\;/\;I_{max}$",fontsize=fontsize)
            plt.xlabel("Normalised radius, r/R",fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.text(0.03, 0.9, '(c)',transform=ax2.transAxes,fontsize=fontsize)
            plt.tick_params(axis='both', which='major', width= tickwidth, length= ticklength)
            
            #Plot the histogram of the inside of the hollow beam
            #The Freedman-Diaconis rule is used to obtain the bin width:
            bin_width = 2*(1.349*std_array[i])/np.cbrt(len(residuals))
            bin_lim = np.amax(abs(residuals))
            bin_no = int(bin_lim / bin_width)
            print(bin_no)
            print("No. of bins used: {}".format(bin_no))
            

            ax = fig.add_axes((1.15+offset_axes,1,0.7,0.35))
            ax.xaxis.set_label_position('top') 
            ax.xaxis.tick_top()
            bin_values, bin_edges, bin_patches = plt.hist(residuals,bins=bin_no,weights=1/residual_err,range=[-bin_lim,bin_lim],density=True)

            gauss_y = norm.pdf(bin_edges, 0, std_array[i])
            plt.plot(bin_edges, gauss_y, 'r--', linewidth=2)
            plt.ylabel("P",fontsize=fontsize)
            plt.xlabel(r"$R_{I\;/\;I_{max}}$",labelpad=10,fontsize=fontsize)
            ylim = ax.get_ylim()
            min_ref = 1*10**(-3)
            ymin = np.argmin(bin_values[bin_values > min_ref])
            ymin = np.argmax(abs(bin_edges[1:][bin_values > min_ref]))
            ylim = [bin_values[bin_values > min_ref][ymin],ylim[-1]+80]
            xsize = abs(bin_edges[1:][bin_values > min_ref][ymin])
            xlim = [-xsize,xsize]
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.yscale('log')
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)

        
            plt.xticks(np.round(np.linspace(-xsize*2/3,xsize*2/3,3),2),fontsize=fontsize)
            plt.minorticks_on()
            plt.tick_params(axis='both', which='major', width= tickwidth, length= ticklength)
            plt.tick_params(axis='x', which='minor', width= tickwidth*2/3, length= ticklength/2)
            
            plt.text(0.03, 0.75, '(d)',transform=ax.transAxes,fontsize=fontsize)
            
            report_path = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Report_plots'
            results_file_index = rootdir.find('New_col\\')+8
            results_file_name = report_path + '\\' + rootdir[results_file_index:].replace("\\", "-") + '_'+ str(i) + '.svg'
            if not os.path.exists(report_path):
                os.makedirs(report_path)
            plt.savefig(results_file_name,dpi=300,bbox_inches='tight')
            
            plt.show()
            
    if report_plot == True:
        break

    act_peak_intensity = peak_intensity * max_I[j] / mean_intensity[j] #Normalise it by the average pixel intensity in the image
    act_b_array = b_array * max_I[j] / mean_intensity[j] #Normalise it by the average pixel intensity in the image
    act_std_array = std_array * max_I[j] / mean_intensity[j]
    act_residuals_array = residuals_array * max_I[j] / mean_intensity[j]
    if max_I_err[j] == np.inf: #The error of saturated values was set to inf previously to average out saturated values. This does not make sense for this purpose
        act_peak_intensity_err = np.sqrt((peak_intensity_err * max_I[j])**2 + (act_peak_intensity * mean_intensity_err)**2)/mean_intensity[j]
        act_b_err_array = np.sqrt((b_err_array * max_I[j])**2 + (act_b_array*mean_intensity_err)**2)/mean_intensity[j]
        act_std_err_array = np.sqrt((std_err_array * max_I[j])**2 + (act_std_array*mean_intensity_err)**2)/mean_intensity[j]
        act_residuals_err_array = np.sqrt((residuals_err_array * max_I[j])**2 + (act_residuals_array*mean_intensity_err)**2)/mean_intensity[j]
    else:
        act_peak_intensity_err = act_peak_intensity * np.sqrt((peak_intensity_err/peak_intensity)**2 + (max_I_err[j]/max_I[j])**2 + (mean_intensity_err/mean_intensity[j])**2)
        act_b_err_array = act_b_array * np.sqrt((b_err_array/b_array)**2 + (max_I_err[j]/max_I[j])**2 + (mean_intensity_err/mean_intensity[j])**2)
        act_std_err_array = act_std_array * np.sqrt((std_err_array/std_array)**2 + (max_I_err[j]/max_I[j])**2 + (mean_intensity_err/mean_intensity[j])**2)
        act_residuals_err_array = abs(act_residuals_array) * np.sqrt((residuals_err_array/residuals_array)**2 + (max_I_err[j]/max_I[j])**2 + (mean_intensity_err/mean_intensity[j])**2)
    
    avg_m_array[j], avg_m_err_array[j] = weighted_avg(m_array,m_err_array)
    m_std_array[j], m_std_err_array[j] = SE_to_std(m_array, avg_m_err_array[j])
    avg_b_array[j], avg_b_err_array[j] = weighted_avg(act_b_array,act_b_err_array)
    b_std_array[j], b_std_err_array[j] = SE_to_std(act_b_array, avg_b_err_array[j])
    avg_I_peak_array[j], avg_I_peak_err_array[j] = weighted_avg(act_peak_intensity,act_peak_intensity_err)
    I_std_array[j], I_std_err_array[j] = SE_to_std(act_peak_intensity, avg_I_peak_err_array[j])
    rel_darkness_array[j] = avg_b_array[j]/avg_I_peak_array[j]*100
    rel_darkness_err_array[j] = np.sqrt(avg_b_err_array[j]**2+(avg_I_peak_err_array[j]*rel_darkness_array[j])**2)/avg_I_peak_array[j]
    avg_std_array[j], avg_std_err_array[j] = weighted_avg(act_std_array,act_std_err_array)
    std_std_array[j], std_std_err_array[j] = SE_to_std(act_std_array, avg_std_err_array[j])
    total_std_array[j], total_std_err_array[j] = weighted_resid_std(act_residuals_array,act_residuals_err_array)
    avg_R, avg_R_err = weighted_avg(peak_pos,peak_pos_err)
    R_std_array[j], R_std_err_array[j] = SE_to_std(peak_pos, avg_R_err)
    
    if image_results == True:
        print("Image with radius: {}".format(R_array[j]))
        print("Average power law exponent: {} +/- {}".format(avg_m_array[j],avg_m_err_array[j]))
        
        print("Standard deviation on power law exponent: {} +/- {}".format(m_std_array[j],m_std_err_array[j]))
        
        print("Average constant: {} +/- {}".format(avg_b_array[j],avg_b_err_array[j]))
        
        print("Standard deviation on constant: {} +/- {}".format(b_std_array[j],b_std_err_array[j]))
        
        print("Average peak intensity: {} +/- {}".format(avg_I_peak_array[j],avg_I_peak_err_array[j]))
        
        print("Standard deviation on peak intensity: {} +/- {}".format(I_std_array[j],I_std_err_array[j]))
    
        print("Average darkness is {} +/- {}% of the peak".format(rel_darkness_array[j],rel_darkness_err_array[j]))
        
        print("Average residual standard deviation: {} +/- {}".format(avg_std_array[j],avg_std_err_array[j]))
        
        print("Total residual standard deviation: {} +/- {}".format(total_std_array[j],total_std_err_array[j]))
        
        
        print("Average normalised ring radius: {} +/- {}".format(avg_R,avg_R_err))
        
        print("Standard deviation on peak radius: {} +/- {}".format(R_std_array[j],R_std_err_array[j]))
    
        avg_grad_peak, avg_grad_peak_err = weighted_avg(peak_diff_pos,peak_diff_pos_err)
        print("Average normalised gradient peak radius: {} +/- {}".format(avg_grad_peak,avg_grad_peak_err))
        
        mid_theta_array = (theta_array[:-1]+theta_array[1:])/2
        
        line_colour = (0.6,0,0)
        dotted_colour = (0.9,0,0)
        
        plt.figure()
        plt.errorbar(mid_theta_array,peak_pos,yerr=peak_pos_err,marker='x',color='black',capsize=2,linestyle='',label='Ring peak')
        plt.errorbar(mid_theta_array,peak_diff_pos,yerr=peak_diff_pos_err,marker='x',color='blue',capsize=2,linestyle='',label='Gradient peak')
        plt.ylabel("Normalised position, r/R",fontsize=fontsize)
        plt.xlabel("Angle (radians)",fontsize=fontsize)
        plt.title("Image at distance: {} mm".format(dist_array[j]))
        plt.legend()
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.show()
        
        #Big combined plot:
        xlim = [-0.1,2*pi+0.1]
            
        fig = plt.figure()
    
        ax = fig.add_axes((0,0,1,0.3))
        plot_data(mid_theta_array,m_array,m_err_array,avg_m_array[j],avg_m_err_array[j],xlim,line_colour,dotted_colour)
        plt.ylabel("Fitted exponent",fontsize=fontsize,labelpad=34)
        plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
                   labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
        ax.xaxis.tick_top()
        plt.xlim(xlim)
        yticks = ax.get_yticks()
        plt.yticks(yticks[1:])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    
        ax = fig.add_axes((0,-0.3,1,0.3))
        plot_data(mid_theta_array,act_b_array,act_b_err_array,avg_b_array[j],avg_b_err_array[j],xlim,line_colour,dotted_colour)
        plt.ylabel("Fitted constant",fontsize=fontsize,labelpad=2)
        plt.xticks([])
        plt.xlim(xlim)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    
        ax = fig.add_axes((0,-0.6,1,0.3))
        plot_data(mid_theta_array,act_std_array,act_std_err_array,avg_std_array[j],avg_std_err_array[j],xlim,line_colour,dotted_colour)
        plt.ylabel("Residual std",fontsize=fontsize,labelpad=28)
        plt.xticks([])
        plt.xlim(xlim)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    
        ax = fig.add_axes((0,-0.9,1,0.3))
        plot_data(mid_theta_array,act_peak_intensity,act_peak_intensity_err,avg_I_peak_array[j],avg_I_peak_err_array[j],xlim,line_colour,dotted_colour)
        plt.ylabel(r"Norm. peak intensity",fontsize=fontsize,labelpad=8)
        plt.xticks([])
        plt.xlim(xlim)
        yticks = ax.get_yticks()
        plt.yticks(yticks[1:-1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    
        ax = fig.add_axes((0,-1.2,1,0.3))
        plot_data(mid_theta_array,peak_pos,peak_pos_err,avg_R,avg_R_err,xlim,line_colour,dotted_colour)
        plt.ylabel("Norm. peak radius",fontsize=fontsize,labelpad=26)
        plt.xlim(xlim)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi],
                   labels=['0',r'$\dfrac{\pi}{2}$',r'$\pi$',r'$\dfrac{3\pi}{2}$',r'$2\pi$'])
        plt.xlabel("Angle (radians)",fontsize=fontsize)
        # plt.savefig(path[:-4]+'_comb_param'+'.svg',dpi=300,bbox_inches='tight')
        plt.show()
    if report_plot == True:   
        print("The report plots have been generated. The code will now terminate.")
        sys.exit()
        
#Write results
if save_info == True:
    d = {'Distance (mm)':dist_array,'Exposure time (us)':exp_array,'Radius (pixels)':R_array,'Radius error (pixels)':R_err_array,
         'Norm. radius std':R_std_array,'Norm. radius std error':R_std_err_array,
         'X offset (pixels)':x_offset_array,'X offset error (pixels)':x_offset_err_array,'Y offset (pixels)':y_offset_array,
         'Y offset error (pixels)':y_offset_err_array,'Exponent':avg_m_array,'Exponent error':avg_m_err_array,
         'Exponent std':m_std_array,'Exponent std error':m_std_err_array, 'Norm. const.':avg_b_array,
         'Norm. const. error':avg_b_err_array,'Norm. const. std':b_std_array,'Norm. const. std error':b_std_err_array,
         'Norm. peak intensity':avg_I_peak_array,'Norm. peak intensity error':avg_I_peak_err_array,'Norm. peak intensity std':I_std_array,
         'Norm. peak intensity std error':I_std_err_array,'Rel. darkness':rel_darkness_array,'Rel. darkness err':rel_darkness_err_array,
         'Avg. resid. std':avg_std_array,'Avg. resid. std error':avg_std_err_array,
         'Resid. std std':std_std_array,'Resid. std std error':std_std_err_array,'Total resid. std':total_std_array,
         'Total resid. std error':total_std_err_array}
    dataframe = pd.DataFrame(d)
    dataframe.to_csv(results_path)

#Big combined plot:
xlim = [dist_array[0]-0.5,dist_array[-1]+0.5]
    
fig = plt.figure()

ax = fig.add_axes((0,0,1,0.3))
plt.errorbar(dist_array,avg_m_array,yerr=avg_m_err_array, xerr=dist_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel("Fitted exponent",fontsize=fontsize,labelpad=34)
ax.xaxis.tick_top()
plt.xlim(xlim)
yticks = ax.get_yticks()
plt.yticks(yticks[1:])
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax = fig.add_axes((0,-0.3,1,0.3))
plt.errorbar(dist_array,avg_b_array,yerr=avg_b_err_array, xerr=dist_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel("Fitted constant",fontsize=fontsize,labelpad=2)
plt.xticks([])
plt.xlim(xlim)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax = fig.add_axes((0,-0.6,1,0.3))
plt.errorbar(dist_array,avg_I_peak_array,yerr=avg_I_peak_err_array, xerr=dist_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.ylabel("Norm. peak intensity",fontsize=fontsize,labelpad=2)
plt.xticks([])
plt.xlim(xlim)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax = fig.add_axes((0,-0.9,1,0.3))
plt.errorbar(dist_array,avg_std_array,yerr=avg_std_err_array, xerr=dist_err,marker='x',color='black',capsize=2,linestyle='',zorder=5)
plt.errorbar(dist_array,total_std_array,yerr=total_std_err_array, xerr=dist_err,marker='x',color='blue',capsize=2,linestyle='',zorder=5)
plt.ylabel("Residual std.",fontsize=fontsize,labelpad=2)
plt.xlabel("Distance (mm)")
plt.xlim(xlim)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()
