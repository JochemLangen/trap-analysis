# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:38:52 2023

@author: jochem langen

-- "centre_photos" programme --
In a very broad sense, this programme does the following:
1. Extract & sort all images and parameters from the specified folder
3. Subtract the background using the appropriate and scaled background file
4. Find the position of the centre of the trap as well as its radius.
5. Combine the different images of the same profile (photos with varying exposure times to obtain a more detailed view of different intensity regimes).
The most recent code can be found in centre_photos_v3.py which uses functions from funcs_photos.py

How to use this programme:
1. Change rootdir to the appropriate folder that contains all the images of the profile that should be compared together
i.e. images of the profile with different exposure times, their background light measurements for the profile at each external parameter (e.g. distance along the optical axis)
The image files should have the following format:
"xx_xxus_xxmm.pgm"
Where the first "xx" can be given any name that does not contain "us" or "mm". The second "xx" contains the total exposure value float number, without the decimal point.
The third "xx" contains the total distance value float number, without the decimal point. 
NOTE: if no distances are used, the last part can be left off the file name.

The background files should be of the form:
"xx_xxus_xxmm_bg.pgm"

If an additional set of images is taken with an adjusted camera position to average out the imaging inaccuracy, these should be placed in a folder called "Adj_cam" inside the original folder. 
These are analysed separately in this file, but some of the analysis may differ.

2. Inside the file, the exp_float_factor needs to be set as the number by which the total exposure number integer should be divided to obtain the correct float value.
3. exposure_err should be updated to be the correct value.
4. saturation_value and pixel_err should be adjusted to reflect the number of bits of the pixels and the reported error from the camera.
5. corner_x and corner_y are the offsets from the centre of the image that are used for the initial "guess" of the centre. As long as this guess is within the circle, the algorithm should find the right position.
6. ring_type determines the polar orientation of the coordinate system and allows the possibility to flip the ring inside out. This should be kept at "inner" as the "outer" functionality has become deprecated.
7. The parameters: plots, subplots, save_fig & save_info determine what output information to generate.
8. distances sets whether a distance parameter is used (which would also be reflected in the file naming).
9. dist_float_factor is the number by which the total distance number integer should be divided to obtain the correct float value.
10. background_gen is set to False when background images are used and true when the "generate_bg" file has been used to generate background estimates.
11. pixel_size should hold the pixel size of the camera used. The currently filled in numbers form an average of the specified pixel size for each dimension.
12. The xticks and yticks should hold the values of the tick labels you want to use multiplied with the pixel size.
13. The following parameters are used within the centre finding algorithm and may be adjusted if the algorithm does not manage to consistently find the centre. They have been determined to provide good results for all of the different set-ups used previously:
- no_theta_points: The number of angles in each fitting range. This determines the number of points used to determine the centre of the circle (and circle radius).
- averaging_int_size: The number of points over each radial line of pixels is averaged to get a smooth curve
- darkness_limit: The minimum value of any pixel to be considered as part of the peak
- drop_limit: To be the drop-off from the first peak, the intensity difference between points must be below this value
- peak_size: When the peak has been found from the averaged points, this parameter determines the +/- area around this point within which the individual peak pixel might fall (determined by the max of this range). This is taken to be the same as the averaging interval.
- R_jump: The jump in corner coordinates if the circle centre guess does not fall in the centre
14. safety_frac sets the safety fraction on the used radius in combining the images. The saturated pixel closest to the centre sets the limit to what should be combined. This adds an additional safety radius reduction to avoid artifacts caused by bleeding.
15. In the unlikely scenario the fitting is not working well even after adjustment of the above parameters, further down in the file R_guess can be adjusted as the guess of the circle radius.
16. If save_info is set to true, an additional folder called "Processed" will be made inside the data folder which contains the results. If save_fig is set to true, the plots will be generated within the main data folder.
The programme generates the following output:
-The ring radius and its error.
-The x position of the circle centre and its error.
-The y position of the circle centre and its error.
-For ring, the final image with the background subtracted and with the relevant sections averaged between exposures.
"""

import os
import sys
from funcs_photos import *
import pandas as pd

#The location of the folder with all the original images to be analysed in this set
rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\Dist_range' #

print(rootdir)

pixel_array = []
dist_array = []
exp_array = []
exp_float_factor = 100 #The position of the floating point in the exposure number in the file_name in mm
exposure_err = 0.005 #In us
name_array = []

bg_pix_array = []
bg_pix_err_array = []
bg_exp_array = []
bg_dist_array = []
bg_dist_err_array = []
bg_exp_err_array = []
bg_exp_dist_array = []
single_dir = True

saturation_value = 2**16-2**4
pixel_err = 2**6

#For two background files that would be equally similar to the data file, which is used is changed in the adj_cam version to average out the effect
if rootdir.find('Adj_cam') == -1:
    bg_choosing_index = 0
else:
    bg_choosing_index = 1


#These parameters are for a guess of the offset from the image centre (prev. used 400 & 0 for x,y offsets 0,0 for corner coords)
corner_x = 0
corner_y = 0


ring_type = "inner"

#The parameters that determine what to do with the analysis: plot, print and/or save.
plots = True
subplots = False
save_fig = False
save_info = False
distances = True #Whether or not distances are used
if distances == True:
    dist_float_factor = 1000 
background_gen = False #Whether or not the background gen file has been used
pixel_size = np.average([6.14/1280,4.9/1024]) #In mm
xticks = np.array([0,1.2,2.4,3.6,4.8,6])/pixel_size
yticks = np.array([0,0.9,1.8,2.7,3.6,4.5])/pixel_size
fontsize = 13

#Setting the parameters for the outer ring determination
no_theta_points = 500 #The number of angles in each fitting range
averaging_int_size = 10 #The number of points over which it is averaged to get a smooth curve
darkness_limit = 0.05 #The minimum value to be considered as the peak
drop_limit = -0.05 #The value below which the difference between points must be to be the drop-off from the first peak
peak_size = averaging_int_size #The +/- area around the first value above this limit in which the max value is taken as the peak
R_jump = 40 #The jump in corner coordinates if the circle does not fall in the centre
safety_frac = 0.05 #The safety fraction on the used radius in combining the images



dir_index = 0
for subdir, dirs, files in os.walk(rootdir):
    if single_dir == True:
        if dir_index > 0:
            break
    dir_index += 1
    directory_name = subdir[subdir[:-1].rfind("\\")+1:]
    for file in files:
        path = os.path.join(subdir,file)
        if path[-3:] == 'pgm':
            im = Image.open(path)
            
            dist_last_index = path.rfind("mm")
            dreduced_path = path[:dist_last_index]
            exp_last_index = path.rfind("us")
            ereduced_path = path[:exp_last_index]
            if path.find('no') == -1: #Images with no in them will not be used
                if path.find('bg') == -1:
                    if distances == True:
                        dist_array += [float(dreduced_path[dreduced_path.rfind("_")+1:dist_last_index])/dist_float_factor]
                    exp_array += [float(ereduced_path[ereduced_path.rfind("_")+1:exp_last_index])/exp_float_factor]
                    name_array += [path[path.rfind("\\")+1:]]
                    
                    pixel_array += [np.array(im).T]
                elif background_gen == False:
                    if distances == True:
                        bg_dist_array += [float(dreduced_path[dreduced_path.rfind("_")+1:dist_last_index])/dist_float_factor]
                    bg_exp_array += [float(ereduced_path[ereduced_path.rfind("_")+1:exp_last_index])/exp_float_factor]
                    
                    bg_pix_array += [np.array(im).T]
        elif path[-3:] == 'csv' and background_gen == True:
            dist_last_index = path.rfind("mm")
            dreduced_path = path[:dist_last_index]
            exp_last_index = path.rfind("us")
            ereduced_path = path[:exp_last_index]
            if path.find('_bg_err') != -1:
                if distances == True:
                    bg_dist_err_array += [float(dreduced_path[dreduced_path.rfind("_")+1:dist_last_index])/dist_float_factor]
                bg_exp_err_array += [float(ereduced_path[ereduced_path.rfind("_")+1:exp_last_index])/exp_float_factor]
                
                bg_pix_err_array += [pd.read_csv(path).values[:,1:]]
            elif path.find('_bg') != -1: 
                if distances == True:
                    bg_dist_array += [float(dreduced_path[dreduced_path.rfind("_")+1:dist_last_index])/dist_float_factor]
                bg_exp_array += [float(ereduced_path[ereduced_path.rfind("_")+1:exp_last_index])/exp_float_factor]
                
                bg_pix_array += [pd.read_csv(path).values[:,1:]]
            

dist_array = np.asarray(dist_array)
exp_array = np.asarray(exp_array)
pixel_array = np.asarray(pixel_array,dtype=float)
name_array = np.asarray(name_array,dtype=str)

if distances == True:
    dist_len = len(dist_array)
    sorted_indices = np.argsort(dist_array)
    dist_array = dist_array[sorted_indices]
    exp_array = exp_array[sorted_indices]
    pixel_array = pixel_array[sorted_indices]
    pixel_err_array = np.full_like(pixel_array,pixel_err)
    name_array = name_array[sorted_indices]
else:
    dist_len = len(exp_array)

#Finding the index of the lowest exposure image within a set and sort arrays:
uniq_dist, dist_counts = np.unique(dist_array, return_counts=True)
cum_count = 0
if distances == True:
    no_images = len(uniq_dist)
    min_indices = np.empty(no_images,dtype=int)

    for i in range(no_images):
        current_range = cum_count+dist_counts[i]
        exp_range = exp_array[cum_count:current_range]
        sorted_indices = np.argsort(exp_range)
        
        dist_array[cum_count:current_range] = dist_array[cum_count:current_range][sorted_indices]
        exp_array[cum_count:current_range] = exp_array[cum_count:current_range][sorted_indices]
        pixel_array[cum_count:current_range] = pixel_array[cum_count:current_range][sorted_indices]
        name_array[cum_count:current_range] = name_array[cum_count:current_range][sorted_indices]
        
        
        min_indices[i] = cum_count
        cum_count += dist_counts[i]
    max_indices = np.append(min_indices[1:],dist_len)  
    
else:
    no_images = 1
    sorted_indices = np.argsort(exp_array)
    exp_array = exp_array[sorted_indices]
    pixel_array = pixel_array[sorted_indices]
    pixel_err_array = np.full_like(pixel_array,pixel_err)
    name_array = name_array[sorted_indices]
    
    min_indices = np.array([0])
    max_indices = np.array([dist_len])


#Define Cartesian Coordinates: We take them as the centre of each pixel
im_shape = np.shape(pixel_array[0])
corner_coords = [im_shape[0]/2+0.1+corner_x,im_shape[1]/2+0.1+corner_y] #Acting guess of centre to find the outer ring (you don't want it to be exactly on a pixel nor half-way (if the half pixel length is used for del_theta, rather than radius))
R_guess = 200
x_offset_guess = 0   #Offset from corner_coords, should not be adjusted
y_offset_guess = 0      #Offset from corner_coords, should not be adjusted


#Subtracting the background:
for i in range(dist_len):
    if distances == True:
        difference_arr = abs(dist_array[i]-bg_dist_array)
        bg_index = np.flatnonzero(difference_arr == difference_arr.min()) #This finds the appropriate background file
        if len(bg_index) != 1: #This checks whether there are not multiple bg files at the same distance, the bg with the closest exposure time is then chosen
            diff_exp_arr = abs(exp_array[i]-np.take(bg_exp_array,bg_index))
            diff_exp_index = np.flatnonzero(diff_exp_arr == diff_exp_arr.min())
            bg_index = np.take(bg_index,diff_exp_index)
            if len(bg_index) == 2:
                bg_index = bg_index[bg_choosing_index]
            elif len(bg_index) > 2:
                print("There are more than 2 options for the background after exp. and dist. being considered!")
                bg_index = bg_index[bg_choosing_index]
            else:
                bg_index = bg_index[0] #Turns the length-1 array into a scalar
        else:
            bg_index = bg_index[0]
    else:
        diff_exp_arr = abs(exp_array[i]-bg_exp_array)
        bg_index = np.flatnonzero(diff_exp_arr == diff_exp_arr.min())[0]
        
    if background_gen == True:
        if distances == True:
            difference_arr = abs(dist_array[i]-bg_dist_err_array)
            bg_err_index = np.flatnonzero(difference_arr == difference_arr.min()) #This finds the appropriate background file
            if len(bg_err_index) != 1: #This checks whether there are not multiple bg files at the same distance, the bg with the closest exposure time is then chosen
                diff_exp_arr = abs(exp_array[i]-np.take(bg_exp_err_array,bg_err_index))
                diff_exp_index = np.flatnonzero(diff_exp_arr == diff_exp_arr.min())
                bg_err_index = np.take(bg_err_index,diff_exp_index)
                if len(bg_err_index) == 2:
                    bg_err_index = bg_err_index[bg_choosing_index]
                elif len(bg_err_index) > 2:
                    print("There are more than 2 options for the background after exp. and dist. being considered!")
                    bg_err_index = bg_err_index[bg_choosing_index]
                else:
                    bg_err_index = bg_err_index[0] #Turns the length-1 array into a scalar
            else:
                bg_err_index = bg_err_index[0]
        else:
            diff_exp_arr = abs(exp_array[i]-bg_exp_err_array)
            bg_err_index = np.flatnonzero(diff_exp_arr == diff_exp_arr.min())[0]
        
    bg_pixels = bg_pix_array[bg_index] #finds the appropriate background pixels measured at the closest distance (rounded down at half-distance)
    
    if background_gen == True:
        scaled_bg_pix = bg_pixels
        scaled_bg_pix_err = bg_pix_err_array[bg_err_index]
        #The bg values are determined for each specific image 
    else:
        scaled_bg_pix = bg_pixels/bg_exp_array[bg_index]*exp_array[i] #This scales the background to the right exposure used
        scaled_bg_pix_err = np.sqrt(((bg_pixels+pixel_err)/bg_exp_array[bg_index]*exp_array[i] - scaled_bg_pix)**2 +
                                    (bg_pixels/(exposure_err + bg_exp_array[bg_index])*exp_array[i] - scaled_bg_pix)**2 +
                                    (bg_pixels/bg_exp_array[bg_index]*(exposure_err+exp_array[i]) - scaled_bg_pix)**2)
        
    
    plt.figure()
    image_x = int(corner_coords[0])
    image_y = int(corner_coords[1])
    image_x_values = np.linspace(0,1280,1280)
    image_ylim = 25000
    plt.errorbar(image_x_values,pixel_array[i][:,image_y],yerr=pixel_err_array[i][:,image_y])
    plt.errorbar(image_x_values,scaled_bg_pix[:,image_y],yerr=scaled_bg_pix_err[:,image_y])
    plt.xlim([image_x-300,image_x+300])
    plt.ylim([0,image_ylim])
    if distances == True:
        plt.title('Original image with exp: '+str(exp_array[i])+" at distance: "+str(dist_array[i]))
    else:
        plt.title('Original image with exp: '+str(exp_array[i]))
    
    plt.show()
    
    pixel_err_array[i][pixel_array[i] >= saturation_value] = np.inf #All saturated pixels get an infinite error, this means that they will be averaged out when combining images
    
    pixel_array[i] -= scaled_bg_pix #This subtracts the scaled background
    pixel_err_array[i] = np.sqrt(scaled_bg_pix_err**2 + pixel_err_array[i]**2) #Propagates final error using calc method
    
    excess_indices = pixel_array[i] < 0 #Determines all pixels for which the background ref value is higher than the measured background
    excess_pix = abs(pixel_array[i][excess_indices]) #Abs values of these excess background pixels
    pixel_err_array[i][excess_indices] = np.sqrt(excess_pix**2 + pixel_err_array[i][excess_indices]**2) #Is this the right method?
    
    pixel_array[i][excess_indices] = 0

    
#Set-up general image fitting param. arrays
R_array = np.empty(no_images)
R_err_array = np.empty(no_images)
x_offset_array = np.empty(no_images)
x_offset_err_array = np.empty(no_images)
y_offset_array = np.empty(no_images)
y_offset_err_array = np.empty(no_images)
if distances == True:
    final_dist_array = dist_array[min_indices]
else:
    final_dist_array = np.array([0])
final_exp_array = exp_array[min_indices]




#Create the folder with the final processed images
new_path = rootdir+'\\Processed'
if not os.path.exists(new_path):
    os.makedirs(new_path)

#Finding the image geometry:
for j,i in enumerate(min_indices):
    pixels = pixel_array[i]
    max_intensity = np.amax(pixels) # The reference parameters are w.r.t. pixels normalised to the max profile intensity
    pixels = pixels/max_intensity

    if plots == True:
        #Plot the original image:
        plt.figure()
        pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
        plt.colorbar(mappable=pixel_im,label='Normalised intensity')
        plt.scatter(corner_coords[0],corner_coords[1],marker='x',color='white')
        plt.gca().set_aspect('equal')
        plt.xlim([0,1279])
        plt.ylim([0,1023])
        plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
        plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
        plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
        plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
        plt.show()

    
    
    coord_jump_x = R_jump
    coord_jump_y = R_jump
    attempts = 0
    jump_index = 0 
    
    while attempts < 20:
        attempts += 1
        x = np.arange(im_shape[0])-corner_coords[0]
        y = np.arange(im_shape[1])-corner_coords[1]
        cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])

        #Convert to Polar coordinate system:
        r, theta, del_theta = CartPolar2(cart_coords)
        
        theta_array = np.linspace(0+del_theta[-1,int(corner_coords[1])],2*pi-del_theta[-1,int(corner_coords[1])],no_theta_points+1)[:-1]
        
        
        try:
            R_array[j], R_err_array[j], x_offset_array[j], x_offset_err_array[j], y_offset_array[j], y_offset_err_array[j] = polar_find_centre(pixels, theta_array, r, theta, del_theta, cart_coords,corner_coords,averaging_int_size,darkness_limit,drop_limit,peak_size,R_guess,x_offset_guess,y_offset_guess,ring_type,plot=plots,subplot=subplots,fontsize=fontsize,path=path,save_figs=save_fig)
            
            
            x -= x_offset_array[j]-corner_coords[0]
            y -= y_offset_array[j]-corner_coords[1]
            cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])

            #Convert to Polar coordinate system:
            r, theta, del_theta = CartPolar2(cart_coords)
            
            corner_coords[0] = x_offset_array[j]
            corner_coords[1] = y_offset_array[j]
            break
        except Exception as ex:
            #Plot the original image:
            plt.figure()
            pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
            plt.colorbar(mappable=pixel_im,label='Normalised intensity')
            plt.scatter(corner_coords[0],corner_coords[1],marker='x',color='white')
            plt.gca().set_aspect('equal')
            plt.xlim([0,1279])
            plt.ylim([0,1023])
            plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
            plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
            plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
            plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
            plt.title("Unsuccesful fitting attempt: {}".format(attempts))
            plt.show()
            
            
            #The method below cycles the starting position from which to find the circle in an octagon increasing in radius after each full loop
            jump_index = attempts % 8
            if attempts == 1:
                corner_coords[0] += coord_jump_x
            elif jump_index == 2:
                corner_coords[1] += coord_jump_y
            elif jump_index == 3:
                corner_coords[0] -= coord_jump_x
            elif jump_index == 4:
                corner_coords[0] -= coord_jump_x
            elif jump_index == 5:
                corner_coords[1] -= coord_jump_y
            elif jump_index == 6:
                corner_coords[1] -= coord_jump_y
            elif jump_index == 7:
                corner_coords[0] += coord_jump_x
            elif jump_index == 0:
                corner_coords[0] += coord_jump_x
            elif jump_index == 1:
                corner_coords[0] += R_jump
                corner_coords[1] += coord_jump_y
                coord_jump_x += R_jump
                coord_jump_y += R_jump
            
            if attempts == 20:
                print("The attempt limit of 20 was reached, please change some of the variables.")
                print("A likely change necessary is the value in R_jump or the corner_x and corner_y values.")
                print("For a small circle, the darkness limit may cause the usuable area for the corner coordinates to be small, change the above values (or the darkness_limit) appropriately.")
                print("The error is: ",ex)
                sys.exit()

    print("The fit took {} attempt(s)".format(attempts))
    #Print the results:
    print("The outer radius is {} +/- {} mm".format(R_array[j]*pixel_size,R_err_array[j]*pixel_size))
    print("The outer radius is {} +/- {} pixels".format(R_array[j],R_err_array[j]))
    print("The x offset is {} +/- {} pixels".format(x_offset_array[j],x_offset_err_array[j]))
    print("The y offset is {} +/- {} pixels".format(y_offset_array[j],y_offset_err_array[j]))
    
    pixels = pixel_array[i]
    pixels_err = pixel_err_array[i]
    exp_val = exp_array[i]

    if plots == True:
        plt.figure()
        image_x = int(corner_coords[0])
        image_y = int(corner_coords[1])
        image_x_values = np.linspace(0,1280,1280)
        image_ylim = 0.02*40000
        plt.errorbar(image_x_values,pixels[:,image_y],yerr=pixels_err[:,image_y])
        plt.ylim([0,image_ylim])
        plt.xlim([image_x-300,image_x+300])
        if distances == True:
            plt.title('Original image with exp: '+str(exp_val)+" at distance: "+str(dist_array[i]))
        else:
            plt.title('Original image with exp: '+str(exp_val))
        plt.show()
    
    for k in range(i+1,max_indices[j]):
        high_arr = pixel_array[k]
        exp_val_high = exp_array[k]
        sc_high_arr = high_arr/exp_val_high*exp_val #This scales the array to the right exposure used
        
        sc_high_arr_err = np.sqrt(((high_arr + pixel_err_array[k])/exp_val_high*exp_val-sc_high_arr)**2 + (high_arr/(exp_val_high+0.005)*exp_val-sc_high_arr)**2 + (high_arr/exp_val_high*(exp_val+0.005)-sc_high_arr)**2)

        pixels, pixels_err, inner_r = avg_arrays(pixels,pixels_err, sc_high_arr,sc_high_arr_err, r, corner_coords, safety_frac)
        
        if plots == True:
            plt.figure()
            plt.plot([image_x-inner_r,image_x-inner_r],[0,saturation_value],color='orange')
            plt.plot([image_x+inner_r,image_x+inner_r],[0,saturation_value],color='orange')
            plt.errorbar(image_x_values,pixels[:,image_y],yerr=pixels_err[:,image_y])
            plt.ylim([0,image_ylim])
            plt.xlim([image_x-300,image_x+300])
            if distances == True:
                plt.title('Added image with exp: '+str(exp_val_high)+" at distance: "+str(dist_array[k]))
            else:
                plt.title('Added image with exp: '+str(exp_val_high))
            plt.show()

    if save_info == True:
        pixels_df = pd.DataFrame(pixels)
        pixels_df.to_csv(new_path+'\\'+name_array[i][:-4]+'.csv')
        
        pixels_err_df = pd.DataFrame(pixels_err)
        pixels_err_df.to_csv(new_path+'\\'+name_array[i][:-4]+'_err.csv')



# Write results
if save_info == True:
    d = {'Distance (mm)':final_dist_array,'Exposure time (us)':final_exp_array,'Radius (pixels)':R_array,'Radius error (pixels)':R_err_array,
         'X offset (pixels)':x_offset_array,'X offset error (pixels)':x_offset_err_array,'Y offset (pixels)':y_offset_array,
         'Y offset error (pixels)':y_offset_err_array}
    dataframe = pd.DataFrame(d)
    dataframe.to_csv(new_path+'\\'+directory_name+'.csv')
